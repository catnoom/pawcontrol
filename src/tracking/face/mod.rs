//! Face tracking, used for eye/blink input.
//!
//! Same two-stage shape as hand tracking: a cheap-to-run detector locates the
//! face, then a landmark model refines it from a tight crop. Unlike the hand
//! models these take **NCHW** input, so the crop is written plane by plane.

pub mod detect;
pub mod eye;
pub mod landmark;

use anyhow::{anyhow, Context, Result};
use glam::Vec2;
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Tensor;

use super::palm::Letterbox;
use super::roi::Roi;
use crate::frame::Frame;
use detect::{Anchors, FaceDetection};

const DETECT_MODEL: &[u8] = include_bytes!("../../../assets/models/face_detection.onnx");
const LANDMARK_MODEL: &[u8] = include_bytes!("../../../assets/models/face_landmark.onnx");

const NMS_IOU: f32 = 0.3;

/// Input scaling for these exports: `value * MUL + ADD` applied to 0..1 RGB.
/// Determined empirically — see `normalization_is_correct`.
const INPUT_MUL: f32 = 1.0;
const INPUT_ADD: f32 = 0.0;

/// One tracked face for one frame.
#[derive(Debug, Clone)]
pub struct Face {
    /// 468 mesh points, normalized to the image (0..1, origin top-left).
    pub landmarks: Vec<Vec2>,
    pub score: f32,
}

#[derive(Debug, Clone, Default)]
pub struct FaceFrame {
    pub faces: Vec<Face>,
}

impl FaceFrame {
    pub fn first(&self) -> Option<&Face> {
        self.faces.first()
    }
}

pub struct FaceTracker {
    detector: Session,
    landmarks: Session,
    anchors: Anchors,
    /// Carried between frames so detection only runs when the face is lost.
    roi: Option<Roi>,
    detect_input: Vec<f32>,
    crop_input: Vec<f32>,
    seq: u64,
}

fn build(model: &[u8], use_gpu: bool) -> Result<(Session, &'static str)> {
    if use_gpu {
        match try_build(model, true) {
            Ok(s) => return Ok((s, "DirectML")),
            Err(e) => log::warn!("DirectML unavailable for face models: {e:#}"),
        }
    }
    Ok((try_build(model, false)?, "CPU"))
}

fn try_build(model: &[u8], gpu: bool) -> Result<Session> {
    let mut builder = Session::builder().map_err(|e| anyhow!("session builder: {e}"))?;
    if gpu {
        builder = builder
            .with_execution_providers([ort::ep::DirectML::default().build()])
            .map_err(|e| anyhow!("register DirectML: {e}"))?;
    }
    builder = builder
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(|e| anyhow!("optimization level: {e}"))?
        .with_intra_threads(1)
        .map_err(|e| anyhow!("thread count: {e}"))?;
    builder
        .commit_from_memory(model)
        .map_err(|e| anyhow!("load model: {e}"))
}

/// Settings the face stage reads each frame.
#[derive(Debug, Clone, Copy)]
pub struct FaceSettings {
    pub enabled: bool,
    pub detect_threshold: f32,
    /// Landmark confidence below which the face is considered lost.
    pub presence_threshold: f32,
    /// Frames between detection attempts while no face is tracked.
    pub redetect_interval: u64,
    /// How much to enlarge the mesh's own bounding box when re-deriving the
    /// crop for the next frame. Too tight and the mesh model degrades, which
    /// shows up as tracking that flickers once per detection interval.
    pub track_roi_scale: f32,
}

impl Default for FaceSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            detect_threshold: 0.6,
            presence_threshold: 0.4,
            redetect_interval: 6,
            track_roi_scale: 1.4,
        }
    }
}

impl FaceTracker {
    pub fn new(use_gpu: bool) -> Result<Self> {
        let (detector, backend) =
            build(DETECT_MODEL, use_gpu).context("loading face detection model")?;
        let (landmarks, _) = build(LANDMARK_MODEL, use_gpu).context("loading face mesh model")?;
        log::info!("face tracking running on {backend}");
        Ok(Self {
            detector,
            landmarks,
            anchors: Anchors::new(),
            roi: None,
            detect_input: Vec::new(),
            crop_input: Vec::new(),
            seq: 0,
        })
    }

    pub fn track(&mut self, frame: &Frame, settings: &FaceSettings) -> Result<FaceFrame> {
        if !settings.enabled {
            self.roi = None;
            return Ok(FaceFrame::default());
        }
        self.seq += 1;

        if self.roi.is_none() && self.seq % settings.redetect_interval.max(1) == 0 {
            self.roi = self.detect(frame, settings)?;
        }

        let Some(roi) = self.roi else {
            return Ok(FaceFrame::default());
        };

        match self.run_landmarks(frame, &roi)? {
            Some((points, score)) if score >= settings.presence_threshold => {
                // Re-seed the crop from the mesh so detection can stay idle.
                self.roi = Some(Roi::enclosing(
                    &points,
                    mesh_rotation(&points),
                    settings.track_roi_scale,
                    Vec2::ZERO,
                ));
                let size = frame.size();
                Ok(FaceFrame {
                    faces: vec![Face {
                        landmarks: points.iter().map(|p| *p / size).collect(),
                        score,
                    }],
                })
            }
            _ => {
                self.roi = None;
                Ok(FaceFrame::default())
            }
        }
    }

    fn detect(&mut self, frame: &Frame, settings: &FaceSettings) -> Result<Option<Roi>> {
        let size = detect::INPUT_SIZE;
        let lb = Letterbox::fit_to(frame.width as usize, frame.height as usize, size);
        write_planar_letterbox(frame, lb, size, INPUT_MUL, INPUT_ADD, &mut self.detect_input);

        let input = Tensor::from_array((
            [1i64, 3, size as i64, size as i64],
            self.detect_input.clone(),
        ))?;
        let outputs = self.detector.run(ort::inputs!["image" => input])?;

        let (_, boxes1) = outputs["box_coords_1"].try_extract_tensor::<f32>()?;
        let (_, scores1) = outputs["box_scores_1"].try_extract_tensor::<f32>()?;
        let (_, boxes2) = outputs["box_coords_2"].try_extract_tensor::<f32>()?;
        let (_, scores2) = outputs["box_scores_2"].try_extract_tensor::<f32>()?;

        let mut dets: Vec<FaceDetection> = Vec::new();
        detect::decode_head(
            boxes1,
            scores1,
            self.anchors.head(0),
            lb,
            settings.detect_threshold,
            &mut dets,
        );
        detect::decode_head(
            boxes2,
            scores2,
            self.anchors.head(1),
            lb,
            settings.detect_threshold,
            &mut dets,
        );

        let dets = super::nms::nms(dets, NMS_IOU, 1);
        Ok(dets.first().map(|d| {
            let half = d.size * 0.5;
            let corners = [d.center - half, d.center + half];
            Roi::enclosing(
                &corners,
                d.rotation(),
                landmark::FACE_BOX_ENLARGE,
                Vec2::ZERO,
            )
        }))
    }

    fn run_landmarks(&mut self, frame: &Frame, roi: &Roi) -> Result<Option<(Vec<Vec2>, f32)>> {
        let size = landmark::INPUT_SIZE;
        roi.sample_into_planar(frame, size, &mut self.crop_input);

        let input = Tensor::from_array((
            [1i64, 3, size as i64, size as i64],
            self.crop_input.clone(),
        ))?;
        let outputs = self.landmarks.run(ort::inputs!["image" => input])?;

        let (_, raw) = outputs["landmarks"].try_extract_tensor::<f32>()?;
        let (_, score) = outputs["scores"].try_extract_tensor::<f32>()?;
        let score = score.first().copied().unwrap_or(0.0);

        Ok(landmark::decode(raw, roi).map(|pts| (pts, score)))
    }
}

/// Roll of the mesh, from the outer eye corners.
///
/// Returned in `Roi`'s convention: the rect's local +x axis runs along the
/// eye line, so the crop comes out upright.
fn mesh_rotation(points: &[Vec2]) -> f32 {
    const RIGHT_OUTER: usize = 33;
    const LEFT_OUTER: usize = 263;
    if points.len() <= LEFT_OUTER {
        return 0.0;
    }
    let d = points[LEFT_OUTER] - points[RIGHT_OUTER];
    d.y.atan2(d.x)
}

/// Letterbox the frame into a square NCHW buffer.
fn write_planar_letterbox(
    frame: &Frame,
    lb: Letterbox,
    size: usize,
    mul: f32,
    add: f32,
    out: &mut Vec<f32>,
) {
    out.clear();
    out.resize(size * size * 3, 0.0);
    let plane = size * size;
    for y in 0..size {
        for x in 0..size {
            let p = lb.to_pixels(Vec2::new(
                (x as f32 + 0.5) / size as f32,
                (y as f32 + 0.5) / size as f32,
            ));
            if p.x < 0.0 || p.y < 0.0 || p.x >= frame.width as f32 || p.y >= frame.height as f32 {
                continue;
            }
            let c = frame.sample(p);
            let i = y * size + x;
            out[i] = c[0] * mul + add;
            out[plane + i] = c[1] * mul + add;
            out[2 * plane + i] = c[2] * mul + add;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn face_frame() -> Option<Frame> {
        let path = std::env::var("PAWCONTROL_FACE_IMAGE").ok()?;
        if !std::path::Path::new(&path).is_file() {
            eprintln!("skipping: PAWCONTROL_FACE_IMAGE={path} does not exist");
            return None;
        }
        let source = image::open(&path).expect("loading face image").to_rgba8();
        // Composite into a webcam-like framing: the detector expects a face
        // occupying part of the scene, not filling it.
        let (fw, fh) = (640u32, 480u32);
        let tw = (fw as f32 * 0.5) as u32;
        let th = source.height() * tw / source.width();
        let resized =
            image::imageops::resize(&source, tw, th, image::imageops::FilterType::Triangle);
        let mut frame = Frame::new(fw, fh);
        frame.rgba.fill(110);
        let (ox, oy) = ((fw - tw) / 2, (fh.saturating_sub(th)) / 2);
        for y in 0..th.min(fh - oy) {
            for x in 0..tw {
                let px = resized.get_pixel(x, y).0;
                let di = (((y + oy) * fw + x + ox) * 4) as usize;
                frame.rgba[di..di + 4].copy_from_slice(&px);
            }
        }
        Some(frame)
    }

    /// The full chain on a real photograph: detect -> crop -> mesh -> EAR.
    ///
    /// This is the only test covering the face stages together; a sign error
    /// in the crop transform still "runs" and produces a mesh, but one that
    /// does not sit on the face — which this catches via the eye readings.
    #[test]
    fn tracks_a_face_and_reads_open_eyes() {
        let _gpu = crate::test_support::gpu_lock();
        let Some(frame) = face_frame() else { return };

        let mut tracker = FaceTracker::new(true).expect("face models");
        let settings = FaceSettings::default();

        let mut found = None;
        for _ in 0..12 {
            let result = tracker.track(&frame, &settings).expect("tracking");
            if !result.faces.is_empty() {
                found = Some(result);
                break;
            }
        }
        let result = found.expect("no face detected in the test image");
        let face = result.first().unwrap();
        assert_eq!(face.landmarks.len(), landmark::LANDMARK_COUNT);

        // The mesh must land inside the frame.
        for (i, p) in face.landmarks.iter().enumerate() {
            assert!(
                (-0.1..=1.1).contains(&p.x) && (-0.1..=1.1).contains(&p.y),
                "landmark {i} off-frame: {p:?}"
            );
        }

        // Both eyes are open in the fixture, so both EARs should sit in the
        // open range. A misplaced mesh produces a degenerate ratio instead.
        for eye in [eye::Eye::Right, eye::Eye::Left] {
            let ear = eye::aspect_ratio(&face.landmarks, eye).expect("ear");
            println!("{eye:?} eye EAR = {ear:.3}");
            assert!(
                (0.12..0.60).contains(&ear),
                "{eye:?} eye EAR {ear:.3} is not a plausible open eye; \
                 the crop transform is probably wrong"
            );
        }

        // The eyes must sit level on a frontal face, pinning orientation.
        let right = face.landmarks[eye::lm::RIGHT_EYE[0]];
        let left = face.landmarks[eye::lm::LEFT_EYE[3]];
        assert!(
            (right.y - left.y).abs() < 0.08,
            "eyes are not level on a frontal face: {right:?} {left:?}"
        );

        // Absolute scale. The eye tests above are ratios, so they hold even if
        // the whole mesh collapses to a dot — which is exactly the bug that
        // slipped through once. These check the mesh is actually face-sized.
        let eye_gap = right.distance(left);
        assert!(
            (0.05..0.60).contains(&eye_gap),
            "inter-eye distance {eye_gap:.4} of frame width is not face-sized; \
             the mesh is probably mis-scaled"
        );

        let min_x = face.landmarks.iter().map(|p| p.x).fold(f32::MAX, f32::min);
        let max_x = face.landmarks.iter().map(|p| p.x).fold(f32::MIN, f32::max);
        let min_y = face.landmarks.iter().map(|p| p.y).fold(f32::MAX, f32::min);
        let max_y = face.landmarks.iter().map(|p| p.y).fold(f32::MIN, f32::max);
        let (w, h) = (max_x - min_x, max_y - min_y);
        println!("mesh spans {w:.3} x {h:.3} of the frame, eye gap {eye_gap:.3}");
        assert!(
            (0.08..0.95).contains(&w) && (0.08..0.95).contains(&h),
            "mesh spans {w:.3}x{h:.3} of the frame, which is not a face"
        );
    }

    /// Tracking must persist frame to frame, not flicker.
    ///
    /// The regression this guards: the mesh model emits *normalized* crop
    /// coordinates, and decoding them as crop pixels collapsed every landmark
    /// to a 2px blob. The next crop was then derived from that blob, so the
    /// face was lost immediately and only reappeared when detection next ran
    /// — one good frame per detection interval.
    #[test]
    fn tracking_persists_across_frames() {
        let _gpu = crate::test_support::gpu_lock();
        let Some(frame) = face_frame() else { return };

        let mut tracker = FaceTracker::new(true).expect("face models");
        let settings = FaceSettings::default();

        // The same still frame every time, so any loss is our own doing.
        let mut pattern = String::new();
        for _ in 0..30 {
            let r = tracker.track(&frame, &settings).expect("tracking");
            pattern.push(if r.first().is_some() { '#' } else { '.' });
        }
        let tracked = pattern.matches('#').count();
        println!("pattern: {pattern} ({tracked}/30)");

        // Detection only runs every `redetect_interval` frames, so the opening
        // gap is expected; everything after it must be continuous.
        assert!(
            tracked >= 24,
            "tracking flickered: only {tracked}/30 frames held a face ({pattern})"
        );
        assert!(
            !pattern.trim_start_matches('.').contains('.'),
            "tracking dropped after acquiring: {pattern}"
        );
    }

    /// The input scaling we settled on must still detect a face.
    ///
    /// These exports carry their own preprocessing, so 0..1 and -1..1 both
    /// work; this pins the one actually in use rather than leaving it to
    /// chance if the model is ever swapped.
    #[test]
    fn configured_normalization_detects() {
        let _gpu = crate::test_support::gpu_lock();
        let Some(frame) = face_frame() else { return };

        let mut tracker = FaceTracker::new(true).expect("face models");
        let settings = FaceSettings::default();
        let roi = tracker
            .detect(&frame, &settings)
            .expect("detect")
            .expect("a face at the configured input scaling");
        // A face in this framing occupies a good fraction of the frame.
        assert!(
            (100.0..500.0).contains(&roi.side),
            "implausible face crop: {roi:?}"
        );
    }
}
