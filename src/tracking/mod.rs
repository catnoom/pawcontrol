//! Two-stage hand tracking, MediaPipe style.
//!
//! Stage 1 (palm detection) is expensive and only answers "where are hands?".
//! Stage 2 (landmarks) is cheap and precise but needs a tight crop to work.
//! So we run stage 1 only when we are short of hands, and otherwise re-derive
//! each crop from the previous frame's landmarks.

pub mod filter;
pub mod hand;
pub mod landmark;
pub mod palm;
pub mod roi;

use anyhow::{anyhow, Context, Result};
use glam::{Vec2, Vec3};
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Tensor;

use crate::frame::Frame;
use filter::HandFilter;
use hand::{Hand, HandFrame, LANDMARK_COUNT};
use palm::{Anchors, Letterbox};
use roi::Roi;

const PALM_MODEL: &[u8] = include_bytes!("../../assets/models/palm_detection.onnx");
const LANDMARK_MODEL: &[u8] = include_bytes!("../../assets/models/hand_landmark.onnx");

/// Below this landmark-model confidence we consider the hand lost and drop its
/// ROI, forcing a fresh palm detection.
const PRESENCE_THRESHOLD: f32 = 0.6;
const PALM_SCORE_THRESHOLD: f32 = 0.5;
const PALM_NMS_IOU: f32 = 0.3;

/// How often to re-run palm detection while we are short of hands. Every frame
/// would be wasteful; this still reacquires a hand within ~100ms.
const REDETECT_INTERVAL: u64 = 4;

pub struct TrackerConfig {
    pub max_hands: usize,
    /// Try the GPU (DirectML) first, falling back to CPU automatically.
    pub use_gpu: bool,
}

impl Default for TrackerConfig {
    fn default() -> Self {
        Self {
            max_hands: 2,
            use_gpu: true,
        }
    }
}

/// A hand we are actively tracking between frames.
struct Track {
    roi: Roi,
    filter: HandFilter,
}

pub struct HandTracker {
    palm: Session,
    landmarks: Session,
    anchors: Anchors,
    tracks: Vec<Track>,
    cfg: TrackerConfig,
    /// Scratch buffers, reused every frame to keep the hot loop allocation-free.
    palm_input: Vec<f32>,
    crop_input: Vec<f32>,
    seq: u64,
    /// Which backend actually got used, for the HUD.
    pub backend: &'static str,
}

fn build_session(model: &[u8], use_gpu: bool) -> Result<(Session, &'static str)> {
    if use_gpu {
        match try_build(model, true) {
            Ok(s) => return Ok((s, "DirectML")),
            Err(e) => log::warn!("DirectML unavailable, falling back to CPU: {e:#}"),
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
        .with_intra_threads(2)
        .map_err(|e| anyhow!("thread count: {e}"))?;
    builder
        .commit_from_memory(model)
        .map_err(|e| anyhow!("load model: {e}"))
}

impl HandTracker {
    pub fn new(cfg: TrackerConfig) -> Result<Self> {
        let (palm, backend) =
            build_session(PALM_MODEL, cfg.use_gpu).context("loading palm detection model")?;
        // Keep both models on the same backend so we report one honest answer.
        let (landmarks, _) = build_session(LANDMARK_MODEL, cfg.use_gpu)
            .context("loading hand landmark model")?;

        log::info!("hand tracking running on {backend}");
        Ok(Self {
            palm,
            landmarks,
            anchors: Anchors::new(),
            tracks: Vec::new(),
            cfg,
            palm_input: Vec::new(),
            crop_input: Vec::new(),
            seq: 0,
            backend,
        })
    }

    /// Run one full tracking step. `dt` is seconds since the previous call and
    /// feeds the smoothing filters.
    pub fn track(&mut self, frame: &Frame, dt: f32) -> Result<HandFrame> {
        self.seq += 1;

        let short_of_hands = self.tracks.len() < self.cfg.max_hands;
        if short_of_hands && self.seq % REDETECT_INTERVAL == 0 {
            self.detect_palms(frame)?;
        }

        let mut hands = Vec::with_capacity(self.tracks.len());
        let mut surviving = Vec::with_capacity(self.tracks.len());

        for mut track in std::mem::take(&mut self.tracks) {
            match self.run_landmarks(frame, &track.roi)? {
                Some(mut hand) if hand.score >= PRESENCE_THRESHOLD => {
                    track.filter.apply(&mut hand.landmarks, dt);

                    // Re-seed the next crop from the smoothed landmarks.
                    let px = to_pixels(&hand.landmarks, frame.size());
                    track.roi = landmark::roi_from_landmarks(&px);

                    hands.push(hand);
                    surviving.push(track);
                }
                // Lost: drop the track so palm detection reacquires it.
                _ => {}
            }
        }

        self.tracks = surviving;
        Ok(HandFrame {
            hands,
            seq: self.seq,
        })
    }

    /// Seed new tracks from palm detection, skipping palms we already track.
    fn detect_palms(&mut self, frame: &Frame) -> Result<()> {
        let size = palm::INPUT_SIZE;
        let lb = Letterbox::fit(frame.width as usize, frame.height as usize);

        // Letterbox the frame into the model's square input.
        self.palm_input.clear();
        self.palm_input.resize(size * size * 3, 0.0);
        for y in 0..size {
            for x in 0..size {
                let p = lb.to_pixels(Vec2::new(
                    (x as f32 + 0.5) / size as f32,
                    (y as f32 + 0.5) / size as f32,
                ));
                let i = (y * size + x) * 3;
                // Outside the letterboxed area we leave black padding.
                if p.x < 0.0 || p.y < 0.0 || p.x >= frame.width as f32 || p.y >= frame.height as f32
                {
                    continue;
                }
                let c = frame.sample(p);
                self.palm_input[i] = c[0];
                self.palm_input[i + 1] = c[1];
                self.palm_input[i + 2] = c[2];
            }
        }

        let input = Tensor::from_array((
            [1i64, size as i64, size as i64, 3],
            self.palm_input.clone(),
        ))?;
        let outputs = self.palm.run(ort::inputs!["input_1" => input])?;

        let (_, boxes) = outputs["Identity"].try_extract_tensor::<f32>()?;
        let (_, scores) = outputs["Identity_1"].try_extract_tensor::<f32>()?;

        let dets = palm::decode(boxes, scores, &self.anchors, lb, PALM_SCORE_THRESHOLD);
        let dets = palm::nms(dets, PALM_NMS_IOU, self.cfg.max_hands);

        for det in dets {
            if self.tracks.len() >= self.cfg.max_hands {
                break;
            }
            // Ignore a palm that lands inside a hand we already follow.
            let already_tracked = self.tracks.iter().any(|t| {
                t.roi.center.distance(det.center) < t.roi.side * 0.5
            });
            if already_tracked {
                continue;
            }
            let roi = Roi::enclosing(
                &det.keypoints,
                det.rotation(),
                landmark::PALM_BOX_ENLARGE,
                Vec2::ZERO,
            );
            self.tracks.push(Track {
                roi,
                filter: HandFilter::default(),
            });
        }
        Ok(())
    }

    /// Crop the ROI and run the landmark model over it.
    fn run_landmarks(&mut self, frame: &Frame, roi: &Roi) -> Result<Option<Hand>> {
        let size = landmark::INPUT_SIZE;
        roi.sample_into(frame, size, &mut self.crop_input);

        let input = Tensor::from_array((
            [1i64, size as i64, size as i64, 3],
            self.crop_input.clone(),
        ))?;
        let outputs = self.landmarks.run(ort::inputs!["input_1" => input])?;

        // Output order is fixed by the model: landmarks, confidence,
        // handedness, world landmarks.
        let (_, raw) = outputs["Identity"].try_extract_tensor::<f32>()?;
        let (_, conf) = outputs["Identity_1"].try_extract_tensor::<f32>()?;
        let (_, handed) = outputs["Identity_2"].try_extract_tensor::<f32>()?;

        let score = conf.first().copied().unwrap_or(0.0);
        let handedness = handed.first().copied().unwrap_or(1.0);

        Ok(landmark::decode(raw, score, handedness, roi, frame.size()))
    }
}

/// Timings and shapes from a single forward pass of each model.
#[derive(Debug)]
pub struct ProbeReport {
    pub backend: &'static str,
    pub palm_ms: f32,
    pub landmark_ms: f32,
    pub palm_candidates: usize,
    pub landmark_score: f32,
}

impl HandTracker {
    /// Run one pass of each model against `frame` and report what happened.
    ///
    /// Exists so the whole inference path — model loading, tensor layout,
    /// output names, execution provider — can be verified without a camera.
    pub fn probe(&mut self, frame: &Frame) -> Result<ProbeReport> {
        let t0 = std::time::Instant::now();
        self.detect_palms(frame)?;
        let palm_ms = t0.elapsed().as_secs_f32() * 1000.0;
        let palm_candidates = self.tracks.len();

        // Exercise the landmark model even when no palm was found, using a
        // centred ROI, so a camera-less run still covers both networks.
        let roi = self.tracks.first().map(|t| t.roi).unwrap_or(Roi {
            center: frame.size() * 0.5,
            side: frame.size().y * 0.6,
            angle: 0.0,
        });
        let t1 = std::time::Instant::now();
        let hand = self.run_landmarks(frame, &roi)?;
        let landmark_ms = t1.elapsed().as_secs_f32() * 1000.0;

        Ok(ProbeReport {
            backend: self.backend,
            palm_ms,
            landmark_ms,
            palm_candidates,
            landmark_score: hand.map(|h| h.score).unwrap_or(0.0),
        })
    }
}

/// Landmarks as pixel positions, a conversion several layers want.
pub fn to_pixels(landmarks: &[Vec3; LANDMARK_COUNT], size: Vec2) -> Vec<Vec2> {
    landmarks.iter().map(|p| Vec2::new(p.x, p.y) * size).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// End-to-end check against a real photograph.
    ///
    /// Skipped unless `PAWCONTROL_TEST_IMAGE` points at an image file, so the
    /// normal test run stays hermetic and offline. This is the only test that
    /// exercises palm detection -> ROI -> landmarks as a chain; the unit tests
    /// elsewhere cover each transform in isolation.
    #[test]
    fn tracks_a_hand_in_a_real_photo() {
        let Ok(path) = std::env::var("PAWCONTROL_TEST_IMAGE") else {
            eprintln!("skipping: set PAWCONTROL_TEST_IMAGE to run");
            return;
        };

        let source = image::open(&path).expect("loading test image").to_rgba8();

        // Composite the photo into a webcam-like framing. Palm detection is
        // trained on hands that occupy part of the scene, not ones that fill
        // the entire frame, so a raw close-up is out of distribution.
        let (fw, fh) = (640u32, 480u32);
        let scale = 0.55;
        let tw = (fw as f32 * scale) as u32;
        let th = source.height() * tw / source.width();
        let resized =
            image::imageops::resize(&source, tw, th, image::imageops::FilterType::Triangle);

        let mut frame = Frame::new(fw, fh);
        // Mid-grey background rather than black: less of a hard edge for the
        // detector to latch onto.
        frame.rgba.fill(110);
        let (ox, oy) = ((fw - tw) / 2, fh.saturating_sub(th) / 2);
        for y in 0..th.min(fh - oy) {
            for x in 0..tw {
                let src = resized.get_pixel(x, y).0;
                let di = (((y + oy) * fw + x + ox) * 4) as usize;
                frame.rgba[di..di + 4].copy_from_slice(&src);
            }
        }

        let mut tracker =
            HandTracker::new(TrackerConfig::default()).expect("creating the tracker");

        // Palm detection only runs on some frames, so step until it engages.
        let mut found = None;
        for _ in 0..12 {
            let result = tracker.track(&frame, 1.0 / 30.0).expect("tracking");
            if !result.hands.is_empty() {
                found = Some(result);
                break;
            }
        }

        let result = found.expect("no hand detected in the test image");
        let hand = &result.hands[0];

        // Landmarks must land inside the frame, near the composited photo.
        for (i, p) in hand.landmarks.iter().enumerate() {
            assert!(
                (-0.1..=1.1).contains(&p.x) && (-0.1..=1.1).contains(&p.y),
                "landmark {i} is off-frame: {p:?}"
            );
        }

        // A plausible hand has a non-degenerate palm.
        let scale = hand.scale();
        assert!(
            (0.02..0.9).contains(&scale),
            "implausible palm scale {scale}"
        );

        // The decisive geometry check: on an open palm every fingertip must sit
        // further from the wrist than its own knuckle. A sign error or a bad
        // rotation in the ROI transform collapses this immediately.
        let wrist = hand.wrist();
        let mut extended = 0;
        for finger in hand::Finger::ALL {
            let tip = wrist.distance(hand.point(finger.tip()));
            let mcp = wrist.distance(hand.point(finger.mcp()));
            if tip > mcp {
                extended += 1;
            }
        }
        assert!(
            extended >= 4,
            "only {extended}/5 fingertips extend beyond their knuckles; \
             the ROI transform is probably wrong"
        );
    }
    /// Tracking must hold up at any hand orientation.
    ///
    /// This is the regression test for the ROI rotation transform: a sign
    /// error there still "works" at 0 degrees and collapses off-axis.
    #[test]
    fn rotation_sweep() {
        let Ok(path) = std::env::var("PAWCONTROL_TEST_IMAGE") else {
            eprintln!("skipping: set PAWCONTROL_TEST_IMAGE to run");
            return;
        };
        let source = image::open(&path).expect("load").to_rgba8();

        for deg in [0i32, 30, 60, 90, 120, 150, 180, 240, 300] {
            let (fw, fh) = (640u32, 480u32);
            let mut frame = Frame::new(fw, fh);
            frame.rgba.fill(110);

            // Rotate the photo about the frame centre by sampling backwards.
            let rad = (deg as f32).to_radians();
            let (s, c) = rad.sin_cos();
            let scale = 0.55f32;
            let cx = fw as f32 / 2.0;
            let cy = fh as f32 / 2.0;
            let sw = source.width() as f32;
            let sh = source.height() as f32;
            let fit = (fw as f32 * scale) / sw;

            for y in 0..fh {
                for x in 0..fw {
                    let dx = x as f32 - cx;
                    let dy = y as f32 - cy;
                    let rx = (dx * c + dy * s) / fit + sw / 2.0;
                    let ry = (-dx * s + dy * c) / fit + sh / 2.0;
                    if rx < 0.0 || ry < 0.0 || rx >= sw || ry >= sh { continue; }
                    let p = source.get_pixel(rx as u32, ry as u32).0;
                    let di = ((y * fw + x) * 4) as usize;
                    frame.rgba[di..di + 4].copy_from_slice(&p);
                }
            }

            let mut tracker = HandTracker::new(TrackerConfig::default()).unwrap();
            let mut best = 0.0f32;
            let mut hands = 0;
            for _ in 0..12 {
                let r = tracker.track(&frame, 1.0 / 30.0).unwrap();
                hands = hands.max(r.hands.len());
                for h in &r.hands { best = best.max(h.score); }
            }
            println!("angle {deg:>3}deg -> hands {hands}, best score {best:.3}");
            assert!(hands >= 1, "lost the hand entirely at {deg} degrees");
            assert!(
                best > 0.8,
                "confidence collapsed to {best:.3} at {deg} degrees"
            );
        }
    }

    /// The curl calibration must read an open palm as "extended".
    ///
    /// `Finger::extended_reference` holds per-finger constants measured from a
    /// real hand; this pins them against an actual photograph so a change to
    /// `Hand::scale` or the landmark decode cannot quietly decalibrate the
    /// intensity knob.
    #[test]
    fn open_palm_reads_as_extended() {
        let Ok(path) = std::env::var("PAWCONTROL_TEST_IMAGE") else {
            eprintln!("skipping: set PAWCONTROL_TEST_IMAGE to run");
            return;
        };
        let source = image::open(&path).expect("load").to_rgba8();
        let (fw, fh) = (640u32, 480u32);
        let tw = (fw as f32 * 0.55) as u32;
        let th = source.height() * tw / source.width();
        let resized =
            image::imageops::resize(&source, tw, th, image::imageops::FilterType::Triangle);
        let mut frame = Frame::new(fw, fh);
        frame.rgba.fill(110);
        let (ox, oy) = ((fw - tw) / 2, fh.saturating_sub(th) / 2);
        for y in 0..th.min(fh - oy) {
            for x in 0..tw {
                let px = resized.get_pixel(x, y).0;
                let di = (((y + oy) * fw + x + ox) * 4) as usize;
                frame.rgba[di..di + 4].copy_from_slice(&px);
            }
        }

        let mut tracker = HandTracker::new(TrackerConfig::default()).unwrap();
        for _ in 0..12 {
            let r = tracker.track(&frame, 1.0 / 30.0).unwrap();
            if let Some(h) = r.hands.first() {
                let curl = h.curl(crate::effect::KNOB_FINGER);
                println!("open-palm middle curl = {curl:.3}");
                assert!(
                    curl < 0.3,
                    "an open palm should rest near zero curl, got {curl:.3}; \
                     the knob would sit off-centre at rest"
                );
                return;
            }
        }
        panic!("no hand detected");
    }
}
