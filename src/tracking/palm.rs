//! Palm detection: the "where are the hands at all" stage.
//!
//! The model is an SSD-style detector over a fixed anchor grid. It emits
//! `[1, 2016, 18]` box regressions and `[1, 2016, 1]` score logits, which we
//! decode against locally generated anchors and then NMS.

use glam::Vec2;

/// Square input resolution the palm model expects.
pub const INPUT_SIZE: usize = 192;

const NUM_ANCHORS: usize = 2016;
/// 4 box values (dx, dy, w, h) + 7 palm keypoints x 2.
const VALUES_PER_BOX: usize = 18;
pub const NUM_KEYPOINTS: usize = 7;

/// Palm keypoints we actually use. 0 is the base of the palm and 2 sits at the
/// middle-finger knuckle; the vector between them gives us hand rotation.
const KP_PALM_BASE: usize = 0;
const KP_MIDDLE_MCP: usize = 2;

/// Anchor grid layout, derived from the model's 2016 anchors:
/// stride 8 over a 24x24 map (2 anchors/cell) = 1152, plus stride 16 over a
/// 12x12 map (6 anchors/cell) = 864.
const LAYERS: [(usize, usize); 2] = [(8, 2), (16, 6)];

/// A detected palm in **pixel** coordinates of the source frame.
#[derive(Debug, Clone, Copy)]
pub struct PalmDetection {
    pub score: f32,
    pub center: Vec2,
    pub size: Vec2,
    pub keypoints: [Vec2; NUM_KEYPOINTS],
}

impl PalmDetection {
    /// Rotation of the palm, in radians, from the palm-base -> middle-knuckle
    /// vector. MediaPipe defines the canonical hand as pointing "up", so we
    /// measure the offset from 90 degrees.
    pub fn rotation(&self) -> f32 {
        let a = self.keypoints[KP_PALM_BASE];
        let b = self.keypoints[KP_MIDDLE_MCP];
        // y is flipped because image space grows downward.
        let angle = (-(b.y - a.y)).atan2(b.x - a.x);
        std::f32::consts::FRAC_PI_2 - angle
    }
}

/// Anchor centers in 0..1 model space. Sizes are fixed at 1.0 by the model's
/// training config, so we only need centers.
pub struct Anchors {
    centers: Vec<Vec2>,
}

impl Anchors {
    pub fn new() -> Self {
        let mut centers = Vec::with_capacity(NUM_ANCHORS);
        for (stride, per_cell) in LAYERS {
            let dim = INPUT_SIZE / stride;
            for y in 0..dim {
                for x in 0..dim {
                    // All anchors in a cell share a center; they differ only in
                    // the aspect ratios the model learned to regress from.
                    let c = Vec2::new(
                        (x as f32 + 0.5) / dim as f32,
                        (y as f32 + 0.5) / dim as f32,
                    );
                    for _ in 0..per_cell {
                        centers.push(c);
                    }
                }
            }
        }
        debug_assert_eq!(centers.len(), NUM_ANCHORS);
        Self { centers }
    }

    pub fn len(&self) -> usize {
        self.centers.len()
    }
}

impl Default for Anchors {
    fn default() -> Self {
        Self::new()
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x.clamp(-100.0, 100.0)).exp())
}

/// Maps a point from letterboxed model space back to source-frame pixels.
#[derive(Debug, Clone, Copy)]
pub struct Letterbox {
    pub scale: f32,
    pub pad: Vec2,
}

impl Letterbox {
    /// Fit a `w x h` frame into a square of `INPUT_SIZE`, preserving aspect.
    pub fn fit(w: usize, h: usize) -> Self {
        let scale = INPUT_SIZE as f32 / w.max(h) as f32;
        let pad = Vec2::new(
            (INPUT_SIZE as f32 - w as f32 * scale) * 0.5,
            (INPUT_SIZE as f32 - h as f32 * scale) * 0.5,
        );
        Self { scale, pad }
    }

    /// Model space (0..1 of the padded square) -> source pixels.
    pub fn to_pixels(&self, p: Vec2) -> Vec2 {
        (p * INPUT_SIZE as f32 - self.pad) / self.scale
    }

    /// Lengths carry the scale but not the padding offset.
    pub fn len_to_pixels(&self, v: Vec2) -> Vec2 {
        v * INPUT_SIZE as f32 / self.scale
    }
}

/// Decode raw model tensors into detections, in source-frame pixels.
pub fn decode(
    boxes: &[f32],
    scores: &[f32],
    anchors: &Anchors,
    lb: Letterbox,
    score_threshold: f32,
) -> Vec<PalmDetection> {
    let n = anchors.len().min(scores.len());
    let mut out = Vec::new();

    for i in 0..n {
        let score = sigmoid(scores[i]);
        if score < score_threshold {
            continue;
        }
        let raw = &boxes[i * VALUES_PER_BOX..(i + 1) * VALUES_PER_BOX];
        let anchor = anchors.centers[i];

        // Regressions are in model pixels; anchors are normalized.
        let inv = 1.0 / INPUT_SIZE as f32;
        let center = Vec2::new(raw[0], raw[1]) * inv + anchor;
        let size = Vec2::new(raw[2], raw[3]) * inv;

        let mut keypoints = [Vec2::ZERO; NUM_KEYPOINTS];
        for (k, kp) in keypoints.iter_mut().enumerate() {
            let kx = raw[4 + k * 2] * inv + anchor.x;
            let ky = raw[5 + k * 2] * inv + anchor.y;
            *kp = lb.to_pixels(Vec2::new(kx, ky));
        }

        out.push(PalmDetection {
            score,
            center: lb.to_pixels(center),
            size: lb.len_to_pixels(size),
            keypoints,
        });
    }
    out
}

fn iou(a: &PalmDetection, b: &PalmDetection) -> f32 {
    let (a0, a1) = (a.center - a.size * 0.5, a.center + a.size * 0.5);
    let (b0, b1) = (b.center - b.size * 0.5, b.center + b.size * 0.5);
    let lo = a0.max(b0);
    let hi = a1.min(b1);
    let inter = (hi - lo).max(Vec2::ZERO);
    let inter_area = inter.x * inter.y;
    let union = a.size.x * a.size.y + b.size.x * b.size.y - inter_area;
    if union <= 0.0 {
        0.0
    } else {
        inter_area / union
    }
}

/// Greedy non-maximum suppression, highest score first.
pub fn nms(mut dets: Vec<PalmDetection>, iou_threshold: f32, max_out: usize) -> Vec<PalmDetection> {
    dets.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    let mut kept: Vec<PalmDetection> = Vec::new();
    for d in dets {
        if kept.len() >= max_out {
            break;
        }
        if kept.iter().all(|k| iou(k, &d) < iou_threshold) {
            kept.push(d);
        }
    }
    kept
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn anchor_count_matches_model() {
        assert_eq!(Anchors::new().len(), NUM_ANCHORS);
    }

    #[test]
    fn letterbox_roundtrips_center() {
        let lb = Letterbox::fit(640, 480);
        // Center of the padded square must land on the center of the frame.
        let p = lb.to_pixels(Vec2::splat(0.5));
        assert!((p.x - 320.0).abs() < 0.01, "x was {}", p.x);
        assert!((p.y - 240.0).abs() < 0.01, "y was {}", p.y);
    }

    #[test]
    fn nms_drops_overlapping_boxes() {
        let mk = |score, x| PalmDetection {
            score,
            center: Vec2::new(x, 0.0),
            size: Vec2::splat(10.0),
            keypoints: [Vec2::ZERO; NUM_KEYPOINTS],
        };
        // Two nearly-identical boxes plus one far away.
        let out = nms(vec![mk(0.9, 0.0), mk(0.8, 1.0), mk(0.7, 100.0)], 0.3, 4);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].score, 0.9);
    }
}
