//! Face detection: a BlazeFace-style SSD over a fixed anchor grid.
//!
//! The model emits two heads rather than one: `[1, 512, 16]` from a stride-16
//! grid and `[1, 384, 16]` from a stride-32 grid, 896 anchors in total. Each
//! row is 4 box values plus 6 keypoints (eyes, nose, mouth, ears) as x/y pairs.

use super::super::nms::Detection;
use glam::Vec2;

/// Square input resolution the detector expects.
pub const INPUT_SIZE: usize = 256;

const VALUES_PER_BOX: usize = 16;
pub const NUM_KEYPOINTS: usize = 6;

/// Keypoint order is fixed by the model.
pub const KP_RIGHT_EYE: usize = 0;
pub const KP_LEFT_EYE: usize = 1;

/// (stride, anchors per cell) per output head, in head order.
///
/// Derived from the output shapes: at 256px, stride 16 gives a 16x16 grid,
/// which at 2 anchors per cell is the 512 rows of the first head; stride 32
/// gives 8x8, which at 6 per cell is the 384 of the second.
pub const HEADS: [(usize, usize); 2] = [(16, 2), (32, 6)];

#[derive(Debug, Clone, Copy)]
pub struct FaceDetection {
    pub score: f32,
    pub center: Vec2,
    pub size: Vec2,
    pub keypoints: [Vec2; NUM_KEYPOINTS],
}

impl Detection for FaceDetection {
    fn score(&self) -> f32 {
        self.score
    }
    fn center(&self) -> Vec2 {
        self.center
    }
    fn size(&self) -> Vec2 {
        self.size
    }
}

impl FaceDetection {
    /// Roll of the face, from the vector between the eye keypoints.
    ///
    /// Zero when the eyes are level. Used to crop an upright face for the
    /// landmark model, which is trained on upright faces.
    pub fn rotation(&self) -> f32 {
        let right = self.keypoints[KP_RIGHT_EYE];
        let left = self.keypoints[KP_LEFT_EYE];
        let d = left - right;
        // Image y grows downward, so negate to get a conventional angle.
        (-d.y).atan2(d.x).neg_zero()
    }
}

trait NegZero {
    fn neg_zero(self) -> Self;
}
impl NegZero for f32 {
    /// Normalizes -0.0 to 0.0 so comparisons in tests behave.
    fn neg_zero(self) -> Self {
        if self == 0.0 {
            0.0
        } else {
            self
        }
    }
}

/// Anchor centers for one head, in 0..1 model space.
pub struct Anchors {
    heads: Vec<Vec<Vec2>>,
}

impl Anchors {
    pub fn new() -> Self {
        let heads = HEADS
            .iter()
            .map(|(stride, per_cell)| {
                let dim = INPUT_SIZE / stride;
                let mut centers = Vec::with_capacity(dim * dim * per_cell);
                for y in 0..dim {
                    for x in 0..dim {
                        let c = Vec2::new(
                            (x as f32 + 0.5) / dim as f32,
                            (y as f32 + 0.5) / dim as f32,
                        );
                        for _ in 0..*per_cell {
                            centers.push(c);
                        }
                    }
                }
                centers
            })
            .collect();
        Self { heads }
    }

    pub fn head(&self, index: usize) -> &[Vec2] {
        &self.heads[index]
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

/// Decode one output head into detections, in source-frame pixels.
pub fn decode_head(
    boxes: &[f32],
    scores: &[f32],
    anchors: &[Vec2],
    lb: super::super::palm::Letterbox,
    score_threshold: f32,
    out: &mut Vec<FaceDetection>,
) {
    let n = anchors.len().min(scores.len());
    let inv = 1.0 / INPUT_SIZE as f32;

    for i in 0..n {
        let score = sigmoid(scores[i]);
        if score < score_threshold {
            continue;
        }
        let raw = &boxes[i * VALUES_PER_BOX..(i + 1) * VALUES_PER_BOX];
        let anchor = anchors[i];

        let center = Vec2::new(raw[0], raw[1]) * inv + anchor;
        let size = Vec2::new(raw[2], raw[3]) * inv;

        let mut keypoints = [Vec2::ZERO; NUM_KEYPOINTS];
        for (k, kp) in keypoints.iter_mut().enumerate() {
            let x = raw[4 + k * 2] * inv + anchor.x;
            let y = raw[5 + k * 2] * inv + anchor.y;
            *kp = lb.to_pixels(Vec2::new(x, y));
        }

        out.push(FaceDetection {
            score,
            center: lb.to_pixels(center),
            size: lb.len_to_pixels(size),
            keypoints,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn anchor_counts_match_the_two_heads() {
        let a = Anchors::new();
        assert_eq!(a.head(0).len(), 512, "stride-16 head");
        assert_eq!(a.head(1).len(), 384, "stride-32 head");
    }

    #[test]
    fn rotation_is_zero_for_level_eyes() {
        let mut keypoints = [Vec2::ZERO; NUM_KEYPOINTS];
        keypoints[KP_RIGHT_EYE] = Vec2::new(100.0, 50.0);
        keypoints[KP_LEFT_EYE] = Vec2::new(140.0, 50.0);
        let d = FaceDetection {
            score: 1.0,
            center: Vec2::new(120.0, 60.0),
            size: Vec2::splat(80.0),
            keypoints,
        };
        assert!(d.rotation().abs() < 1e-6, "{}", d.rotation());
    }

    #[test]
    fn rotation_follows_a_tilted_head() {
        // Left eye lower on screen => head rolled clockwise => negative angle.
        let mut keypoints = [Vec2::ZERO; NUM_KEYPOINTS];
        keypoints[KP_RIGHT_EYE] = Vec2::new(100.0, 50.0);
        keypoints[KP_LEFT_EYE] = Vec2::new(140.0, 90.0);
        let d = FaceDetection {
            score: 1.0,
            center: Vec2::new(120.0, 70.0),
            size: Vec2::splat(80.0),
            keypoints,
        };
        let expected = -std::f32::consts::FRAC_PI_4;
        assert!((d.rotation() - expected).abs() < 1e-5, "{}", d.rotation());
    }
}
