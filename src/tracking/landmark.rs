//! Hand landmark stage: turns an upright hand crop into 21 positioned joints.

use super::hand::{lm, Hand, Handedness, LANDMARK_COUNT};
use super::roi::Roi;
use glam::{Vec2, Vec3};

/// Square input resolution the landmark model expects.
pub const INPUT_SIZE: usize = 224;

/// Enlargement applied to the palm-keypoint box to get the landmark crop.
pub const PALM_BOX_ENLARGE: f32 = 3.0;
/// Enlargement and shift used to re-derive the ROI from the previous frame's
/// landmarks, which is what makes frame-to-frame tracking stable.
pub const HAND_BOX_ENLARGE: f32 = 1.65;
pub const HAND_BOX_SHIFT: Vec2 = Vec2::new(0.0, -0.1);

/// Decode the landmark model's outputs into a `Hand` in normalized image space.
///
/// `raw` is the 63-value `[x, y, z] * 21` tensor in crop-pixel coordinates,
/// `handedness` is the model's 0=left / 1=right scalar.
pub fn decode(
    raw: &[f32],
    score: f32,
    handedness: f32,
    roi: &Roi,
    frame_size: Vec2,
) -> Option<Hand> {
    if raw.len() < LANDMARK_COUNT * 3 {
        return None;
    }
    let crop = INPUT_SIZE as f32;
    let half = crop * 0.5;
    let mut landmarks = [Vec3::ZERO; LANDMARK_COUNT];

    for (i, out) in landmarks.iter_mut().enumerate() {
        let x = raw[i * 3];
        let y = raw[i * 3 + 1];
        let z = raw[i * 3 + 2];

        // Crop-space offset from the crop center, rotated into image space.
        let offset = Vec2::new(x - half, y - half);
        let px = roi.center + roi.crop_offset_to_image(offset, crop);

        // z shares the landmark model's x/y units, so it scales the same way.
        let depth = z * (roi.side / crop) / frame_size.x;

        *out = Vec3::new(px.x / frame_size.x, px.y / frame_size.y, depth);
    }

    Some(Hand {
        landmarks,
        handedness: if handedness < 0.5 {
            Handedness::Left
        } else {
            Handedness::Right
        },
        score,
    })
}

/// Rotation implied by the wrist -> middle-knuckle vector.
///
/// Matches the palm detector's convention: zero when the hand points up.
pub fn landmark_rotation(points: &[Vec2]) -> f32 {
    let a = points[lm::WRIST];
    let b = points[lm::MIDDLE_MCP];
    std::f32::consts::FRAC_PI_2 - (-(b.y - a.y)).atan2(b.x - a.x)
}

/// Derive the next frame's ROI from this frame's landmarks.
///
/// This is the tracking fast path: as long as it keeps producing confident
/// landmarks we never have to re-run palm detection.
pub fn roi_from_landmarks(points: &[Vec2]) -> Roi {
    let angle = landmark_rotation(points);
    Roi::enclosing(points, angle, HAND_BOX_ENLARGE, HAND_BOX_SHIFT)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn upright_roi() -> Roi {
        Roi {
            center: Vec2::new(320.0, 240.0),
            side: 224.0,
            angle: 0.0,
        }
    }

    #[test]
    fn decode_places_crop_center_at_roi_center() {
        // Every landmark at the crop center must land on the ROI center.
        let raw = vec![112.0, 112.0, 0.0].repeat(LANDMARK_COUNT);
        let frame = Vec2::new(640.0, 480.0);
        let hand = decode(&raw, 0.99, 1.0, &upright_roi(), frame).unwrap();
        let p = hand.point(0);
        assert!((p.x - 0.5).abs() < 1e-5, "{p:?}");
        assert!((p.y - 0.5).abs() < 1e-5, "{p:?}");
        assert_eq!(hand.handedness, Handedness::Right);
    }

    #[test]
    fn decode_scales_with_roi_size() {
        // Same crop offset in a 2x larger ROI must move 2x as far in the image.
        let frame = Vec2::new(640.0, 480.0);
        let mut raw = vec![112.0, 112.0, 0.0].repeat(LANDMARK_COUNT);
        raw[0] = 224.0; // landmark 0 at the right edge of the crop

        let small = decode(&raw, 1.0, 0.0, &upright_roi(), frame).unwrap();
        let big_roi = Roi { side: 448.0, ..upright_roi() };
        let big = decode(&raw, 1.0, 0.0, &big_roi, frame).unwrap();

        let d_small = small.point(0).x - 0.5;
        let d_big = big.point(0).x - 0.5;
        assert!((d_big - d_small * 2.0).abs() < 1e-5, "{d_small} {d_big}");
    }

    #[test]
    fn rotation_is_zero_for_upward_hand() {
        let mut pts = [Vec2::ZERO; LANDMARK_COUNT];
        pts[lm::WRIST] = Vec2::new(100.0, 200.0);
        pts[lm::MIDDLE_MCP] = Vec2::new(100.0, 100.0); // straight up (y is down)
        assert!(landmark_rotation(&pts).abs() < 1e-6);
    }

    #[test]
    fn rotation_is_quarter_turn_for_rightward_hand() {
        let mut pts = [Vec2::ZERO; LANDMARK_COUNT];
        pts[lm::WRIST] = Vec2::new(100.0, 100.0);
        pts[lm::MIDDLE_MCP] = Vec2::new(200.0, 100.0); // pointing right
        let a = landmark_rotation(&pts);
        assert!((a - std::f32::consts::FRAC_PI_2).abs() < 1e-6, "{a}");
    }
}
