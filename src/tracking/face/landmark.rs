//! Face landmark stage: 468 mesh points from an upright face crop.

use super::super::roi::Roi;
use glam::Vec2;

/// Square input resolution the landmark model expects.
pub const INPUT_SIZE: usize = 192;
pub const LANDMARK_COUNT: usize = 468;

/// How much to enlarge the detector's box to frame the landmark crop.
///
/// The detector box hugs the face; the mesh model expects some margin around
/// it, including forehead and chin.
pub const FACE_BOX_ENLARGE: f32 = 1.5;

/// Decode the mesh into image-space pixels.
///
/// `raw` is `[x, y, z] * 468` in crop-pixel coordinates.
pub fn decode(raw: &[f32], roi: &Roi) -> Option<Vec<Vec2>> {
    if raw.len() < LANDMARK_COUNT * 3 {
        return None;
    }
    let crop = INPUT_SIZE as f32;
    let half = crop * 0.5;

    let mut out = Vec::with_capacity(LANDMARK_COUNT);
    for i in 0..LANDMARK_COUNT {
        let offset = Vec2::new(raw[i * 3] - half, raw[i * 3 + 1] - half);
        out.push(roi.center + roi.crop_offset_to_image(offset, crop));
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crop_centre_maps_to_roi_centre() {
        let roi = Roi {
            center: Vec2::new(320.0, 240.0),
            side: 192.0,
            angle: 0.0,
        };
        let raw = vec![96.0, 96.0, 0.0].repeat(LANDMARK_COUNT);
        let pts = decode(&raw, &roi).unwrap();
        assert_eq!(pts.len(), LANDMARK_COUNT);
        assert!((pts[0] - roi.center).length() < 1e-4, "{:?}", pts[0]);
    }

    #[test]
    fn short_tensor_is_rejected() {
        let roi = Roi {
            center: Vec2::ZERO,
            side: 1.0,
            angle: 0.0,
        };
        assert!(decode(&[0.0; 30], &roi).is_none());
    }
}
