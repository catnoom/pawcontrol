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
/// `raw` is `[x, y, z] * 468` in **normalized** crop coordinates: 0..1 across
/// the crop, not crop pixels. (The hand landmark model emits pixels, so the
/// two decoders differ here.)
pub fn decode(raw: &[f32], roi: &Roi) -> Option<Vec<Vec2>> {
    if raw.len() < LANDMARK_COUNT * 3 {
        return None;
    }
    let crop = INPUT_SIZE as f32;

    let mut out = Vec::with_capacity(LANDMARK_COUNT);
    for i in 0..LANDMARK_COUNT {
        // Normalized -> crop pixels relative to the crop centre.
        let offset = Vec2::new(raw[i * 3] - 0.5, raw[i * 3 + 1] - 0.5) * crop;
        out.push(roi.center + roi.crop_offset_to_image(offset, crop));
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn roi() -> Roi {
        Roi {
            center: Vec2::new(320.0, 240.0),
            side: 192.0,
            angle: 0.0,
        }
    }

    #[test]
    fn crop_centre_maps_to_roi_centre() {
        // 0.5 is the centre in normalized crop space.
        let raw = vec![0.5, 0.5, 0.0].repeat(LANDMARK_COUNT);
        let pts = decode(&raw, &roi()).unwrap();
        assert_eq!(pts.len(), LANDMARK_COUNT);
        assert!((pts[0] - roi().center).length() < 1e-4, "{:?}", pts[0]);
    }

    #[test]
    fn mesh_spans_the_crop_at_full_scale() {
        // A point at the crop's right edge must land half a side away, not a
        // fraction of a pixel: this is the scale the ratio-based eye tests
        // cannot see.
        let mut raw = vec![0.5, 0.5, 0.0].repeat(LANDMARK_COUNT);
        raw[0] = 1.0; // landmark 0 at the right edge
        raw[1] = 0.0; // and the top edge
        let pts = decode(&raw, &roi()).unwrap();
        let r = roi();
        assert!(
            (pts[0].x - (r.center.x + r.side * 0.5)).abs() < 1e-3,
            "x landed at {} not {}",
            pts[0].x,
            r.center.x + r.side * 0.5
        );
        assert!(
            (pts[0].y - (r.center.y - r.side * 0.5)).abs() < 1e-3,
            "y landed at {}",
            pts[0].y
        );
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
