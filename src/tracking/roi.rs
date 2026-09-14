//! Rotated region-of-interest: the bridge between the two models.
//!
//! The palm detector says roughly where a hand is; the landmark model wants a
//! tight, upright, square crop of it. Rather than materializing a rotated
//! image (crop -> warpAffine -> crop, as the OpenCV reference does), we build
//! one rotated rect and sample it directly, which is both cheaper and exactly
//! invertible when mapping landmarks back.

use crate::frame::Frame;
use glam::Vec2;

/// A square, rotated crop window in source-frame pixel coordinates.
#[derive(Debug, Clone, Copy)]
pub struct Roi {
    pub center: Vec2,
    /// Side length in pixels (always square; the models take square input).
    pub side: f32,
    /// Rotation in radians. 0 means the hand already points "up".
    pub angle: f32,
}

impl Roi {
    /// Local axes expressed in image space. `u` is local +x, `v` is local +y
    /// (downward, since image space grows downward).
    pub fn axes(&self) -> (Vec2, Vec2) {
        let (s, c) = self.angle.sin_cos();
        (Vec2::new(c, s), Vec2::new(-s, c))
    }

    /// Build the ROI that encloses `points`, measured in a frame rotated by
    /// `angle`.
    ///
    /// `shift` displaces the center along the local axes in units of the
    /// bounding box size; `enlarge` scales the final square.
    pub fn enclosing(points: &[Vec2], angle: f32, enlarge: f32, shift: Vec2) -> Self {
        let (s, c) = angle.sin_cos();
        let u = Vec2::new(c, s);
        let v = Vec2::new(-s, c);

        let (mut u0, mut u1) = (f32::MAX, f32::MIN);
        let (mut v0, mut v1) = (f32::MAX, f32::MIN);
        for p in points {
            let (pu, pv) = (p.dot(u), p.dot(v));
            u0 = u0.min(pu);
            u1 = u1.max(pu);
            v0 = v0.min(pv);
            v1 = v1.max(pv);
        }

        let (w, h) = (u1 - u0, v1 - v0);
        let cu = (u0 + u1) * 0.5 + shift.x * w;
        let cv = (v0 + v1) * 0.5 + shift.y * h;

        Self {
            // Convert the aligned-space center back into image space.
            center: u * cu + v * cv,
            side: w.max(h) * enlarge,
            angle,
        }
    }


    /// Map an offset in crop pixels (relative to crop center) into an image
    /// offset. Used to place landmarks the model reports in its own space.
    pub fn crop_offset_to_image(&self, offset: Vec2, crop_px: f32) -> Vec2 {
        let (u, v) = self.axes();
        let scaled = offset * (self.side / crop_px);
        u * scaled.x + v * scaled.y
    }

    /// Sample this ROI into a `size x size` NHWC RGB tensor in 0..1.
    pub fn sample_into(&self, frame: &Frame, size: usize, out: &mut Vec<f32>) {
        out.clear();
        out.reserve(size * size * 3);
        let (u, v) = self.axes();
        let step = self.side / size as f32;
        // Position of the crop's (0,0) texel center in image space.
        let origin = self.center - (u + v) * (self.side * 0.5) + (u + v) * (step * 0.5);

        for y in 0..size {
            let row = origin + v * (y as f32 * step);
            for x in 0..size {
                let p = row + u * (x as f32 * step);
                let c = frame.sample(p);
                out.push(c[0]);
                out.push(c[1]);
                out.push(c[2]);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;


    #[test]
    fn quarter_turn_rotates_local_axes() {
        let roi = Roi {
            center: Vec2::ZERO,
            side: 2.0,
            angle: std::f32::consts::FRAC_PI_2,
        };
        let (u, v) = roi.axes();
        // Local +x should point down the image, local +y to the left.
        assert!((u - Vec2::new(0.0, 1.0)).length() < 1e-6, "{u:?}");
        assert!((v - Vec2::new(-1.0, 0.0)).length() < 1e-6, "{v:?}");
    }

    #[test]
    fn enclosing_centers_on_points() {
        let pts = [
            Vec2::new(10.0, 10.0),
            Vec2::new(30.0, 10.0),
            Vec2::new(30.0, 20.0),
            Vec2::new(10.0, 20.0),
        ];
        let roi = Roi::enclosing(&pts, 0.0, 1.0, Vec2::ZERO);
        assert!((roi.center - Vec2::new(20.0, 15.0)).length() < 1e-4, "{:?}", roi.center);
        // Square on the long side.
        assert!((roi.side - 20.0).abs() < 1e-4, "{}", roi.side);
    }

    #[test]
    fn sampling_is_inverse_of_mapping() {
        // A gradient frame lets us verify the sampler hits the right texels.
        let mut f = Frame::new(64, 64);
        for y in 0..64 {
            for x in 0..64 {
                let i = (y * 64 + x) * 4;
                f.rgba[i] = (x * 4) as u8;
                f.rgba[i + 1] = (y * 4) as u8;
            }
        }
        let roi = Roi {
            center: Vec2::new(32.0, 32.0),
            side: 16.0,
            angle: 0.0,
        };
        let mut buf = Vec::new();
        roi.sample_into(&f, 8, &mut buf);
        assert_eq!(buf.len(), 8 * 8 * 3);
        // Values must increase left-to-right in red and top-to-bottom in green.
        let px = |x: usize, y: usize, c: usize| buf[(y * 8 + x) * 3 + c];
        assert!(px(7, 0, 0) > px(0, 0, 0));
        assert!(px(0, 7, 1) > px(0, 0, 1));
    }
}
