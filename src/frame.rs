//! The camera frame, shared by the tracker (CPU sampling) and renderer (GPU upload).

use glam::Vec2;

/// An RGBA8 frame. RGBA rather than RGB so it uploads to a wgpu texture without
/// a repack, at the cost of one padding byte per pixel.
#[derive(Clone)]
pub struct Frame {
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
    /// Incremented per captured frame so consumers can detect staleness.
    pub seq: u64,
}

impl Frame {
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            rgba: vec![0; (width * height * 4) as usize],
            seq: 0,
        }
    }

    pub fn size(&self) -> Vec2 {
        Vec2::new(self.width as f32, self.height as f32)
    }

    #[inline]
    fn texel(&self, x: i32, y: i32) -> [f32; 3] {
        // Clamp-to-edge: ROIs routinely extend past the frame border.
        let x = x.clamp(0, self.width as i32 - 1) as usize;
        let y = y.clamp(0, self.height as i32 - 1) as usize;
        let i = (y * self.width as usize + x) * 4;
        [
            self.rgba[i] as f32,
            self.rgba[i + 1] as f32,
            self.rgba[i + 2] as f32,
        ]
    }

    /// Bilinear sample at a pixel-space position, returning 0..1 RGB — the
    /// normalization both ONNX models expect.
    pub fn sample(&self, p: Vec2) -> [f32; 3] {
        let px = p.x - 0.5;
        let py = p.y - 0.5;
        let x0 = px.floor();
        let y0 = py.floor();
        let fx = px - x0;
        let fy = py - y0;
        let (x0, y0) = (x0 as i32, y0 as i32);

        let c00 = self.texel(x0, y0);
        let c10 = self.texel(x0 + 1, y0);
        let c01 = self.texel(x0, y0 + 1);
        let c11 = self.texel(x0 + 1, y0 + 1);

        let mut out = [0.0; 3];
        for c in 0..3 {
            let top = c00[c] + (c10[c] - c00[c]) * fx;
            let bot = c01[c] + (c11[c] - c01[c]) * fx;
            out[c] = (top + (bot - top) * fy) / 255.0;
        }
        out
    }
}
