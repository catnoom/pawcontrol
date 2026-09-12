//! Effects: what happens inside the region.
//!
//! An effect is a WGSL `effect(uv)` function plus up to four runtime
//! parameters. To add one, drop a `.wgsl` file in `assets/shaders/effects/`
//! and add a line to `registry()`.

use crate::region::Region;
use crate::tracking::hand::{Finger, HandFrame};

/// What an effect can react to when computing its parameters.
#[allow(dead_code)] // `time` is available to effects that need animation
pub struct EffectCtx<'a> {
    pub hands: &'a HandFrame,
    pub region: Region,
    pub time: f32,
}

#[allow(dead_code)]
impl EffectCtx<'_> {
    /// A 0..1 control value taken from the first hand's thumb-index pinch.
    ///
    /// This is the generic "knob" gesture: spread the pinch to dial the
    /// current effect up, close it to dial it down.
    pub fn pinch_knob(&self) -> Option<f32> {
        let hand = self.hands.hands.first()?;
        let d = hand.pinch_distance(Finger::Thumb, Finger::Index);
        // 0.3..1.2 palm-widths maps to the full range; outside that it clamps.
        Some(((d - 0.3) / 0.9).clamp(0.0, 1.0))
    }

    /// Diagonal of the region in normalized units, for size-aware effects.
    pub fn region_extent(&self) -> f32 {
        match self.region.corners() {
            Some(c) => (c[2] - c[0]).length().max(1e-4),
            None => 0.0,
        }
    }
}

pub trait Effect: Send {
    fn name(&self) -> &str;
    /// WGSL defining `fn effect(uv: vec2<f32>) -> vec3<f32>`.
    fn shader(&self) -> &'static str;
    /// Values exposed to the shader as `g.params`.
    fn params(&self, ctx: &EffectCtx) -> [f32; 4];
}

/// Mosaic — the effect from the reference clip.
pub struct Pixelate {
    /// Block size range in pixels, mapped from the pinch knob.
    pub min_block: f32,
    pub max_block: f32,
}

impl Default for Pixelate {
    fn default() -> Self {
        Self {
            min_block: 6.0,
            max_block: 48.0,
        }
    }
}

impl Effect for Pixelate {
    fn name(&self) -> &str {
        "pixelate"
    }
    fn shader(&self) -> &'static str {
        include_str!("../../assets/shaders/effects/pixelate.wgsl")
    }
    fn params(&self, ctx: &EffectCtx) -> [f32; 4] {
        let k = ctx.pinch_knob().unwrap_or(0.5);
        [self.min_block + (self.max_block - self.min_block) * k, 0.0, 0.0, 0.0]
    }
}

pub struct Blur {
    pub max_radius: f32,
}

impl Default for Blur {
    fn default() -> Self {
        Self { max_radius: 6.0 }
    }
}

impl Effect for Blur {
    fn name(&self) -> &str {
        "blur"
    }
    fn shader(&self) -> &'static str {
        include_str!("../../assets/shaders/effects/blur.wgsl")
    }
    fn params(&self, ctx: &EffectCtx) -> [f32; 4] {
        let k = ctx.pinch_knob().unwrap_or(0.5);
        [1.0 + self.max_radius * k, 0.0, 0.0, 0.0]
    }
}

#[derive(Default)]
pub struct RgbShift;

impl Effect for RgbShift {
    fn name(&self) -> &str {
        "rgb-shift"
    }
    fn shader(&self) -> &'static str {
        include_str!("../../assets/shaders/effects/rgb_shift.wgsl")
    }
    fn params(&self, ctx: &EffectCtx) -> [f32; 4] {
        let k = ctx.pinch_knob().unwrap_or(0.5);
        [2.0 + 30.0 * k, 0.0, 0.0, 0.0]
    }
}

#[derive(Default)]
pub struct EdgeGlow;

impl Effect for EdgeGlow {
    fn name(&self) -> &str {
        "edge-glow"
    }
    fn shader(&self) -> &'static str {
        include_str!("../../assets/shaders/effects/edge_glow.wgsl")
    }
    fn params(&self, ctx: &EffectCtx) -> [f32; 4] {
        let k = ctx.pinch_knob().unwrap_or(0.5);
        [2.0 + 8.0 * k, 1.5, 0.0, 0.0]
    }
}

#[derive(Default)]
pub struct Swirl;

impl Effect for Swirl {
    fn name(&self) -> &str {
        "swirl"
    }
    fn shader(&self) -> &'static str {
        include_str!("../../assets/shaders/effects/swirl.wgsl")
    }
    fn params(&self, ctx: &EffectCtx) -> [f32; 4] {
        let k = ctx.pinch_knob().unwrap_or(0.5);
        // Scale the falloff with the region so the swirl fills whatever
        // window the hands make.
        let extent = ctx.region_extent().max(0.15);
        [0.5 + 4.0 * k, extent * 0.6, 0.0, 0.0]
    }
}

/// Every effect, in cycle order. Thumb-to-pinky steps through this list.
pub fn registry() -> Vec<Box<dyn Effect>> {
    vec![
        Box::new(Pixelate::default()),
        Box::new(Blur::default()),
        Box::new(RgbShift),
        Box::new(EdgeGlow),
        Box::new(Swirl),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_effect_defines_the_entry_point() {
        for e in registry() {
            assert!(
                e.shader().contains("fn effect(uv: vec2<f32>) -> vec3<f32>"),
                "{} is missing the effect() entry point",
                e.name()
            );
        }
    }

    #[test]
    fn effect_names_are_unique() {
        let reg = registry();
        let mut names: Vec<&str> = reg.iter().map(|e| e.name()).collect();
        names.sort_unstable();
        let before = names.len();
        names.dedup();
        assert_eq!(before, names.len(), "duplicate effect names: {names:?}");
    }

    #[test]
    fn params_are_finite_without_hands() {
        let hands = HandFrame::default();
        let ctx = EffectCtx {
            hands: &hands,
            region: Region::None,
            time: 0.0,
        };
        for e in registry() {
            for v in e.params(&ctx) {
                assert!(v.is_finite(), "{} produced a non-finite param", e.name());
            }
        }
    }
}
