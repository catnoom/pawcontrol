//! Effects: what happens inside the region.
//!
//! An effect is a WGSL `effect(uv)` function plus up to four runtime
//! parameters. To add one, drop a `.wgsl` file in `assets/shaders/effects/`
//! and add a line to `registry()`.

use crate::region::Region;
use crate::settings::Tunable;
use crate::tracking::hand::{Finger, HandFrame};

/// The finger that drives the intensity knob.
///
/// Deliberately *not* index or thumb: `TwoHandQuad` builds the region from
/// those two fingertips, so using them would make the knob and the window the
/// same control. Middle, ring and pinky are free.
pub const KNOB_FINGER: Finger = Finger::Middle;

/// Update the intensity knob from the current hands.
///
/// Returns the new value, or `current` unchanged when the region is not live.
/// Holding the value while the window is down means curling a finger between
/// poses cannot silently move the setting: bring the window back up and the
/// effect is exactly where you left it.
pub fn update_knob(current: f32, hands: &HandFrame, region_active: bool) -> f32 {
    if !region_active {
        return current;
    }
    hands.max_curl(KNOB_FINGER).unwrap_or(current)
}

/// What an effect can react to when computing its parameters.
pub struct EffectCtx {
    pub region: Region,
    /// Intensity control, 0..1: 0 with the middle finger extended, 1 fully
    /// curled. Held steady while the region is off — see [`update_knob`].
    pub knob: f32,
}

impl EffectCtx {
    /// Intensity control, 0..1. Curl the middle finger to raise it.
    pub fn knob(&self) -> f32 {
        self.knob.clamp(0.0, 1.0)
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
    /// Knobs this effect offers the control panel. Default: none.
    fn tunables(&mut self) -> Vec<Tunable<'_>> {
        Vec::new()
    }
}

/// Mosaic — the effect from the reference clip.
pub struct Pixelate {
    /// Block size range in pixels, mapped from the pinch knob.
    pub min_block: f32,
    pub max_block: f32,
}

impl Default for Pixelate {
    fn default() -> Self {
        // An open hand rests at the low end, so the default pose already looks
        // like the reference clip; curling dials it chunkier.
        Self {
            min_block: 12.0,
            max_block: 64.0,
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
        let k = ctx.knob();
        [self.min_block + (self.max_block - self.min_block) * k, 0.0, 0.0, 0.0]
    }
    fn tunables(&mut self) -> Vec<Tunable<'_>> {
        vec![
            Tunable::new("block @ knob 0", &mut self.min_block, 1.0, 64.0),
            Tunable::new("block @ knob 1", &mut self.max_block, 1.0, 160.0),
        ]
    }
}

pub struct Blur {
    pub min_radius: f32,
    pub max_radius: f32,
}

impl Default for Blur {
    fn default() -> Self {
        Self {
            min_radius: 2.0,
            max_radius: 10.0,
        }
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
        let k = ctx.knob();
        [self.min_radius + self.max_radius * k, 0.0, 0.0, 0.0]
    }
    fn tunables(&mut self) -> Vec<Tunable<'_>> {
        vec![
            Tunable::new("radius @ knob 0", &mut self.min_radius, 0.0, 20.0),
            Tunable::new("extra radius @ knob 1", &mut self.max_radius, 0.0, 40.0),
        ]
    }
}

pub struct RgbShift {
    pub base: f32,
    pub span: f32,
}

impl Default for RgbShift {
    fn default() -> Self {
        Self {
            base: 3.0,
            span: 33.0,
        }
    }
}

impl Effect for RgbShift {
    fn name(&self) -> &str {
        "rgb-shift"
    }
    fn shader(&self) -> &'static str {
        include_str!("../../assets/shaders/effects/rgb_shift.wgsl")
    }
    fn params(&self, ctx: &EffectCtx) -> [f32; 4] {
        let k = ctx.knob();
        [self.base + self.span * k, 0.0, 0.0, 0.0]
    }
    fn tunables(&mut self) -> Vec<Tunable<'_>> {
        vec![
            Tunable::new("split @ knob 0", &mut self.base, 0.0, 40.0),
            Tunable::new("extra split @ knob 1", &mut self.span, 0.0, 80.0),
        ]
    }
}

pub struct EdgeGlow {
    pub base: f32,
    pub span: f32,
    pub hue_speed: f32,
}

impl Default for EdgeGlow {
    fn default() -> Self {
        Self {
            base: 3.0,
            span: 9.0,
            hue_speed: 1.5,
        }
    }
}

impl Effect for EdgeGlow {
    fn name(&self) -> &str {
        "edge-glow"
    }
    fn shader(&self) -> &'static str {
        include_str!("../../assets/shaders/effects/edge_glow.wgsl")
    }
    fn params(&self, ctx: &EffectCtx) -> [f32; 4] {
        let k = ctx.knob();
        [self.base + self.span * k, self.hue_speed, 0.0, 0.0]
    }
    fn tunables(&mut self) -> Vec<Tunable<'_>> {
        vec![
            Tunable::new("gain @ knob 0", &mut self.base, 0.0, 20.0),
            Tunable::new("extra gain @ knob 1", &mut self.span, 0.0, 30.0),
            Tunable::new("hue speed", &mut self.hue_speed, 0.0, 6.0),
        ]
    }
}

pub struct Swirl {
    pub base: f32,
    pub span: f32,
    pub falloff: f32,
}

impl Default for Swirl {
    fn default() -> Self {
        Self {
            base: 0.6,
            span: 4.4,
            falloff: 0.6,
        }
    }
}

impl Effect for Swirl {
    fn name(&self) -> &str {
        "swirl"
    }
    fn shader(&self) -> &'static str {
        include_str!("../../assets/shaders/effects/swirl.wgsl")
    }
    fn params(&self, ctx: &EffectCtx) -> [f32; 4] {
        let k = ctx.knob();
        // Scale the falloff with the region so the swirl fills whatever
        // window the hands make.
        let extent = ctx.region_extent().max(0.15);
        [self.base + self.span * k, extent * self.falloff, 0.0, 0.0]
    }
    fn tunables(&mut self) -> Vec<Tunable<'_>> {
        vec![
            Tunable::new("twist @ knob 0", &mut self.base, 0.0, 6.0),
            Tunable::new("extra twist @ knob 1", &mut self.span, 0.0, 12.0),
            Tunable::new("falloff x region", &mut self.falloff, 0.1, 2.0),
        ]
    }
}

/// Every effect, in cycle order. Thumb-to-pinky steps through this list.
pub fn registry() -> Vec<Box<dyn Effect>> {
    vec![
        Box::new(Pixelate::default()),
        Box::new(Blur::default()),
        Box::new(RgbShift::default()),
        Box::new(EdgeGlow::default()),
        Box::new(Swirl::default()),
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

    use crate::tracking::hand::{lm, Hand, LANDMARK_COUNT};
    use glam::{Vec2, Vec3};

    /// A hand whose middle fingertip sits `reach` palm-widths from its knuckle.
    fn hand_with_middle_reach(reach: f32) -> Hand {
        let mut landmarks = [Vec3::ZERO; LANDMARK_COUNT];
        // Palm width of 0.1: index/pinky knuckles 0.1 apart.
        landmarks[lm::INDEX_MCP] = Vec3::new(0.5, 0.5, 0.0);
        landmarks[lm::PINKY_MCP] = Vec3::new(0.6, 0.5, 0.0);
        landmarks[lm::WRIST] = Vec3::new(0.55, 0.5, 0.0);
        landmarks[lm::MIDDLE_MCP] = Vec3::new(0.55, 0.5, 0.0);
        // Tip placed straight "up" from the knuckle.
        landmarks[lm::MIDDLE_TIP] = Vec3::new(0.55, 0.5 - reach * 0.1, 0.0);
        Hand {
            landmarks,
            score: 1.0,
        }
    }

    fn frame_with_reach(reach: f32) -> HandFrame {
        HandFrame {
            hands: vec![hand_with_middle_reach(reach)],
        }
    }

    #[test]
    fn extended_middle_finger_reads_as_zero_curl() {
        // 0.88 palm-widths is the measured extended reach.
        let k = update_knob(0.5, &frame_with_reach(0.88), true);
        assert!(k < 0.05, "extended finger should rest at zero, got {k}");
    }

    #[test]
    fn curled_middle_finger_reads_as_full_curl() {
        // 40% of extended is the fully-curled reference.
        let k = update_knob(0.5, &frame_with_reach(0.88 * 0.4), true);
        assert!(k > 0.95, "curled finger should reach one, got {k}");
    }

    #[test]
    fn curl_is_monotonic_between_the_extremes() {
        let mut last = -1.0;
        for step in 0..10 {
            let reach = 0.88 - (step as f32 / 10.0) * 0.5;
            let k = update_knob(0.0, &frame_with_reach(reach), true);
            assert!(k >= last, "knob went backwards at reach {reach}: {k} < {last}");
            last = k;
        }
    }

    #[test]
    fn knob_holds_its_value_while_the_region_is_off() {
        // Curling with no region up must not move the setting.
        let held = update_knob(0.25, &frame_with_reach(0.88 * 0.4), false);
        assert_eq!(held, 0.25, "knob moved while the region was off");

        // ...and resumes tracking the moment the region returns.
        let live = update_knob(0.25, &frame_with_reach(0.88 * 0.4), true);
        assert!(live > 0.95, "knob did not resume: {live}");
    }

    #[test]
    fn knob_holds_when_no_hands_are_tracked() {
        let held = update_knob(0.7, &HandFrame::default(), true);
        assert_eq!(held, 0.7);
    }

    #[test]
    fn knob_is_independent_of_the_region_fingers() {
        // Moving index and thumb (the quad corners) must not disturb the knob.
        let mut a = frame_with_reach(0.6);
        let before = update_knob(0.0, &a, true);
        a.hands[0].landmarks[lm::INDEX_TIP] = Vec3::new(0.9, 0.1, 0.0);
        a.hands[0].landmarks[lm::THUMB_TIP] = Vec3::new(0.1, 0.9, 0.0);
        let after = update_knob(0.0, &a, true);
        assert_eq!(before, after, "quad fingers leaked into the knob");
        let _ = Vec2::ZERO;
    }

    #[test]
    fn params_are_finite_without_hands() {
        let ctx = EffectCtx {
            region: Region::None,
            knob: 0.0,
        };
        for e in registry() {
            for v in e.params(&ctx) {
                assert!(v.is_finite(), "{} produced a non-finite param", e.name());
            }
        }
    }
}
