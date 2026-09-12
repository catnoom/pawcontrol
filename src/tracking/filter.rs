//! One Euro filter — the difference between a usable effect and a jittery one.
//!
//! Landmark models are stable to within a pixel or two, but at 2.6x ROI
//! magnification that noise becomes very visible on a quad drawn between
//! fingertips. A plain low-pass would add lag when the hand moves fast, so we
//! use the One Euro filter: it smooths hard when the hand is still and barely
//! at all when it is moving.
//!
//! Casiez, Roussel & Vogel, CHI 2012.

use glam::Vec3;

#[derive(Debug, Clone, Copy)]
pub struct OneEuroConfig {
    /// Cutoff at zero velocity; lower means smoother but laggier when still.
    pub min_cutoff: f32,
    /// How aggressively the cutoff opens up with speed.
    pub beta: f32,
    /// Cutoff for the velocity estimate itself.
    pub derivative_cutoff: f32,
}

impl Default for OneEuroConfig {
    fn default() -> Self {
        // Tuned for normalized (0..1) landmark coordinates at ~30-60 Hz.
        //
        // `beta` has to be large here because hand speeds in normalized units
        // are small numbers (a brisk move is ~1.0/sec). With a small beta the
        // cutoff barely opens and the hand visibly lags behind the effect.
        Self {
            min_cutoff: 1.0,
            beta: 20.0,
            derivative_cutoff: 1.0,
        }
    }
}

fn alpha(cutoff: f32, dt: f32) -> f32 {
    let tau = 1.0 / (2.0 * std::f32::consts::PI * cutoff);
    1.0 / (1.0 + tau / dt)
}

#[derive(Debug, Clone, Copy, Default)]
struct Lowpass {
    value: Option<Vec3>,
}

impl Lowpass {
    fn apply(&mut self, x: Vec3, a: f32) -> Vec3 {
        let out = match self.value {
            Some(prev) => prev + (x - prev) * a,
            None => x,
        };
        self.value = Some(out);
        out
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct OneEuro {
    x: Lowpass,
    dx: Lowpass,
    prev: Option<Vec3>,
}

#[allow(dead_code)] // reset is for re-acquisition
impl OneEuro {
    pub fn filter(&mut self, x: Vec3, dt: f32, cfg: &OneEuroConfig) -> Vec3 {
        let dt = dt.max(1e-4);
        let velocity = match self.prev {
            Some(p) => (x - p) / dt,
            None => Vec3::ZERO,
        };
        self.prev = Some(x);

        let dx_hat = self.dx.apply(velocity, alpha(cfg.derivative_cutoff, dt));
        let cutoff = cfg.min_cutoff + cfg.beta * dx_hat.length();
        self.x.apply(x, alpha(cutoff, dt))
    }

    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

/// One filter per landmark, for one tracked hand.
#[derive(Debug, Clone)]
pub struct HandFilter {
    joints: [OneEuro; super::hand::LANDMARK_COUNT],
    cfg: OneEuroConfig,
}

impl Default for HandFilter {
    fn default() -> Self {
        Self {
            joints: [OneEuro::default(); super::hand::LANDMARK_COUNT],
            cfg: OneEuroConfig::default(),
        }
    }
}

#[allow(dead_code)]
impl HandFilter {
    pub fn apply(&mut self, landmarks: &mut [Vec3; super::hand::LANDMARK_COUNT], dt: f32) {
        for (j, p) in self.joints.iter_mut().zip(landmarks.iter_mut()) {
            *p = j.filter(*p, dt, &self.cfg);
        }
    }

    pub fn reset(&mut self) {
        for j in &mut self.joints {
            j.reset();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn converges_to_a_static_signal() {
        let mut f = OneEuro::default();
        let target = Vec3::new(0.5, 0.5, 0.0);
        let mut out = Vec3::ZERO;
        for _ in 0..120 {
            out = f.filter(target, 1.0 / 60.0, &OneEuroConfig::default());
        }
        assert!((out - target).length() < 1e-3, "{out:?}");
    }

    #[test]
    fn suppresses_jitter_around_a_fixed_point() {
        let cfg = OneEuroConfig::default();
        let mut f = OneEuro::default();
        let mut worst: f32 = 0.0;
        // Alternating +/- noise around 0.5, like a landmark sitting still.
        for i in 0..200 {
            let noise = if i % 2 == 0 { 0.01 } else { -0.01 };
            let out = f.filter(Vec3::new(0.5 + noise, 0.5, 0.0), 1.0 / 60.0, &cfg);
            if i > 50 {
                worst = worst.max((out.x - 0.5).abs());
            }
        }
        assert!(worst < 0.005, "jitter not suppressed: {worst}");
    }

    #[test]
    fn tracks_fast_motion_without_excessive_lag() {
        let cfg = OneEuroConfig::default();
        let mut f = OneEuro::default();
        let mut out = Vec3::ZERO;
        // A steady sweep; the filter should end up close to the true position.
        for i in 0..60 {
            let x = i as f32 * 0.01;
            out = f.filter(Vec3::new(x, 0.0, 0.0), 1.0 / 60.0, &cfg);
        }
        let truth = 59.0 * 0.01;
        assert!((out.x - truth).abs() < 0.05, "lag too high: {} vs {truth}", out.x);
    }
}
