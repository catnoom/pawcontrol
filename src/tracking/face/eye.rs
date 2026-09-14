//! Eye openness and blink detection.
//!
//! Openness is measured as the eye aspect ratio (EAR): the eyelid gap divided
//! by the eye's width. Dividing by the width is what makes it scale-free — it
//! reads the same whether the face is near or far, which a raw pixel gap would
//! not.
//!
//! Soukupova & Cech, "Real-Time Eye Blink Detection using Facial Landmarks".

use glam::Vec2;

/// Landmark indices into the 468-point face mesh.
///
/// Note these follow MediaPipe's convention, which names eyes from the
/// *subject's* point of view: `RIGHT_EYE` is the eye on the user's right,
/// which appears on the left of a non-mirrored camera image.
pub mod lm {
    /// Right eye: outer corner, two upper lid points, inner corner, two lower.
    pub const RIGHT_EYE: [usize; 6] = [33, 160, 158, 133, 153, 144];
    /// Left eye, same ordering.
    pub const LEFT_EYE: [usize; 6] = [362, 385, 387, 263, 373, 380];
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Eye {
    /// The user's right eye.
    Right,
    /// The user's left eye.
    Left,
}

impl Eye {
    pub fn landmarks(self) -> [usize; 6] {
        match self {
            Eye::Right => lm::RIGHT_EYE,
            Eye::Left => lm::LEFT_EYE,
        }
    }
}

/// Eye aspect ratio: roughly 0.3 wide open, under 0.15 closed.
///
/// Returns `None` if the landmark slice is too short for the mesh.
pub fn aspect_ratio(landmarks: &[Vec2], eye: Eye) -> Option<f32> {
    let idx = eye.landmarks();
    if landmarks.len() <= idx.iter().copied().max()? {
        return None;
    }
    let p = |i: usize| landmarks[idx[i]];

    // Two vertical lid measurements, averaged, over the corner-to-corner width.
    let lid = p(1).distance(p(5)) + p(2).distance(p(4));
    let width = p(0).distance(p(3));
    if width <= 1e-6 {
        return None;
    }
    Some(lid / (2.0 * width))
}

#[derive(Debug, Clone, Copy)]
pub struct BlinkConfig {
    /// EAR below this counts as closed.
    pub close_below: f32,
    /// EAR above this counts as open again. Higher than `close_below` to give
    /// hysteresis, so an eye hovering at the threshold does not chatter.
    pub open_above: f32,
    /// How long the eye must stay shut to count as a *long* blink, in seconds.
    /// Comfortably above an involuntary blink, which is ~0.1-0.4s.
    pub hold_secs: f32,
}

impl Default for BlinkConfig {
    fn default() -> Self {
        Self {
            close_below: 0.18,
            open_above: 0.25,
            hold_secs: 0.7,
        }
    }
}

/// What the detector saw this frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlinkEvent {
    /// The eye has been shut for longer than `hold_secs`. Fires once per
    /// closure, not repeatedly while it stays shut.
    LongBlink,
}

/// Tracks one eye across frames.
#[derive(Debug, Clone, Copy, Default)]
pub struct BlinkDetector {
    closed: bool,
    /// Seconds the eye has been continuously shut.
    closed_for: f32,
    /// Set once the long blink has fired, to avoid repeating while held.
    fired: bool,
}

impl BlinkDetector {
    /// Feed one frame's EAR. `None` means the eye was not measurable — the
    /// face was lost — which resets rather than counting as a closure, so
    /// losing tracking cannot be mistaken for a deliberate blink.
    pub fn update(&mut self, ear: Option<f32>, dt: f32, cfg: &BlinkConfig) -> Option<BlinkEvent> {
        let Some(ear) = ear else {
            *self = Self::default();
            return None;
        };

        if self.closed {
            if ear > cfg.open_above {
                *self = Self::default();
                return None;
            }
            self.closed_for += dt;
        } else if ear < cfg.close_below {
            self.closed = true;
            self.closed_for = 0.0;
            self.fired = false;
        }

        if self.closed && !self.fired && self.closed_for >= cfg.hold_secs {
            self.fired = true;
            return Some(BlinkEvent::LongBlink);
        }
        None
    }

    /// Seconds the eye has been shut, for a progress readout.
    pub fn closed_for(&self) -> f32 {
        if self.closed {
            self.closed_for
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A synthetic eye: `gap` is the lid separation, width fixed at 1.0.
    fn eye_landmarks(gap: f32) -> Vec<Vec2> {
        let mut pts = vec![Vec2::ZERO; 468];
        let idx = lm::RIGHT_EYE;
        pts[idx[0]] = Vec2::new(0.0, 0.0); // outer corner
        pts[idx[3]] = Vec2::new(1.0, 0.0); // inner corner
        pts[idx[1]] = Vec2::new(0.33, -gap / 2.0);
        pts[idx[2]] = Vec2::new(0.66, -gap / 2.0);
        pts[idx[5]] = Vec2::new(0.33, gap / 2.0);
        pts[idx[4]] = Vec2::new(0.66, gap / 2.0);
        pts
    }

    #[test]
    fn aspect_ratio_matches_the_lid_gap() {
        // Width 1.0, both lid gaps equal `gap`, so EAR == gap.
        let ear = aspect_ratio(&eye_landmarks(0.3), Eye::Right).unwrap();
        assert!((ear - 0.3).abs() < 1e-5, "{ear}");
    }

    #[test]
    fn aspect_ratio_is_scale_free() {
        // Doubling the whole eye must not change the ratio.
        let small = aspect_ratio(&eye_landmarks(0.3), Eye::Right).unwrap();
        let mut big = eye_landmarks(0.3);
        for p in big.iter_mut() {
            *p *= 3.0;
        }
        let large = aspect_ratio(&big, Eye::Right).unwrap();
        assert!((small - large).abs() < 1e-5, "{small} vs {large}");
    }

    #[test]
    fn short_blink_does_not_fire() {
        let cfg = BlinkConfig::default();
        let mut d = BlinkDetector::default();
        // Shut for 0.3s — an ordinary involuntary blink.
        for _ in 0..18 {
            assert!(d.update(Some(0.1), 1.0 / 60.0, &cfg).is_none());
        }
        // Then open again.
        assert!(d.update(Some(0.3), 1.0 / 60.0, &cfg).is_none());
    }

    #[test]
    fn long_blink_fires_once() {
        let cfg = BlinkConfig::default();
        let mut d = BlinkDetector::default();
        let mut fired = 0;
        // Held shut for a full second, well past hold_secs.
        for _ in 0..60 {
            if d.update(Some(0.1), 1.0 / 60.0, &cfg).is_some() {
                fired += 1;
            }
        }
        assert_eq!(fired, 1, "long blink should fire exactly once while held");
    }

    #[test]
    fn reopening_allows_another_blink() {
        let cfg = BlinkConfig::default();
        let mut d = BlinkDetector::default();
        for _ in 0..60 {
            d.update(Some(0.1), 1.0 / 60.0, &cfg);
        }
        // Open well past the hysteresis threshold, then blink again.
        d.update(Some(0.35), 1.0 / 60.0, &cfg);
        let mut fired = 0;
        for _ in 0..60 {
            if d.update(Some(0.1), 1.0 / 60.0, &cfg).is_some() {
                fired += 1;
            }
        }
        assert_eq!(fired, 1);
    }

    #[test]
    fn hysteresis_keeps_a_marginal_eye_shut() {
        let cfg = BlinkConfig::default();
        let mut d = BlinkDetector::default();
        d.update(Some(0.1), 1.0 / 60.0, &cfg); // closed
        // Between close_below and open_above: must still count as closed.
        for _ in 0..60 {
            d.update(Some(0.21), 1.0 / 60.0, &cfg);
        }
        assert!(d.closed_for() > 0.5, "reopened too eagerly");
    }

    #[test]
    fn losing_the_face_resets_rather_than_counting_as_closed() {
        let cfg = BlinkConfig::default();
        let mut d = BlinkDetector::default();
        for _ in 0..20 {
            d.update(Some(0.1), 1.0 / 60.0, &cfg);
        }
        // Face lost part-way through a closure.
        assert!(d.update(None, 1.0 / 60.0, &cfg).is_none());
        assert_eq!(d.closed_for(), 0.0, "a lost face must not accumulate");
        // And it must not fire on the next frame either.
        assert!(d.update(Some(0.1), 1.0 / 60.0, &cfg).is_none());
    }
}
