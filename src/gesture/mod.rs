//! Gesture recognition: turns landmark geometry into discrete events.
//!
//! Detectors are independent and stateful. Add a feature by implementing
//! `GestureDetector` and pushing it into the `GestureEngine`.

use crate::tracking::hand::{Finger, HandFrame};

/// Something a hand did. Consumers match on this to drive behaviour.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GestureEvent {
    /// Two fingertips on the same hand came together.
    TouchStart { hand: usize, a: Finger, b: Finger },
    /// ...and separated again.
    TouchEnd { hand: usize, a: Finger, b: Finger },
    /// The number of tracked hands changed.
    HandCountChanged { count: usize },
}

pub trait GestureDetector: Send {
    fn name(&self) -> &str;
    fn update(&mut self, frame: &HandFrame, dt: f32, out: &mut Vec<GestureEvent>);
}

/// Runs every detector over each frame and collects their events.
pub struct GestureEngine {
    detectors: Vec<Box<dyn GestureDetector>>,
    events: Vec<GestureEvent>,
}

impl GestureEngine {
    pub fn new() -> Self {
        Self {
            detectors: Vec::new(),
            events: Vec::new(),
        }
    }

    pub fn with(mut self, d: impl GestureDetector + 'static) -> Self {
        log::debug!("gesture detector registered: {}", d.name());
        self.detectors.push(Box::new(d));
        self
    }

    pub fn update(&mut self, frame: &HandFrame, dt: f32) -> &[GestureEvent] {
        self.events.clear();
        for d in &mut self.detectors {
            d.update(frame, dt, &mut self.events);
        }
        &self.events
    }
}

impl Default for GestureEngine {
    fn default() -> Self {
        Self::new()
    }
}

/// Fires when two fingertips touch.
///
/// Thresholds are in palm-widths, so the gesture behaves identically whether
/// the hand is close to the camera or far away. Separate enter/exit distances
/// give hysteresis, which stops the event flickering when the fingers hover
/// right at the boundary.
pub struct FingerTouch {
    a: Finger,
    b: Finger,
    enter: f32,
    exit: f32,
    /// Per-hand contact state, indexed like `HandFrame::hands`.
    contact: Vec<bool>,
    /// Refuses a re-trigger until this reaches zero, so one physical tap
    /// cannot emit a burst of events.
    cooldown: Vec<f32>,
    cooldown_secs: f32,
}

impl FingerTouch {
    pub fn new(a: Finger, b: Finger) -> Self {
        Self {
            a,
            b,
            enter: 0.32,
            exit: 0.45,
            contact: Vec::new(),
            cooldown: Vec::new(),
            cooldown_secs: 0.35,
        }
    }

}

impl GestureDetector for FingerTouch {
    fn name(&self) -> &str {
        "finger-touch"
    }

    fn update(&mut self, frame: &HandFrame, dt: f32, out: &mut Vec<GestureEvent>) {
        self.contact.resize(frame.hands.len(), false);
        self.cooldown.resize(frame.hands.len(), 0.0);

        for (i, hand) in frame.hands.iter().enumerate() {
            self.cooldown[i] = (self.cooldown[i] - dt).max(0.0);
            let d = hand.pinch_distance(self.a, self.b);

            if !self.contact[i] && d < self.enter {
                self.contact[i] = true;
                if self.cooldown[i] <= 0.0 {
                    self.cooldown[i] = self.cooldown_secs;
                    out.push(GestureEvent::TouchStart {
                        hand: i,
                        a: self.a,
                        b: self.b,
                    });
                }
            } else if self.contact[i] && d > self.exit {
                self.contact[i] = false;
                out.push(GestureEvent::TouchEnd {
                    hand: i,
                    a: self.a,
                    b: self.b,
                });
            }
        }
    }
}

/// Reports when hands enter or leave the frame.
#[derive(Default)]
pub struct HandCount {
    last: Option<usize>,
}

impl GestureDetector for HandCount {
    fn name(&self) -> &str {
        "hand-count"
    }

    fn update(&mut self, frame: &HandFrame, _dt: f32, out: &mut Vec<GestureEvent>) {
        let count = frame.hands.len();
        if self.last != Some(count) {
            self.last = Some(count);
            out.push(GestureEvent::HandCountChanged { count });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tracking::hand::{lm, Hand, LANDMARK_COUNT};
    use glam::Vec3;

    /// A synthetic hand with a controllable thumb/pinky gap.
    fn hand_with_gap(gap: f32) -> Hand {
        let mut landmarks = [Vec3::ZERO; LANDMARK_COUNT];
        // Give the hand a palm width of 0.1 so gaps are easy to reason about.
        landmarks[lm::INDEX_MCP] = Vec3::new(0.5, 0.5, 0.0);
        landmarks[lm::PINKY_MCP] = Vec3::new(0.6, 0.5, 0.0);
        landmarks[lm::WRIST] = Vec3::new(0.5, 0.5, 0.0);
        landmarks[lm::MIDDLE_MCP] = Vec3::new(0.5, 0.5, 0.0);
        landmarks[lm::THUMB_TIP] = Vec3::new(0.5, 0.5, 0.0);
        landmarks[lm::PINKY_TIP] = Vec3::new(0.5 + gap, 0.5, 0.0);
        Hand {
            landmarks,
            score: 1.0,
        }
    }

    fn frame(gap: f32) -> HandFrame {
        HandFrame {
            hands: vec![hand_with_gap(gap)],
        }
    }

    #[test]
    fn touch_fires_once_on_contact() {
        let mut d = FingerTouch::new(Finger::Thumb, Finger::Pinky);
        let mut out = Vec::new();

        // Far apart: silence.
        d.update(&frame(0.08), 0.016, &mut out);
        assert!(out.is_empty());

        // Closed: one event.
        out.clear();
        d.update(&frame(0.01), 0.016, &mut out);
        assert_eq!(out.len(), 1, "{out:?}");

        // Still closed: must not repeat.
        out.clear();
        d.update(&frame(0.01), 0.016, &mut out);
        assert!(out.is_empty(), "re-fired while held: {out:?}");
    }

    #[test]
    fn hysteresis_prevents_flicker_at_the_boundary() {
        let mut d = FingerTouch::new(Finger::Thumb, Finger::Pinky);
        let mut out = Vec::new();
        d.update(&frame(0.01), 0.016, &mut out); // contact
        out.clear();

        // Nudge just past the enter threshold but inside the exit threshold:
        // this must NOT count as a release.
        d.update(&frame(0.035), 0.016, &mut out);
        assert!(out.is_empty(), "released too eagerly: {out:?}");

        // Well past the exit threshold: now it releases.
        d.update(&frame(0.06), 0.016, &mut out);
        assert!(matches!(out[0], GestureEvent::TouchEnd { .. }), "{out:?}");
    }

    #[test]
    fn cooldown_blocks_rapid_retrigger() {
        let mut d = FingerTouch::new(Finger::Thumb, Finger::Pinky);
        let mut out = Vec::new();
        d.update(&frame(0.01), 0.016, &mut out);
        d.update(&frame(0.06), 0.016, &mut out); // release
        out.clear();
        // Immediate re-touch inside the cooldown window is swallowed.
        d.update(&frame(0.01), 0.016, &mut out);
        assert!(out.is_empty(), "{out:?}");
    }

    #[test]
    fn hand_count_reports_changes_only() {
        let mut d = HandCount::default();
        let mut out = Vec::new();
        d.update(&frame(0.05), 0.016, &mut out);
        assert_eq!(out.len(), 1);
        out.clear();
        d.update(&frame(0.05), 0.016, &mut out);
        assert!(out.is_empty());
    }
}
