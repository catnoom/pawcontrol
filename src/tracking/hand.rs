//! The vocabulary every other layer speaks: a tracked hand and its landmarks.

use glam::{Vec2, Vec3};

/// MediaPipe's 21-landmark topology.
///
/// ```text
///        8   12  16  20      <- tips
///        |   |   |   |
///        7   11  15  19
///        |   |   |   |
///        6   10  14  18
///        |   |   |   |
///    4   5---9---13--17      <- MCP knuckles
///     \   \  |  /  /
///      3    \ | /
///       2    \|/
///        1----0               <- wrist
/// ```
#[allow(dead_code)] // complete landmark table, kept as the reference topology
pub mod lm {
    pub const WRIST: usize = 0;
    pub const THUMB_CMC: usize = 1;
    pub const THUMB_MCP: usize = 2;
    pub const THUMB_IP: usize = 3;
    pub const THUMB_TIP: usize = 4;
    pub const INDEX_MCP: usize = 5;
    pub const INDEX_PIP: usize = 6;
    pub const INDEX_DIP: usize = 7;
    pub const INDEX_TIP: usize = 8;
    pub const MIDDLE_MCP: usize = 9;
    pub const MIDDLE_PIP: usize = 10;
    pub const MIDDLE_DIP: usize = 11;
    pub const MIDDLE_TIP: usize = 12;
    pub const RING_MCP: usize = 13;
    pub const RING_PIP: usize = 14;
    pub const RING_DIP: usize = 15;
    pub const RING_TIP: usize = 16;
    pub const PINKY_MCP: usize = 17;
    pub const PINKY_PIP: usize = 18;
    pub const PINKY_DIP: usize = 19;
    pub const PINKY_TIP: usize = 20;
}

pub const LANDMARK_COUNT: usize = 21;

/// Connected joint pairs, for drawing the skeleton overlay.
pub const BONES: [(usize, usize); 21] = [
    (0, 1), (1, 2), (2, 3), (3, 4),             // thumb
    (0, 5), (5, 6), (6, 7), (7, 8),             // index
    (9, 10), (10, 11), (11, 12),                // middle
    (13, 14), (14, 15), (15, 16),               // ring
    (0, 17), (17, 18), (18, 19), (19, 20),      // pinky
    (5, 9), (9, 13), (13, 17),                  // palm arch
];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Finger {
    Thumb,
    Index,
    Middle,
    Ring,
    Pinky,
}

#[allow(dead_code)] // mcp() is used by tests and custom gestures
impl Finger {
    pub const ALL: [Finger; 5] = [
        Finger::Thumb,
        Finger::Index,
        Finger::Middle,
        Finger::Ring,
        Finger::Pinky,
    ];

    pub fn tip(self) -> usize {
        match self {
            Finger::Thumb => lm::THUMB_TIP,
            Finger::Index => lm::INDEX_TIP,
            Finger::Middle => lm::MIDDLE_TIP,
            Finger::Ring => lm::RING_TIP,
            Finger::Pinky => lm::PINKY_TIP,
        }
    }

    /// Knuckle at the base of the finger.
    pub fn mcp(self) -> usize {
        match self {
            Finger::Thumb => lm::THUMB_CMC,
            Finger::Index => lm::INDEX_MCP,
            Finger::Middle => lm::MIDDLE_MCP,
            Finger::Ring => lm::RING_MCP,
            Finger::Pinky => lm::PINKY_MCP,
        }
    }

    /// Fingertip-to-knuckle distance, in palm-widths, for a fully extended
    /// finger.
    ///
    /// Measured from a reference hand (thumb 1.12, index 0.85, middle 0.88,
    /// ring 0.88, pinky 0.68). Individual hands vary by roughly 10%, which the
    /// clamp in `Hand::curl` absorbs.
    pub fn extended_reference(self) -> f32 {
        match self {
            Finger::Thumb => 1.12,
            Finger::Index => 0.85,
            Finger::Middle => 0.88,
            Finger::Ring => 0.88,
            Finger::Pinky => 0.68,
        }
    }

    /// Middle joint, used to decide whether the finger is curled.
    pub fn pip(self) -> usize {
        match self {
            Finger::Thumb => lm::THUMB_MCP,
            Finger::Index => lm::INDEX_PIP,
            Finger::Middle => lm::MIDDLE_PIP,
            Finger::Ring => lm::RING_PIP,
            Finger::Pinky => lm::PINKY_PIP,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Handedness {
    Left,
    Right,
}

/// One tracked hand for one frame.
///
/// Landmark x/y are normalized to the *image* (0..1, origin top-left) so the
/// whole pipeline stays resolution-independent; z is relative depth in roughly
/// the same units as x, negative meaning closer to the camera.
#[derive(Debug, Clone)]
pub struct Hand {
    pub landmarks: [Vec3; LANDMARK_COUNT],
    pub handedness: Handedness,
    /// Landmark-model presence score, 0..1.
    pub score: f32,
}

#[allow(dead_code)] // query surface for writing new gestures/regions
impl Hand {
    pub fn point(&self, idx: usize) -> Vec2 {
        self.landmarks[idx].truncate()
    }

    pub fn tip(&self, finger: Finger) -> Vec2 {
        self.point(finger.tip())
    }

    pub fn wrist(&self) -> Vec2 {
        self.point(lm::WRIST)
    }

    /// Rough palm width in normalized units.
    ///
    /// Every gesture threshold is expressed as a multiple of this so that
    /// gestures fire the same way whether the hand is near or far from the
    /// camera.
    pub fn scale(&self) -> f32 {
        let a = self.point(lm::INDEX_MCP).distance(self.point(lm::PINKY_MCP));
        let b = self.point(lm::WRIST).distance(self.point(lm::MIDDLE_MCP));
        // `b` alone collapses when the hand points at the camera, `a` alone
        // collapses when the hand turns edge-on; the max survives both.
        a.max(b).max(1e-4)
    }

    /// Distance between two fingertips, in palm-widths.
    pub fn pinch_distance(&self, a: Finger, b: Finger) -> f32 {
        self.tip(a).distance(self.tip(b)) / self.scale()
    }

    /// How curled a finger is: 0.0 fully extended, 1.0 fully curled.
    ///
    /// Measured as fingertip-to-knuckle distance in palm-widths, so it is
    /// invariant to hand rotation and to distance from the camera — unlike a
    /// raw pixel distance, which would drift as the user moves.
    pub fn curl(&self, finger: Finger) -> f32 {
        let extended = finger.extended_reference();
        // A fully curled finger brings its tip back toward its knuckle,
        // bottoming out near 40% of the extended distance.
        let curled = extended * 0.40;
        let d = self.point(finger.tip()).distance(self.point(finger.mcp())) / self.scale();
        ((extended - d) / (extended - curled)).clamp(0.0, 1.0)
    }

    /// Whether a finger is extended, judged by tip-vs-knuckle distance from
    /// the wrist. Robust to hand rotation, unlike a pure y comparison.
    pub fn is_extended(&self, finger: Finger) -> bool {
        let wrist = self.wrist();
        let tip = wrist.distance(self.point(finger.tip()));
        let pip = wrist.distance(self.point(finger.pip()));
        tip > pip * 1.15
    }

    pub fn extended_count(&self) -> usize {
        Finger::ALL.iter().filter(|f| self.is_extended(**f)).count()
    }

    /// Centroid of the palm, steadier than any single landmark.
    pub fn palm_center(&self) -> Vec2 {
        let idx = [
            lm::WRIST,
            lm::INDEX_MCP,
            lm::MIDDLE_MCP,
            lm::RING_MCP,
            lm::PINKY_MCP,
        ];
        idx.iter().map(|i| self.point(*i)).sum::<Vec2>() / idx.len() as f32
    }
}

/// All hands seen in one frame, newest first in confidence order.
#[derive(Debug, Clone, Default)]
pub struct HandFrame {
    pub hands: Vec<Hand>,
    /// Monotonic frame counter, so consumers can tell stale data from fresh.
    #[allow(dead_code)]
    pub seq: u64,
}

#[allow(dead_code)]
impl HandFrame {
    pub fn get(&self, which: Handedness) -> Option<&Hand> {
        self.hands.iter().find(|h| h.handedness == which)
    }

    /// The strongest curl of `finger` across all tracked hands.
    ///
    /// Taking the maximum means either hand can drive the control, which
    /// avoids depending on hand ordering — that is not stable across a hand
    /// being lost and reacquired.
    pub fn max_curl(&self, finger: Finger) -> Option<f32> {
        self.hands
            .iter()
            .map(|h| h.curl(finger))
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
    }

    /// The two hands ordered left-to-right *on screen*, which is what region
    /// builders actually care about. Falls back to x-position rather than
    /// trusting the model's handedness flag.
    pub fn left_right_pair(&self) -> Option<(&Hand, &Hand)> {
        if self.hands.len() < 2 {
            return None;
        }
        let mut sorted: Vec<&Hand> = self.hands.iter().collect();
        sorted.sort_by(|a, b| {
            a.palm_center()
                .x
                .partial_cmp(&b.palm_center().x)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        Some((sorted[0], sorted[1]))
    }
}
