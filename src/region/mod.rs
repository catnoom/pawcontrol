//! Regions: which part of the frame an effect applies to.
//!
//! A `RegionSource` maps the current hands to a shape. Add a feature by
//! implementing the trait — the renderer masks whatever shape comes back.

use crate::tracking::hand::{Finger, HandFrame};
use glam::Vec2;

/// A masked area, in normalized image coordinates (0..1, origin top-left).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Region {
    /// No hands, no effect.
    None,
    /// A convex quad. Corner order defines the outline path; winding may be
    /// either direction, so the inside test only requires consistency.
    Quad([Vec2; 4]),
}

#[allow(dead_code)] // contains() mirrors the shader for CPU-side hit tests
impl Region {
    pub fn is_active(&self) -> bool {
        !matches!(self, Region::None)
    }

    /// Point-in-shape test, mirroring what the shader does.
    pub fn contains(&self, p: Vec2) -> bool {
        match self {
            Region::None => false,
            Region::Quad(c) => {
                let mut positive = 0;
                let mut negative = 0;
                for i in 0..4 {
                    let a = c[i];
                    let b = c[(i + 1) % 4];
                    let cross = (b - a).perp_dot(p - a);
                    if cross > 0.0 {
                        positive += 1;
                    } else if cross < 0.0 {
                        negative += 1;
                    }
                }
                // Inside iff the point is on the same side of every edge.
                positive == 0 || negative == 0
            }
        }
    }

    pub fn corners(&self) -> Option<[Vec2; 4]> {
        match self {
            Region::Quad(c) => Some(*c),
            Region::None => None,
        }
    }
}

pub trait RegionSource: Send {
    fn name(&self) -> &str;
    fn region(&self, frame: &HandFrame) -> Region;
}

/// The reel's region: a quad spanning both hands, anchored at the index and
/// thumb tips. The left hand supplies the left edge, the right hand the right.
pub struct TwoHandQuad {
    pub top: Finger,
    pub bottom: Finger,
}

impl Default for TwoHandQuad {
    fn default() -> Self {
        Self {
            top: Finger::Index,
            bottom: Finger::Thumb,
        }
    }
}

impl RegionSource for TwoHandQuad {
    fn name(&self) -> &str {
        "two-hand-quad"
    }

    fn region(&self, frame: &HandFrame) -> Region {
        let Some((left, right)) = frame.left_right_pair() else {
            return Region::None;
        };
        // Walk the corners as a loop so the outline draws correctly:
        // left-top -> right-top -> right-bottom -> left-bottom.
        Region::Quad([
            left.tip(self.top),
            right.tip(self.top),
            right.tip(self.bottom),
            left.tip(self.bottom),
        ])
    }
}

/// A quad from a single hand's thumb and index, useful for one-handed effects.
/// Not wired up by default; swap it in via `App::region_source`.
#[allow(dead_code)]
#[derive(Default)]
pub struct SingleHandBox;

impl RegionSource for SingleHandBox {
    fn name(&self) -> &str {
        "single-hand-box"
    }

    fn region(&self, frame: &HandFrame) -> Region {
        let Some(hand) = frame.hands.first() else {
            return Region::None;
        };
        let a = hand.tip(Finger::Thumb);
        let b = hand.tip(Finger::Index);
        // Axis-aligned box spanned by the two fingertips.
        let (min, max) = (a.min(b), a.max(b));
        Region::Quad([
            Vec2::new(min.x, min.y),
            Vec2::new(max.x, min.y),
            Vec2::new(max.x, max.y),
            Vec2::new(min.x, max.y),
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tracking::hand::{lm, Hand, Handedness, LANDMARK_COUNT};
    use glam::Vec3;

    fn hand_at(x: f32, index_y: f32, thumb_y: f32) -> Hand {
        let mut landmarks = [Vec3::ZERO; LANDMARK_COUNT];
        for p in landmarks.iter_mut() {
            *p = Vec3::new(x, 0.5, 0.0);
        }
        landmarks[lm::INDEX_TIP] = Vec3::new(x, index_y, 0.0);
        landmarks[lm::THUMB_TIP] = Vec3::new(x, thumb_y, 0.0);
        // Palm landmarks decide which hand is "left" via palm_center.
        landmarks[lm::WRIST] = Vec3::new(x, 0.5, 0.0);
        Hand {
            landmarks,
            handedness: Handedness::Right,
            score: 1.0,
        }
    }

    #[test]
    fn no_region_with_fewer_than_two_hands() {
        let src = TwoHandQuad::default();
        assert_eq!(src.region(&HandFrame::default()), Region::None);
        let one = HandFrame {
            hands: vec![hand_at(0.3, 0.2, 0.6)],
            seq: 0,
        };
        assert_eq!(src.region(&one), Region::None);
    }

    #[test]
    fn quad_orders_corners_left_to_right() {
        // Deliberately pass the right-most hand first: the source must sort.
        let frame = HandFrame {
            hands: vec![hand_at(0.8, 0.2, 0.6), hand_at(0.2, 0.2, 0.6)],
            seq: 0,
        };
        let region = TwoHandQuad::default().region(&frame);
        let c = region.corners().expect("expected a quad");
        assert!(c[0].x < c[1].x, "top edge runs left to right: {c:?}");
        assert!(c[3].x < c[2].x, "bottom edge runs left to right: {c:?}");
        assert!(c[0].y < c[3].y, "index tip sits above thumb tip: {c:?}");
    }

    #[test]
    fn contains_matches_the_quad() {
        let frame = HandFrame {
            hands: vec![hand_at(0.2, 0.2, 0.6), hand_at(0.8, 0.2, 0.6)],
            seq: 0,
        };
        let region = TwoHandQuad::default().region(&frame);
        assert!(region.contains(Vec2::new(0.5, 0.4)), "center should be inside");
        assert!(!region.contains(Vec2::new(0.05, 0.4)), "left of the quad");
        assert!(!region.contains(Vec2::new(0.5, 0.9)), "below the quad");
    }

    #[test]
    fn contains_is_winding_agnostic() {
        // Same square, corners listed the other way round.
        let cw = Region::Quad([
            Vec2::new(0.0, 0.0),
            Vec2::new(1.0, 0.0),
            Vec2::new(1.0, 1.0),
            Vec2::new(0.0, 1.0),
        ]);
        let ccw = Region::Quad([
            Vec2::new(0.0, 1.0),
            Vec2::new(1.0, 1.0),
            Vec2::new(1.0, 0.0),
            Vec2::new(0.0, 0.0),
        ]);
        assert!(cw.contains(Vec2::splat(0.5)));
        assert!(ccw.contains(Vec2::splat(0.5)));
        assert!(!cw.contains(Vec2::new(1.5, 0.5)));
        assert!(!ccw.contains(Vec2::new(1.5, 0.5)));
    }
}
