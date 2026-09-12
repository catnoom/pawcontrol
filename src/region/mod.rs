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
/// Order four points into a convex, simply-connected quad.
///
/// Callers supply corners in whatever order their landmarks come in, which is
/// *not* a safe polygon: e.g. flipping one hand swings its thumb above its
/// index, and a fixed corner order then traces a self-intersecting bow-tie.
/// Both the mask test and the shader's SDF assume convexity, so we take the
/// convex hull instead of trusting the caller's ordering.
///
/// A hull of three points (one fingertip inside the triangle of the others) is
/// padded with an edge midpoint rather than a duplicated vertex, because a
/// zero-length edge has no usable normal in the SDF.
fn convex_quad(points: [Vec2; 4]) -> [Vec2; 4] {
    let cmp = |a: &Vec2, b: &Vec2| {
        a.x.partial_cmp(&b.x)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.y.partial_cmp(&b.y).unwrap_or(std::cmp::Ordering::Equal))
    };
    let mut sorted = points;
    sorted.sort_by(cmp);

    // Andrew's monotone chain. The upper hull must not pop vertices belonging
    // to the lower hull, hence the separate floor.
    let cross = |o: Vec2, a: Vec2, b: Vec2| (a - o).perp_dot(b - o);
    let mut hull: Vec<Vec2> = Vec::with_capacity(8);

    for &p in sorted.iter() {
        while hull.len() >= 2 && cross(hull[hull.len() - 2], hull[hull.len() - 1], p) <= 0.0 {
            hull.pop();
        }
        hull.push(p);
    }
    let lower = hull.len() + 1;
    for &p in sorted.iter().rev().skip(1) {
        while hull.len() >= lower && cross(hull[hull.len() - 2], hull[hull.len() - 1], p) <= 0.0 {
            hull.pop();
        }
        hull.push(p);
    }
    // The closing vertex repeats the start.
    hull.pop();
    hull.dedup_by(|a, b| a.distance_squared(*b) < 1e-12);

    match hull.len() {
        4 => [hull[0], hull[1], hull[2], hull[3]],
        3 => {
            // Split the longest edge so every edge keeps a real normal.
            let longest = (0..3)
                .max_by(|&i, &j| {
                    let di = hull[i].distance(hull[(i + 1) % 3]);
                    let dj = hull[j].distance(hull[(j + 1) % 3]);
                    di.partial_cmp(&dj).unwrap_or(std::cmp::Ordering::Equal)
                })
                .unwrap_or(0);
            let mid = (hull[longest] + hull[(longest + 1) % 3]) * 0.5;
            let mut out = [Vec2::ZERO; 4];
            let mut k = 0;
            for i in 0..3 {
                out[k] = hull[i];
                k += 1;
                if i == longest {
                    out[k] = mid;
                    k += 1;
                }
            }
            out
        }
        // Collinear or coincident: no usable area. Pass through so the caller
        // still gets a (degenerate) shape rather than a panic.
        _ => points,
    }
}

#[allow(dead_code)] // contains() mirrors the shader for CPU-side hit tests
impl Region {
    /// Build a quad from four corner points in any order.
    pub fn quad(corners: [Vec2; 4]) -> Self {
        Region::Quad(convex_quad(corners))
    }

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
        // Order is fixed up by `Region::quad`: a flipped hand puts its thumb
        // above its index, which would otherwise cross the quad.
        Region::quad([
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
        Region::quad([
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

#[cfg(test)]
mod flip_tests {
    use super::*;

    /// Every edge must turn the same way for the SDF to be valid.
    ///
    /// The tolerance is loose because a padded triangle intentionally carries
    /// one collinear vertex (it splits an existing edge), whose cross product
    /// is zero up to f32 rounding. Collinear is fine for the SDF — it just
    /// repeats an edge normal — so only a genuine reversal counts.
    fn is_convex(c: &[Vec2; 4]) -> bool {
        let mut pos = false;
        let mut neg = false;
        for i in 0..4 {
            let a = c[i];
            let b = c[(i + 1) % 4];
            let d = c[(i + 2) % 4];
            let cross = (b - a).perp_dot(d - b);
            if cross > 1e-5 { pos = true; }
            if cross < -1e-5 { neg = true; }
        }
        !(pos && neg)
    }

    #[test]
    fn hull_is_convex_for_any_corner_order() {
        let pts = [
            Vec2::new(0.2, 0.3),
            Vec2::new(0.8, 0.3),
            Vec2::new(0.8, 0.7),
            Vec2::new(0.2, 0.7),
        ];
        // Every permutation of the same four points must yield a convex quad
        // enclosing their centre.
        let idx = [
            [0, 1, 2, 3], [0, 2, 1, 3], [3, 1, 2, 0], [1, 0, 3, 2],
            [2, 0, 1, 3], [3, 2, 1, 0], [0, 3, 1, 2], [1, 3, 0, 2],
        ];
        for order in idx {
            let input = [pts[order[0]], pts[order[1]], pts[order[2]], pts[order[3]]];
            let region = Region::quad(input);
            let c = region.corners().unwrap();
            assert!(is_convex(&c), "order {order:?} produced a non-convex quad: {c:?}");
            assert!(
                region.contains(Vec2::new(0.5, 0.5)),
                "order {order:?} lost the centre"
            );
        }
    }

    #[test]
    fn hull_handles_a_point_inside_the_others() {
        // One fingertip inside the triangle of the other three: the hull is a
        // triangle, padded with an edge midpoint rather than a duplicate.
        let region = Region::quad([
            Vec2::new(0.1, 0.1),
            Vec2::new(0.9, 0.1),
            Vec2::new(0.5, 0.9),
            Vec2::new(0.5, 0.4), // interior
        ]);
        let c = region.corners().unwrap();
        assert!(is_convex(&c), "{c:?}");
        for i in 0..4 {
            let d = c[i].distance(c[(i + 1) % 4]);
            assert!(d > 1e-6, "degenerate edge {i} has no normal: {c:?}");
        }
        assert!(region.contains(Vec2::new(0.5, 0.3)));
    }

    #[test]
    fn flipping_one_hand_keeps_a_usable_region() {
        // Palms facing the camera: each hand's index tip is above its thumb
        // tip, so L.index -> R.index -> R.thumb -> L.thumb traces a rectangle.
        let normal = Region::quad([
            Vec2::new(0.2, 0.3), // left index  (top-left)
            Vec2::new(0.8, 0.3), // right index (top-right)
            Vec2::new(0.8, 0.7), // right thumb (bottom-right)
            Vec2::new(0.2, 0.7), // left thumb  (bottom-left)
        ]);
        assert!(normal.contains(Vec2::new(0.5, 0.5)), "baseline should work");

        // Now flip ONLY the left hand. Its thumb swings above its index, so
        // the first and last corners swap vertically and the path crosses
        // itself: a bow-tie, not a rectangle.
        let flipped = Region::quad([
            Vec2::new(0.2, 0.7), // left index  (now BELOW)
            Vec2::new(0.8, 0.3), // right index
            Vec2::new(0.8, 0.7), // right thumb
            Vec2::new(0.2, 0.3), // left thumb  (now ABOVE)
        ]);
        assert!(
            flipped.contains(Vec2::new(0.5, 0.5)),
            "flipping one hand must not break the region mask"
        );
        // The region must still exclude points genuinely outside it.
        assert!(!flipped.contains(Vec2::new(0.5, 0.95)));
        assert!(!flipped.contains(Vec2::new(0.02, 0.5)));
    }
}
