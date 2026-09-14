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

impl Region {
    /// Build a quad from four corner points in any order.
    pub fn quad(corners: [Vec2; 4]) -> Self {
        Region::Quad(convex_quad(corners))
    }

    pub fn is_active(&self) -> bool {
        !matches!(self, Region::None)
    }

    pub fn corners(&self) -> Option<[Vec2; 4]> {
        match self {
            Region::Quad(c) => Some(*c),
            Region::None => None,
        }
    }
}

/// Resolve which region to draw, honouring a freeze request.
///
/// Freezing latches the region as it was at the moment it was requested, so
/// the effect stays put while the hands move away or drop out of frame. A
/// freeze asked for while no region is live is held pending rather than
/// discarded — the moment a region appears, it latches.
pub fn resolve(live: Region, latched: &mut Option<Region>, freeze: bool) -> Region {
    if !freeze {
        *latched = None;
        return live;
    }
    if latched.is_none() && live.is_active() {
        *latched = Some(live);
    }
    latched.unwrap_or(live)
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tracking::hand::{lm, Hand, LANDMARK_COUNT};
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
            score: 1.0,
        }
    }

    fn square(offset: f32) -> Region {
        Region::quad([
            Vec2::new(0.1 + offset, 0.1),
            Vec2::new(0.9 + offset, 0.1),
            Vec2::new(0.9 + offset, 0.9),
            Vec2::new(0.1 + offset, 0.9),
        ])
    }

    #[test]
    fn unfrozen_follows_the_live_region() {
        let mut latched = None;
        let a = resolve(square(0.0), &mut latched, false);
        let b = resolve(square(0.5), &mut latched, false);
        assert_ne!(a, b, "should track the hands when not frozen");
        assert!(latched.is_none());
    }

    #[test]
    fn freezing_latches_the_region_in_place() {
        let mut latched = None;
        let frozen = resolve(square(0.0), &mut latched, true);
        // The hands move; the drawn region must not.
        let after = resolve(square(0.5), &mut latched, true);
        assert_eq!(frozen, after, "frozen region moved with the hands");
    }

    #[test]
    fn frozen_region_survives_losing_the_hands() {
        let mut latched = None;
        let frozen = resolve(square(0.0), &mut latched, true);
        let after = resolve(Region::None, &mut latched, true);
        assert_eq!(frozen, after, "frozen region vanished when hands were lost");
        assert!(after.is_active());
    }

    #[test]
    fn unfreezing_returns_to_the_live_region() {
        let mut latched = None;
        resolve(square(0.0), &mut latched, true);
        let live = resolve(square(0.5), &mut latched, false);
        assert_eq!(live, square(0.5));
        assert!(latched.is_none(), "latch must be released");
    }

    #[test]
    fn freezing_with_no_region_latches_once_one_appears() {
        let mut latched = None;
        // Asked for with no hands up: nothing to latch yet.
        assert_eq!(resolve(Region::None, &mut latched, true), Region::None);
        assert!(latched.is_none());
        // Hands appear — now it latches.
        let latched_region = resolve(square(0.2), &mut latched, true);
        assert_eq!(latched_region, square(0.2));
        assert_eq!(resolve(square(0.9), &mut latched, true), square(0.2));
    }

    #[test]
    fn no_region_with_fewer_than_two_hands() {
        let src = TwoHandQuad::default();
        assert_eq!(src.region(&HandFrame::default()), Region::None);
        let one = HandFrame {
            hands: vec![hand_at(0.3, 0.2, 0.6)],
        };
        assert_eq!(src.region(&one), Region::None);
    }

    #[test]
    fn quad_orders_corners_left_to_right() {
        // Deliberately pass the right-most hand first: the source must sort.
        let frame = HandFrame {
            hands: vec![hand_at(0.8, 0.2, 0.6), hand_at(0.2, 0.2, 0.6)],
        };
        let region = TwoHandQuad::default().region(&frame);
        let c = region.corners().expect("expected a quad");
        assert!(c[0].x < c[1].x, "top edge runs left to right: {c:?}");
        assert!(c[3].x < c[2].x, "bottom edge runs left to right: {c:?}");
        assert!(c[0].y < c[3].y, "index tip sits above thumb tip: {c:?}");
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
            // The hull must keep all four input points, just reordered.
            for p in input {
                assert!(
                    c.iter().any(|q| q.distance(p) < 1e-6),
                    "order {order:?} dropped corner {p:?}"
                );
            }
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
        assert!(is_convex(&normal.corners().unwrap()), "baseline should work");

        // Now flip ONLY the left hand. Its thumb swings above its index, so
        // the first and last corners swap vertically and the path crosses
        // itself: a bow-tie, not a rectangle.
        let flipped = Region::quad([
            Vec2::new(0.2, 0.7), // left index  (now BELOW)
            Vec2::new(0.8, 0.3), // right index
            Vec2::new(0.8, 0.7), // right thumb
            Vec2::new(0.2, 0.3), // left thumb  (now ABOVE)
        ]);
        // Convexity is the property the shader's SDF depends on; a bow-tie
        // fails it, which is exactly the bug this guards.
        assert!(
            is_convex(&flipped.corners().unwrap()),
            "flipping one hand produced a self-intersecting quad"
        );
    }
}
