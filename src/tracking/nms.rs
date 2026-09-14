//! Non-maximum suppression, shared by the palm and face detectors.

use glam::Vec2;

/// A detection with a box, so overlapping duplicates can be suppressed.
pub trait Detection: Copy {
    fn score(&self) -> f32;
    fn center(&self) -> Vec2;
    fn size(&self) -> Vec2;
}

fn iou<D: Detection>(a: &D, b: &D) -> f32 {
    let (a0, a1) = (a.center() - a.size() * 0.5, a.center() + a.size() * 0.5);
    let (b0, b1) = (b.center() - b.size() * 0.5, b.center() + b.size() * 0.5);
    let inter = (a1.min(b1) - a0.max(b0)).max(Vec2::ZERO);
    let inter_area = inter.x * inter.y;
    let union = a.size().x * a.size().y + b.size().x * b.size().y - inter_area;
    if union <= 0.0 {
        0.0
    } else {
        inter_area / union
    }
}

/// Greedy non-maximum suppression, highest score first.
pub fn nms<D: Detection>(mut dets: Vec<D>, iou_threshold: f32, max_out: usize) -> Vec<D> {
    dets.sort_by(|a, b| {
        b.score()
            .partial_cmp(&a.score())
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut kept: Vec<D> = Vec::new();
    for d in dets {
        if kept.len() >= max_out {
            break;
        }
        if kept.iter().all(|k| iou(k, &d) < iou_threshold) {
            kept.push(d);
        }
    }
    kept
}
