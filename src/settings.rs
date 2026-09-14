//! Runtime-tunable settings, shared between the control panel and the pipeline.

use crate::tracking::face::FaceSettings;
use crate::tracking::filter::OneEuroConfig;
use std::sync::{Arc, Mutex};

/// A single float knob a subsystem exposes to the control panel.
///
/// Borrowing the value directly means a component declares its tunables once
/// and the panel edits them in place — no copying settings back and forth, and
/// no central registry to keep in sync.
pub struct Tunable<'a> {
    pub label: &'static str,
    pub value: &'a mut f32,
    pub min: f32,
    pub max: f32,
}

impl<'a> Tunable<'a> {
    pub fn new(label: &'static str, value: &'a mut f32, min: f32, max: f32) -> Self {
        Self {
            label,
            value,
            min,
            max,
        }
    }
}

/// Settings the tracking thread reads. Shared because tracking runs on its own
/// thread; it re-reads these once per frame, so edits apply live.
#[derive(Clone, Copy, Debug)]
pub struct TrackingSettings {
    pub max_hands: usize,
    /// Landmark confidence below which a hand is considered lost.
    pub presence_threshold: f32,
    /// Palm-detector confidence needed to seed a new track.
    pub palm_score_threshold: f32,
    /// Frames between palm-detection attempts while short of hands.
    pub redetect_interval: u64,
    pub filter: OneEuroConfig,
    pub face: FaceSettings,
}

impl Default for TrackingSettings {
    fn default() -> Self {
        Self {
            max_hands: 2,
            presence_threshold: 0.6,
            palm_score_threshold: 0.5,
            redetect_interval: 4,
            filter: OneEuroConfig::default(),
            face: FaceSettings::default(),
        }
    }
}

pub type Shared<T> = Arc<Mutex<T>>;

pub fn shared<T>(value: T) -> Shared<T> {
    Arc::new(Mutex::new(value))
}
