//! Camera capture on its own thread.
//!
//! Capture, inference and rendering all run at different rates, so they are
//! decoupled through `FrameBus`: the camera publishes the newest frame and
//! consumers take whatever is current. Nobody blocks anybody.

use anyhow::{anyhow, Context, Result};
use std::sync::{Arc, Mutex};

use nokhwa::pixel_format::RgbAFormat;
use nokhwa::utils::{
    ApiBackend, CameraFormat, CameraIndex, FrameFormat, RequestedFormat, RequestedFormatType,
    Resolution,
};
use nokhwa::Camera;

use crate::frame::Frame;

/// A single-slot mailbox holding the most recent frame.
///
/// Deliberately lossy: if a consumer is slow it skips frames rather than
/// building a backlog, which is what you want for a live effect.
#[derive(Default)]
pub struct FrameBus {
    slot: Mutex<Option<Arc<Frame>>>,
}

impl FrameBus {
    pub fn publish(&self, frame: Arc<Frame>) {
        *self.slot.lock().unwrap() = Some(frame);
    }

    pub fn latest(&self) -> Option<Arc<Frame>> {
        self.slot.lock().unwrap().clone()
    }
}

pub struct CameraConfig {
    pub index: u32,
    /// Requested capture size. The driver may pick something close instead.
    pub width: u32,
    pub height: u32,
}

impl Default for CameraConfig {
    fn default() -> Self {
        // 640x480 keeps the CPU-side ROI sampling cheap while giving the
        // landmark model plenty of detail.
        Self {
            index: 0,
            width: 640,
            height: 480,
        }
    }
}

/// Device names that are virtual capture sources rather than real cameras.
/// Machines with OBS, NDI or similar tools installed often expose several of
/// these ahead of the physical webcam, so index 0 is a poor default.
const VIRTUAL_HINTS: [&str; 6] = ["ndi", "virtual", "obs", "dummy", "screen", "capture card"];

fn looks_virtual(name: &str) -> bool {
    let lower = name.to_ascii_lowercase();
    VIRTUAL_HINTS.iter().any(|h| lower.contains(h))
}

/// Pick a sensible default camera: the first device that does not look like a
/// virtual source, falling back to index 0.
pub fn auto_index() -> u32 {
    let Ok(devices) = nokhwa::query(ApiBackend::Auto) else {
        return 0;
    };
    for device in &devices {
        if !looks_virtual(&device.human_name()) {
            if let CameraIndex::Index(i) = device.index() {
                log::info!("auto-selected camera {i}: {}", device.human_name());
                return *i;
            }
        }
    }
    log::warn!("no physical camera identified; falling back to camera 0");
    0
}

/// List attached cameras, for diagnostics when the requested one fails.
pub fn list() -> Vec<String> {
    nokhwa::query(ApiBackend::Auto)
        .map(|devices| {
            devices
                .iter()
                .map(|d| format!("{}: {}", d.index(), d.human_name()))
                .collect()
        })
        .unwrap_or_default()
}

/// Open the camera and stream frames into `bus` until the process exits.
///
/// Returns the negotiated resolution, which may differ from the request.
///
/// The capture device is created *inside* the worker thread: nokhwa's `Camera`
/// owns a platform backend that is not `Send`, so it cannot be built here and
/// moved across. The thread reports success or failure back over a channel.
pub fn spawn(cfg: CameraConfig, bus: Arc<FrameBus>) -> Result<(u32, u32)> {
    let (tx, rx) = std::sync::mpsc::channel::<Result<(u32, u32)>>();

    std::thread::Builder::new()
        .name("camera".into())
        .spawn(move || {
            let mut camera = match open(&cfg) {
                Ok(c) => c,
                Err(e) => {
                    let _ = tx.send(Err(e));
                    return;
                }
            };

            let resolution = camera.resolution();
            let (width, height) = (resolution.width_x, resolution.height_y);
            if width == 0 || height == 0 {
                let _ = tx.send(Err(anyhow!("camera reported a zero-sized resolution")));
                return;
            }
            log::info!("camera {} streaming at {width}x{height}", cfg.index);
            let _ = tx.send(Ok((width, height)));

            let mut seq = 0u64;
            let mut failures = 0u32;
            loop {
                match camera.frame() {
                    Ok(buffer) => {
                        failures = 0;
                        let mut frame = Frame::new(width, height);
                        if let Err(e) = buffer.decode_image_to_buffer::<RgbAFormat>(&mut frame.rgba)
                        {
                            log::warn!("frame decode failed: {e}");
                            continue;
                        }
                        seq += 1;
                        frame.seq = seq;
                        bus.publish(Arc::new(frame));
                    }
                    Err(e) => {
                        failures += 1;
                        log::warn!("camera read failed ({failures}): {e}");
                        // A few dropped frames are normal; a sustained failure
                        // means the device is gone.
                        if failures > 60 {
                            log::error!("camera stopped responding, capture thread exiting");
                            return;
                        }
                        std::thread::sleep(std::time::Duration::from_millis(10));
                    }
                }
            }
        })
        .context("spawning camera thread")?;

    rx.recv()
        .context("camera thread exited before reporting readiness")?
}

fn open(cfg: &CameraConfig) -> Result<Camera> {
    let requested = RequestedFormat::new::<RgbAFormat>(RequestedFormatType::Closest(
        CameraFormat::new(
            Resolution::new(cfg.width, cfg.height),
            FrameFormat::MJPEG,
            30,
        ),
    ));

    let mut camera = Camera::new(CameraIndex::Index(cfg.index), requested).with_context(|| {
        let available = list();
        if available.is_empty() {
            "no cameras found".to_string()
        } else {
            format!("could not open camera {}; available: {available:?}", cfg.index)
        }
    })?;

    camera.open_stream().context("opening camera stream")?;
    Ok(camera)
}
