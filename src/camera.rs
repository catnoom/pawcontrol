//! Camera capture on its own thread.
//!
//! Capture, inference and rendering all run at different rates, so they are
//! decoupled through `FrameBus`: the camera publishes the newest frame and
//! consumers take whatever is current. Nobody blocks anybody.

use anyhow::{anyhow, Context, Result};
use std::sync::mpsc::{Receiver, Sender};
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

/// A request to the capture thread. The camera device is not `Send`, so it
/// can only be reconfigured from inside its own thread.
pub enum CameraCommand {
    SetResolution(u32, u32),
}

/// What the control panel needs to know about the camera.
#[derive(Clone, Debug, Default)]
pub struct CameraState {
    /// Resolutions the device reports it supports, de-duplicated and sorted.
    pub available: Vec<(u32, u32)>,
    /// The resolution actually in use — the driver may not honour a request.
    pub current: (u32, u32),
    /// Set when the last resolution change was rejected.
    pub last_error: Option<String>,
}

/// Handle to a running capture thread.
pub struct CameraHandle {
    commands: Sender<CameraCommand>,
    pub state: Arc<Mutex<CameraState>>,
}

impl CameraHandle {
    pub fn set_resolution(&self, width: u32, height: u32) {
        // A dead capture thread is not fatal; the app keeps rendering the
        // last frame, and the panel shows the error.
        let _ = self.commands.send(CameraCommand::SetResolution(width, height));
    }

    pub fn state(&self) -> CameraState {
        self.state.lock().unwrap().clone()
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
/// The capture device is created *inside* the worker thread: nokhwa's `Camera`
/// owns a platform backend that is not `Send`, so it cannot be built here and
/// moved across. Reconfiguration therefore goes through a command channel
/// rather than a direct call.
pub fn spawn(cfg: CameraConfig, bus: Arc<FrameBus>) -> Result<CameraHandle> {
    let (tx, rx) = std::sync::mpsc::channel::<CameraCommand>();
    let (ready_tx, ready_rx) = std::sync::mpsc::channel::<Result<CameraState>>();
    let state = Arc::new(Mutex::new(CameraState::default()));
    let thread_state = state.clone();

    std::thread::Builder::new()
        .name("camera".into())
        .spawn(move || {
            let mut camera = match open(&cfg) {
                Ok(c) => c,
                Err(e) => {
                    let _ = ready_tx.send(Err(e));
                    return;
                }
            };

            let resolution = camera.resolution();
            let (width, height) = (resolution.width_x, resolution.height_y);
            if width == 0 || height == 0 {
                let _ = ready_tx.send(Err(anyhow!("camera reported a zero-sized resolution")));
                return;
            }

            let initial = CameraState {
                available: supported_resolutions(&mut camera),
                current: (width, height),
                last_error: None,
            };
            log::info!("camera {} streaming at {width}x{height}", cfg.index);
            *thread_state.lock().unwrap() = initial.clone();
            let _ = ready_tx.send(Ok(initial));

            capture_loop(&mut camera, &bus, &rx, &thread_state);
        })
        .context("spawning camera thread")?;

    let initial = ready_rx
        .recv()
        .context("camera thread exited before reporting readiness")??;
    *state.lock().unwrap() = initial;

    Ok(CameraHandle {
        commands: tx,
        state,
    })
}

/// Distinct resolutions the device advertises, sorted by pixel count.
///
/// Returns an empty list if the query fails — the panel then falls back to
/// showing only the current resolution rather than failing to open.
fn supported_resolutions(camera: &mut Camera) -> Vec<(u32, u32)> {
    let Ok(formats) = camera.compatible_camera_formats() else {
        log::warn!("could not enumerate camera formats");
        return Vec::new();
    };
    let mut seen: Vec<(u32, u32)> = formats
        .iter()
        .map(|f| (f.resolution().width_x, f.resolution().height_y))
        .collect();
    seen.sort_unstable_by_key(|(w, h)| (*w as u64) * (*h as u64));
    seen.dedup();
    seen
}

fn capture_loop(
    camera: &mut Camera,
    bus: &Arc<FrameBus>,
    commands: &Receiver<CameraCommand>,
    state: &Arc<Mutex<CameraState>>,
) {
    let mut seq = 0u64;
    let mut failures = 0u32;
    loop {
        // Apply any pending reconfiguration before grabbing the next frame.
        while let Ok(command) = commands.try_recv() {
            match command {
                CameraCommand::SetResolution(w, h) => {
                    apply_resolution(camera, w, h, state);
                    // The stream restarts underneath us; treat the next few
                    // reads as fresh rather than counting earlier failures.
                    failures = 0;
                }
            }
        }

        match camera.frame() {
            Ok(buffer) => {
                failures = 0;
                // Size the frame from the buffer itself, so a resolution
                // change cannot desync the texture from the pixels.
                let res = buffer.resolution();
                let mut frame = Frame::new(res.width_x, res.height_y);
                if let Err(e) = buffer.decode_image_to_buffer::<RgbAFormat>(&mut frame.rgba) {
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
                // A few dropped frames are normal; a sustained failure means
                // the device is gone.
                if failures > 60 {
                    log::error!("camera stopped responding, capture thread exiting");
                    return;
                }
                std::thread::sleep(std::time::Duration::from_millis(10));
            }
        }
    }
}

fn apply_resolution(camera: &mut Camera, w: u32, h: u32, state: &Arc<Mutex<CameraState>>) {
    let result = camera.set_resolution(Resolution::new(w, h));
    let mut state = state.lock().unwrap();
    match result {
        Ok(()) => {
            // Trust the device over the request: it may have snapped to the
            // nearest mode it actually supports.
            let actual = camera.resolution();
            state.current = (actual.width_x, actual.height_y);
            state.last_error = None;
            log::info!("camera resolution now {}x{}", actual.width_x, actual.height_y);
        }
        Err(e) => {
            log::warn!("camera rejected {w}x{h}: {e}");
            state.last_error = Some(format!("{w}x{h} rejected: {e}"));
        }
    }
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
