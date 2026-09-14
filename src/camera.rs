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
    /// Carries the whole format triple, not just a size: a webcam mode is
    /// (resolution, pixel format, frame rate) and only complete triples it
    /// advertises are accepted.
    SetFormat(CameraFormat),
}

/// What the control panel needs to know about the camera.
#[derive(Clone, Debug, Default)]
pub struct CameraState {
    /// Every mode the device advertises, sorted by pixel count then frame rate.
    pub available: Vec<CameraFormat>,
    /// The resolution actually in use — the driver may not honour a request.
    pub current: (u32, u32),
    /// The full mode in use, for display.
    pub current_format: Option<CameraFormat>,
    /// Set when the last change was rejected.
    pub last_error: Option<String>,
}

impl CameraState {
    /// Distinct resolutions, for the panel's dropdown.
    pub fn resolutions(&self) -> Vec<(u32, u32)> {
        let mut out: Vec<(u32, u32)> = self
            .available
            .iter()
            .map(|f| (f.resolution().width_x, f.resolution().height_y))
            .collect();
        out.sort_unstable_by_key(|(w, h)| (*w as u64) * (*h as u64));
        out.dedup();
        out
    }
}

/// Pick the best advertised mode at a given resolution.
///
/// Preferring MJPEG matters: uncompressed modes (YUYV/NV12) are limited by USB
/// bandwidth, so a webcam typically offers 1080p only as MJPEG. Asking for
/// 1080p while holding the 640x480 pixel format is what produced
/// "Failed to fulfill requested format".
pub fn best_mode(available: &[CameraFormat], width: u32, height: u32) -> Option<CameraFormat> {
    available
        .iter()
        .filter(|f| f.resolution().width_x == width && f.resolution().height_y == height)
        .copied()
        .max_by_key(|f| {
            let mjpeg = f.format() == FrameFormat::MJPEG;
            // Prefer the fastest mode that is still a sane capture rate;
            // anything above 60 is ranked below it rather than chased.
            let fps = f.frame_rate();
            let rank = if fps <= 60 { fps } else { 0 };
            (mjpeg, rank)
        })
}

/// Handle to a running capture thread.
pub struct CameraHandle {
    commands: Sender<CameraCommand>,
    pub state: Arc<Mutex<CameraState>>,
}

impl CameraHandle {
    /// Request a resolution, resolved against the device's advertised modes.
    pub fn set_resolution(&self, width: u32, height: u32) {
        let mode = {
            let state = self.state.lock().unwrap();
            best_mode(&state.available, width, height)
        };
        match mode {
            Some(mode) => {
                // A dead capture thread is not fatal; the app keeps rendering
                // the last frame, and the panel shows the error.
                let _ = self.commands.send(CameraCommand::SetFormat(mode));
            }
            None => {
                self.state.lock().unwrap().last_error =
                    Some(format!("{width}x{height} is not an advertised mode"));
            }
        }
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
                available: supported_modes(&mut camera),
                current: (width, height),
                current_format: Some(camera.camera_format()),
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

/// Every mode the device advertises, sorted by pixel count then frame rate.
///
/// Returns an empty list if the query fails — the panel then falls back to
/// showing only the current mode rather than failing to open.
pub fn supported_modes(camera: &mut Camera) -> Vec<CameraFormat> {
    let Ok(mut formats) = camera.compatible_camera_formats() else {
        log::warn!("could not enumerate camera formats");
        return Vec::new();
    };
    formats.sort_unstable_by_key(|f| {
        let r = f.resolution();
        ((r.width_x as u64) * (r.height_y as u64), f.frame_rate())
    });
    formats
}

/// Open the camera briefly and report every mode it advertises.
pub fn list_modes(cfg: &CameraConfig) -> Result<Vec<CameraFormat>> {
    let mut camera = open(cfg)?;
    Ok(supported_modes(&mut camera))
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
                CameraCommand::SetFormat(mode) => {
                    apply_format(camera, mode, state);
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

fn apply_format(camera: &mut Camera, mode: CameraFormat, state: &Arc<Mutex<CameraState>>) {
    // `set_camera_requset` resolves the request against the device's own list
    // and reports back what it settled on. Ask for the exact advertised mode
    // first, then fall back to the nearest match rather than giving up: a
    // device can list a mode it will not actually grant.
    let exact = RequestedFormat::new::<RgbAFormat>(RequestedFormatType::Exact(mode));
    let result = match camera.set_camera_requset(exact) {
        Ok(actual) => Ok(actual),
        Err(exact_err) => {
            log::debug!("exact mode {mode} refused ({exact_err}); trying closest");
            let closest = RequestedFormat::new::<RgbAFormat>(RequestedFormatType::Closest(mode));
            camera.set_camera_requset(closest)
        }
    };

    let mut state = state.lock().unwrap();
    match result {
        Ok(actual) => {
            let res = actual.resolution();
            state.current = (res.width_x, res.height_y);
            state.current_format = Some(actual);
            state.last_error = None;
            log::info!("camera mode now {actual}");
        }
        Err(e) => {
            log::warn!("camera rejected {mode}: {e}");
            state.last_error = Some(format!("{mode} rejected: {e}"));
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
