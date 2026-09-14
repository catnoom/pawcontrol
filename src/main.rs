//! Hand-tracked camera effects.
//!
//! Pipeline: camera -> hand landmarks -> gestures -> region -> masked effect.
//! Each of those stages is a trait with swappable implementations; see the
//! `gesture`, `region` and `effect` modules to add new behaviour.

mod app;
mod camera;
mod effect;
mod frame;
mod gesture;
mod region;
mod render;
mod settings;
mod tracking;
mod ui;

#[cfg(test)]
mod test_support {
    use std::sync::{Mutex, MutexGuard};

    /// Serializes tests that drive the GPU.
    ///
    /// Several tests each stand up their own device — two create a wgpu
    /// device, three create DirectML ONNX sessions. Running those together on
    /// one adapter reliably trips a device reset
    /// (`DXGI_ERROR_DEVICE_REMOVED`). That is an artifact of the harness, not
    /// the app: at runtime there is exactly one wgpu device and one pair of
    /// sessions, never several being built at once.
    static GPU: Mutex<()> = Mutex::new(());

    /// Poisoning is ignored: a panicking GPU test should fail on its own
    /// assertion, not cascade into every other test.
    pub fn gpu_lock() -> MutexGuard<'static, ()> {
        GPU.lock().unwrap_or_else(|e| e.into_inner())
    }
}

use anyhow::Result;
use winit::event_loop::{ControlFlow, EventLoop};

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.iter().any(|a| a == "--help" || a == "-h") {
        print_usage();
        return Ok(());
    }
    if args.iter().any(|a| a == "--list-cameras") {
        for name in camera::list() {
            println!("{name}");
        }
        return Ok(());
    }
    if args.iter().any(|a| a == "--list-modes") {
        let index = args
            .iter()
            .position(|a| a == "--camera")
            .and_then(|i| args.get(i + 1))
            .and_then(|v| v.parse::<u32>().ok())
            .unwrap_or_else(camera::auto_index);
        let cfg = camera::CameraConfig {
            index,
            ..Default::default()
        };
        println!("camera {index} advertises:");
        for mode in camera::list_modes(&cfg)? {
            println!(
                "  {:>5} x {:<5}  {:>3} fps  {:?}",
                mode.resolution().width_x,
                mode.resolution().height_y,
                mode.frame_rate(),
                mode.format()
            );
        }
        return Ok(());
    }
    if args.iter().any(|a| a == "--selftest") {
        return selftest();
    }

    // `--camera N` overrides the auto-selected device.
    let camera = args
        .iter()
        .position(|a| a == "--camera")
        .and_then(|i| args.get(i + 1))
        .map(|v| {
            v.parse::<u32>()
                .map_err(|_| anyhow::anyhow!("--camera expects a number, got {v:?}"))
        })
        .transpose()?;

    let event_loop = EventLoop::new()?;
    // Poll rather than Wait: frames keep arriving from the camera whether or
    // not the OS sends us window events.
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = app::App::new(app::AppOptions { camera })?;
    event_loop.run_app(&mut app)?;
    Ok(())
}

fn print_usage() {
    println!(
        "pawcontrol — hand-tracked camera effects\n\n\
         USAGE:\n\
         \x20 pawcontrol [--camera N]\n\n\
         OPTIONS:\n\
         \x20 --camera N       use this capture device (default: first non-virtual)\n\
         \x20 --list-cameras   print detected capture devices and exit\n\
         \x20 --list-modes     print the camera's supported modes and exit\n\
         \x20 --selftest       verify models and GPU backend, then exit\n\
         \x20 --help           show this message\n\n\
         CONTROLS:\n\
         \x20 thumb + pinky    cycle effect (or press space)\n\
         \x20 curl middle      dial intensity (only while the window is up)\n\
         \x20 long eye blink   freeze / unfreeze the zone\n\
         \x20 f                freeze / unfreeze the zone\n\
         \x20 h                show/hide the control panel\n\
         \x20 d / o / m        toggle skeleton / outline / mirror\n\
         \x20 esc              quit"
    );
}

/// Verify the inference path without touching the camera.
///
/// Reports which execution provider actually loaded, which is the thing most
/// likely to differ between machines.
fn selftest() -> Result<()> {
    use frame::Frame;
    use tracking::{HandTracker, TrackerConfig};

    println!("cameras detected: {:?}", camera::list());

    // A flat grey frame: we are checking plumbing, not detection quality.
    let mut probe_frame = Frame::new(640, 480);
    probe_frame.rgba.fill(128);

    let mut tracker = HandTracker::new(
        TrackerConfig::default(),
        settings::shared(settings::TrackingSettings::default()),
    )?;
    // First pass includes lazy allocation; report a warm one.
    let _ = tracker.probe(&probe_frame)?;
    let report = tracker.probe(&probe_frame)?;

    println!("inference backend: {}", report.backend);
    println!("palm detection:    {:.1} ms", report.palm_ms);
    println!("hand landmarks:    {:.1} ms", report.landmark_ms);
    println!("palm candidates:   {}", report.palm_candidates);
    println!("landmark score:    {:.3}", report.landmark_score);
    println!("\nself-test passed: both models loaded and ran.");
    Ok(())
}
