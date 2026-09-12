//! Application wiring: threads, event loop, and the per-frame pipeline.

use anyhow::{Context, Result};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use winit::application::ApplicationHandler;
use winit::event::{ElementState, WindowEvent};
use winit::event_loop::ActiveEventLoop;
use winit::keyboard::{Key, NamedKey};
use winit::window::{Window, WindowId};

use crate::camera::{self, CameraConfig, FrameBus};
use crate::effect::{self, Effect, EffectCtx};
use crate::gesture::{FingerTouch, GestureEngine, GestureEvent, HandCount};
use crate::region::{RegionSource, TwoHandQuad};
use crate::render::overlay::LineInstance;
use crate::render::{RenderInput, Renderer};
use crate::tracking::hand::{Finger, HandFrame, BONES};
use crate::tracking::{HandTracker, TrackerConfig};

/// Shared latest tracking result.
type HandSlot = Arc<Mutex<Arc<HandFrame>>>;

pub struct AppOptions {
    pub camera: Option<u32>,
}

pub struct App {
    window: Option<Arc<Window>>,
    renderer: Option<Renderer>,

    bus: Arc<FrameBus>,
    hands: HandSlot,
    camera_size: (u32, u32),

    effects: Vec<Box<dyn Effect>>,
    effect_index: usize,
    region_source: Box<dyn RegionSource>,
    gestures: GestureEngine,

    mirror: bool,
    show_skeleton: bool,
    show_outline: bool,

    start: Instant,
    last_frame: Instant,
    fps_counter: (u32, Instant),
    lines: Vec<LineInstance>,
    event_buf: Vec<GestureEvent>,
}

impl App {
    pub fn new(options: AppOptions) -> Result<Self> {
        let bus = Arc::new(FrameBus::default());
        let cfg = CameraConfig {
            index: options.camera.unwrap_or_else(camera::auto_index),
            ..Default::default()
        };
        let camera_size = camera::spawn(cfg, bus.clone())
            .context("starting camera capture")?;

        let hands: HandSlot = Arc::new(Mutex::new(Arc::new(HandFrame::default())));
        spawn_tracking(bus.clone(), hands.clone());

        let effects = effect::registry();
        let region_source: Box<dyn RegionSource> = Box::new(TwoHandQuad::default());

        log::info!(
            "region source: {} | effects: {}",
            region_source.name(),
            effects.iter().map(|e| e.name()).collect::<Vec<_>>().join(", ")
        );

        Ok(Self {
            window: None,
            renderer: None,
            bus,
            hands,
            camera_size,
            effects,
            effect_index: 0,
            region_source,
            gestures: GestureEngine::new()
                // Thumb-to-pinky cycles the effect, as in the reference clip.
                .with(FingerTouch::new(Finger::Thumb, Finger::Pinky))
                .with(HandCount::default()),
            mirror: true,
            show_skeleton: false,
            show_outline: true,
            start: Instant::now(),
            last_frame: Instant::now(),
            fps_counter: (0, Instant::now()),
            lines: Vec::new(),
            event_buf: Vec::new(),
        })
    }

    fn cycle_effect(&mut self, delta: i32) {
        let n = self.effects.len() as i32;
        if n == 0 {
            return;
        }
        self.effect_index = (((self.effect_index as i32 + delta) % n + n) % n) as usize;
        log::info!("effect -> {}", self.effects[self.effect_index].name());
    }

    /// Build the debug skeleton geometry for the current hands.
    fn build_lines(&mut self, hands: &HandFrame) {
        self.lines.clear();
        if !self.show_skeleton {
            return;
        }
        for hand in &hands.hands {
            for (a, b) in BONES {
                self.lines.push(LineInstance::new(
                    hand.point(a),
                    hand.point(b),
                    [0.2, 1.0, 0.7, 0.9],
                    1.5,
                ));
            }
            // Mark the fingertips that drive the region.
            for finger in [Finger::Thumb, Finger::Index] {
                let p = hand.tip(finger);
                let r = 0.006;
                self.lines.push(LineInstance::new(
                    p - glam::Vec2::new(r, 0.0),
                    p + glam::Vec2::new(r, 0.0),
                    [1.0, 0.3, 0.4, 1.0],
                    3.0,
                ));
            }
        }
    }

    fn draw(&mut self) {
        let now = Instant::now();
        let dt = (now - self.last_frame).as_secs_f32();
        self.last_frame = now;

        // Take the latest tracking result and mirror it into screen space if
        // the preview is flipped, so gestures and regions agree with what the
        // user sees.
        let mut hands = (*self.hands.lock().unwrap()).as_ref().clone();
        if self.mirror {
            for hand in &mut hands.hands {
                for p in hand.landmarks.iter_mut() {
                    p.x = 1.0 - p.x;
                }
            }
        }

        // Copy events out before handling them: the returned slice borrows
        // `self.gestures`, and the handlers need `&mut self`. The buffer is
        // reused so this costs no allocation per frame.
        self.event_buf.clear();
        self.event_buf.extend_from_slice(self.gestures.update(&hands, dt));
        for i in 0..self.event_buf.len() {
            match self.event_buf[i] {
                GestureEvent::TouchStart {
                    a: Finger::Thumb,
                    b: Finger::Pinky,
                    ..
                } => {
                    self.cycle_effect(1);
                }
                GestureEvent::HandCountChanged { count } => {
                    log::debug!("hands: {count}");
                }
                _ => {}
            }
        }

        let region = self.region_source.region(&hands);
        let time = self.start.elapsed().as_secs_f32();
        let params = {
            let ctx = EffectCtx {
                hands: &hands,
                region,
                time,
            };
            self.effects[self.effect_index].params(&ctx)
        };

        self.build_lines(&hands);

        let frame = self.bus.latest();
        let Some(renderer) = self.renderer.as_mut() else {
            return;
        };

        let result = renderer.render(RenderInput {
            frame: frame.as_deref(),
            region,
            effect_index: self.effect_index,
            params,
            time,
            mirror: self.mirror,
            show_outline: self.show_outline,
            lines: &self.lines,
        });
        if let Err(e) = result {
            log::error!("render failed: {e:#}");
        }

        // Cheap console HUD; a text overlay would need a font stack.
        self.fps_counter.0 += 1;
        if self.fps_counter.1.elapsed().as_secs_f32() >= 2.0 {
            let fps = self.fps_counter.0 as f32 / self.fps_counter.1.elapsed().as_secs_f32();
            log::info!(
                "{fps:.0} fps | effect: {} | hands: {} | region: {}",
                self.effects[self.effect_index].name(),
                hands.hands.len(),
                if region.is_active() { "on" } else { "off" }
            );
            self.fps_counter = (0, Instant::now());
        }
    }
}

fn spawn_tracking(bus: Arc<FrameBus>, out: HandSlot) {
    std::thread::Builder::new()
        .name("tracking".into())
        .spawn(move || {
            let mut tracker = match HandTracker::new(TrackerConfig::default()) {
                Ok(t) => t,
                Err(e) => {
                    log::error!("hand tracking unavailable: {e:#}");
                    return;
                }
            };
            let mut last_seq = 0;
            let mut last = Instant::now();
            loop {
                let Some(frame) = bus.latest() else {
                    std::thread::sleep(std::time::Duration::from_millis(4));
                    continue;
                };
                if frame.seq == last_seq {
                    // No new camera frame yet; don't burn a core re-running
                    // inference on data we already processed.
                    std::thread::sleep(std::time::Duration::from_millis(1));
                    continue;
                }
                last_seq = frame.seq;

                let dt = last.elapsed().as_secs_f32();
                last = Instant::now();

                match tracker.track(&frame, dt) {
                    Ok(result) => *out.lock().unwrap() = Arc::new(result),
                    Err(e) => log::warn!("tracking step failed: {e:#}"),
                }
            }
        })
        .expect("spawning tracking thread");
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let attrs = Window::default_attributes()
            .with_title("pawcontrol — hand effects")
            .with_inner_size(winit::dpi::LogicalSize::new(
                self.camera_size.0,
                self.camera_size.1,
            ));
        let window = match event_loop.create_window(attrs) {
            Ok(w) => Arc::new(w),
            Err(e) => {
                log::error!("could not create window: {e}");
                event_loop.exit();
                return;
            }
        };

        match pollster::block_on(Renderer::new(
            window.clone(),
            self.camera_size,
            &self.effects,
        )) {
            Ok(r) => {
                log::info!("renderer: {} via {}", r.adapter_name, r.backend);
                self.renderer = Some(r);
            }
            Err(e) => {
                log::error!("could not initialise the renderer: {e:#}");
                event_loop.exit();
                return;
            }
        }
        self.window = Some(window);
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                if let Some(r) = self.renderer.as_mut() {
                    r.resize(size.width, size.height);
                }
            }
            WindowEvent::RedrawRequested => self.draw(),
            WindowEvent::KeyboardInput { event, .. } if event.state == ElementState::Pressed => {
                match event.logical_key.as_ref() {
                    Key::Named(NamedKey::Escape) => event_loop.exit(),
                    Key::Named(NamedKey::Space) => self.cycle_effect(1),
                    Key::Character("m") => self.mirror = !self.mirror,
                    Key::Character("d") => self.show_skeleton = !self.show_skeleton,
                    Key::Character("o") => self.show_outline = !self.show_outline,
                    _ => {}
                }
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(w) = &self.window {
            w.request_redraw();
        }
    }
}
