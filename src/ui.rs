//! The control panel: an egui overlay for every runtime tunable.
//!
//! Components expose their own knobs via `Tunable`, so this module renders
//! whatever the effect, gesture and tracking layers declare rather than
//! hard-coding a list that drifts out of date.

use egui::{Context, ViewportId};
use winit::event::WindowEvent;
use winit::window::Window;

use crate::camera::CameraHandle;
use crate::effect::Effect;
use crate::gesture::GestureEngine;
use crate::settings::{Shared, TrackingSettings};

/// Tessellated panel geometry, handed to the renderer.
pub struct UiOutput {
    pub primitives: Vec<egui::ClippedPrimitive>,
    pub textures_delta: egui::TexturesDelta,
    pub pixels_per_point: f32,
}

/// Read-only figures shown at the top of the panel.
pub struct Stats {
    pub fps: f32,
    pub hands: usize,
    pub region_active: bool,
    pub adapter: String,
    pub inference_backend: String,
}

/// Everything the panel may edit, borrowed from `App` for one frame.
pub struct PanelState<'a> {
    pub effects: &'a mut [Box<dyn Effect>],
    pub effect_index: &'a mut usize,
    pub knob: &'a mut f32,
    pub knob_manual: &'a mut bool,
    pub mirror: &'a mut bool,
    pub show_skeleton: &'a mut bool,
    pub show_outline: &'a mut bool,
    pub outline_color: &'a mut [f32; 3],
    pub outline_width: &'a mut f32,
    pub gestures: &'a mut GestureEngine,
    pub tracking: &'a Shared<TrackingSettings>,
    pub camera: &'a CameraHandle,
    pub stats: Stats,
}

pub struct Ui {
    ctx: Context,
    state: egui_winit::State,
    /// Whether the panel is showing. Toggled with `h`.
    pub open: bool,
}

impl Ui {
    pub fn new(window: &Window) -> Self {
        let ctx = Context::default();
        let state = egui_winit::State::new(
            ctx.clone(),
            ViewportId::ROOT,
            window,
            Some(window.scale_factor() as f32),
            None,
            None,
        );
        Self {
            ctx,
            state,
            open: true,
        }
    }

    /// Feed a window event to egui. Returns true when egui consumed it, in
    /// which case the app should not also act on it.
    pub fn on_window_event(&mut self, window: &Window, event: &WindowEvent) -> bool {
        self.state.on_window_event(window, event).consumed
    }

    pub fn run(&mut self, window: &Window, mut panel: PanelState<'_>) -> UiOutput {
        let raw_input = self.state.take_egui_input(window);
        let open = self.open;
        let full_output = self.ctx.run_ui(raw_input, |ui| {
            let ctx = ui.ctx().clone();
            let ctx = &ctx;
            if open {
                draw_panel(ctx, &mut panel);
            }
        });

        self.state
            .handle_platform_output(window, full_output.platform_output);

        let pixels_per_point = full_output.pixels_per_point;
        UiOutput {
            primitives: self.ctx.tessellate(full_output.shapes, pixels_per_point),
            textures_delta: full_output.textures_delta,
            pixels_per_point,
        }
    }
}

fn draw_panel(ctx: &Context, panel: &mut PanelState<'_>) {
    egui::Window::new("pawcontrol")
        .default_width(330.0)
        .default_pos([12.0, 12.0])
        .show(ctx, |ui| {
            let s = &panel.stats;
            ui.label(format!(
                "{:.0} fps   ·   {} hand{}   ·   region {}",
                s.fps,
                s.hands,
                if s.hands == 1 { "" } else { "s" },
                if s.region_active { "on" } else { "off" }
            ));
            ui.label(
                egui::RichText::new(format!("{} · {}", s.adapter, s.inference_backend))
                    .small()
                    .weak(),
            );
            ui.separator();

            effect_section(ui, panel);
            ui.separator();
            display_section(ui, panel);
            ui.separator();
            gesture_section(ui, panel);
            ui.separator();
            camera_section(ui, panel);
            ui.separator();
            tracking_section(ui, panel);

            ui.separator();
            ui.label(
                egui::RichText::new("h hides this panel · esc quits")
                    .small()
                    .weak(),
            );
        });
}

fn effect_section(ui: &mut egui::Ui, panel: &mut PanelState<'_>) {
    ui.heading("effect");

    let current = *panel.effect_index;
    egui::ComboBox::from_label("active")
        .selected_text(panel.effects[current].name().to_owned())
        .show_ui(ui, |ui| {
            for (i, effect) in panel.effects.iter().enumerate() {
                ui.selectable_value(panel.effect_index, i, effect.name());
            }
        });

    ui.checkbox(panel.knob_manual, "set intensity by hand (override gesture)");
    ui.add_enabled(
        *panel.knob_manual,
        egui::Slider::new(panel.knob, 0.0..=1.0).text("intensity"),
    );
    if !*panel.knob_manual {
        ui.label(
            egui::RichText::new(format!("curl-driven: {:.2}", *panel.knob))
                .small()
                .weak(),
        );
    }

    let index = *panel.effect_index;
    for t in panel.effects[index].tunables() {
        ui.add(egui::Slider::new(t.value, t.min..=t.max).text(t.label));
    }
}

fn display_section(ui: &mut egui::Ui, panel: &mut PanelState<'_>) {
    ui.heading("display");
    ui.checkbox(panel.mirror, "mirror preview");
    ui.checkbox(panel.show_skeleton, "hand skeleton");
    ui.checkbox(panel.show_outline, "region outline");
    ui.horizontal(|ui| {
        ui.color_edit_button_rgb(panel.outline_color);
        ui.add(egui::Slider::new(panel.outline_width, 0.2..=8.0).text("outline px"));
    });
}

fn gesture_section(ui: &mut egui::Ui, panel: &mut PanelState<'_>) {
    ui.heading("gestures");
    for (name, tunables) in panel.gestures.tunables() {
        ui.label(egui::RichText::new(name).small().weak());
        for t in tunables {
            ui.add(egui::Slider::new(t.value, t.min..=t.max).text(t.label));
        }
    }
}

fn camera_section(ui: &mut egui::Ui, panel: &mut PanelState<'_>) {
    ui.heading("camera");
    let state = panel.camera.state();
    let (w, h) = state.current;

    // Fall back to the current mode alone if the device would not enumerate.
    let mut options = state.resolutions();
    if options.is_empty() {
        options.push((w, h));
    }

    let mut chosen = (w, h);
    egui::ComboBox::from_label("resolution")
        .selected_text(format!("{w} x {h}"))
        .show_ui(ui, |ui| {
            for option in &options {
                ui.selectable_value(
                    &mut chosen,
                    *option,
                    format!("{} x {}", option.0, option.1),
                );
            }
        });
    // Requesting the mode already in use would pointlessly restart the stream.
    if chosen != (w, h) {
        panel.camera.set_resolution(chosen.0, chosen.1);
    }

    if let Some(err) = &state.last_error {
        ui.colored_label(egui::Color32::from_rgb(220, 120, 90), err);
    }
    // Showing the pixel format matters: high resolutions are usually only
    // offered as MJPEG, so this explains why a mode was or was not available.
    if let Some(mode) = &state.current_format {
        ui.label(
            egui::RichText::new(format!(
                "{:?} · {} fps",
                mode.format(),
                mode.frame_rate()
            ))
            .small()
            .weak(),
        );
    }
    ui.label(
        egui::RichText::new("switching restarts the capture stream")
            .small()
            .weak(),
    );
}

fn tracking_section(ui: &mut egui::Ui, panel: &mut PanelState<'_>) {
    ui.heading("tracking");
    let Ok(mut t) = panel.tracking.lock() else {
        ui.label("settings unavailable");
        return;
    };

    ui.add(egui::Slider::new(&mut t.max_hands, 1..=2).text("max hands"));
    ui.add(egui::Slider::new(&mut t.presence_threshold, 0.1..=0.95).text("keep hand above"));
    ui.add(egui::Slider::new(&mut t.palm_score_threshold, 0.1..=0.95).text("detect palm above"));
    ui.add(egui::Slider::new(&mut t.redetect_interval, 1..=30).text("redetect every N frames"));

    ui.label(egui::RichText::new("smoothing (one euro)").small().weak());
    ui.add(
        egui::Slider::new(&mut t.filter.min_cutoff, 0.1..=10.0)
            .text("min cutoff — lower is smoother at rest"),
    );
    ui.add(
        egui::Slider::new(&mut t.filter.beta, 0.0..=80.0)
            .text("beta — higher cuts lag when moving"),
    );
    ui.add(egui::Slider::new(&mut t.filter.derivative_cutoff, 0.1..=10.0).text("speed cutoff"));
}
