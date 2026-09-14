# pawcontrol

Real-time camera effects driven by hand and finger tracking, in Rust.

Hold both hands up and a quad is
drawn between your index and thumb tips, with everything inside it pixelated.
Touch thumb to pinky to cycle to the next effect. Curl your middle finger to
dial the current effect's intensity — that control is live only while the
window is up, so it holds its setting between poses.

## Running

```sh
cargo run --release
```

| Flag | Meaning |
| --- | --- |
| `--camera N` | use capture device N (default: first non-virtual device) |
| | (resolution is changed from the control panel, not the CLI) |
| `--list-cameras` | print detected devices and exit |
| `--list-modes` | print the camera's supported modes and exit |
| `--selftest` | verify the models and GPU backend without opening the camera |

Start with `--selftest`. It reports which execution provider actually loaded,
which is the thing most likely to differ between machines:

```
inference backend: DirectML
palm detection:    1.7 ms
hand landmarks:    2.4 ms
```

### Controls

| Input | Action |
| --- | --- |
| thumb + pinky | next effect |
| curl middle finger | intensity, 0 extended -> 1 fully curled (needs the window up) |
| `space` | next effect (keyboard fallback) |
| long right-eye blink | freeze / unfreeze the zone |
| `f` | freeze the zone in place (hands can then move away) |
| `h` | show/hide the control panel |
| `d` | toggle the debug skeleton |
| `o` | toggle the region outline |
| `m` | toggle mirroring |
| `esc` | quit |

## Control panel

Press `h` for an on-screen panel covering every runtime tunable: effect
selection and each effect's own parameters, the intensity knob (with a manual
override for the gesture), outline colour and width, gesture thresholds and
debounce, the camera resolution, and the tracking settings — hand limit,
confidence thresholds, re-detection interval and the One Euro smoothing
constants.

The resolution list comes from the device itself, so it only offers modes the
camera actually supports. Switching restarts the capture stream and resizes the
GPU texture; the panel reports the mode the driver settled on, which is not
always the one requested.

A webcam mode is a *triple* — resolution, pixel format and frame rate — not
just a size. High resolutions are usually offered only as MJPEG, since
uncompressed modes run out of USB bandwidth, so picking 1080p also means
switching pixel format. The panel shows the format and frame rate in use, and
`--list-modes` prints everything the device advertises.

Components declare their own knobs by returning `Tunable`s, so a new effect or
gesture gets panel controls without touching the UI code:

```rust
fn tunables(&mut self) -> Vec<Tunable<'_>> {
    vec![Tunable::new("block size", &mut self.block, 1.0, 64.0)]
}
```

Tracking settings live behind a `Shared<TrackingSettings>` because tracking runs
on its own thread; it re-reads them once per frame, so edits apply live.

## How it works

Three threads, each running at its own rate and never blocking the others:

```
camera thread ──▶ [latest frame] ──▶ inference thread ──▶ [latest hands]
                        │                                       │
                        └────────────▶ render thread (vsync) ◀───┘
```

Both handoffs are single-slot mailboxes that keep only the newest value, so a
slow consumer drops frames instead of building a backlog.

Each frame flows through four independent stages:

| Stage | Module | Responsibility |
| --- | --- | --- |
| tracking | `src/tracking/` | frame → 21 landmarks per hand |
| gesture | `src/gesture/` | landmarks → discrete events |
| region | `src/region/` | landmarks → a masked shape |
| effect | `src/effect/` | shape → what to draw inside it |

### Face tracking

A second two-stage pair runs alongside the hands, on the same thread: a
256x256 BlazeFace detector and a 192x192 mesh model emitting 468 landmarks.
These are **NCHW**, unlike the hand models, so crops are written plane by plane.

Eye openness is the eye aspect ratio (EAR) — the eyelid gap over the eye's
width. Dividing by the width makes it scale-free, so it reads the same near or
far from the camera; a raw pixel gap would not. Open eyes measure about 0.3,
closed under 0.15.

The blink trigger uses hysteresis (separate shut/open thresholds) so an eye
hovering at the boundary does not chatter, and a hold duration well above an
involuntary blink (~0.1-0.4s). Losing the face *resets* the timer rather than
counting as a closure, so tracking dropouts cannot be mistaken for a
deliberate blink.

Note that "right eye" follows MediaPipe's convention — the user's own right,
which appears on the left of an unmirrored image.

One trap worth recording: the face mesh emits **normalized** 0..1 crop
coordinates, while the hand landmark model emits crop *pixels*. The two
decoders differ for that reason.

### Hand tracking

Two ONNX models, the standard MediaPipe pair, run through `ort` with the
DirectML execution provider (AMD-friendly; falls back to CPU automatically):

1. **Palm detection** (192×192) — an SSD detector over 2016 anchors. Expensive,
   so it only runs when we are short of hands, every few frames.
2. **Hand landmarks** (224×224) — 21 joints from a tight, upright crop.

In the steady state palm detection never runs: each frame's crop is re-derived
from the previous frame's landmarks. Landmarks are smoothed with a
[One Euro filter](https://gery.casiez.net/1euro/), which smooths hard when the
hand is still and backs off when it moves, so the quad neither jitters nor lags.

Models live in `assets/models/` and are embedded in the binary at compile time,
so there is no runtime asset path to get wrong.

## Adding things

### A new effect

Drop a WGSL file in `assets/shaders/effects/` defining one function:

```wgsl
fn effect(uv: vec2<f32>) -> vec3<f32> {
    return 1.0 - cam(uv);   // invert
}
```

It is compiled between a shared prelude (which supplies `cam()`,
`resolution()`, `time()`, `region_center()` and the region mask) and the entry
points, so you only write the interesting part. Then implement `Effect` and add
it to `registry()` in `src/effect/mod.rs`. `params()` feeds up to four floats to
the shader as `g.params`; `EffectCtx::knob()` gives you the 0..1 intensity
value.

The knob reads middle-finger curl, deliberately *not* index or thumb: those two
fingertips define the quad, so using them would make the knob and the window
the same control. It is measured in palm-widths (see
`Finger::extended_reference`), so it does not drift as you move toward or away
from the camera, and `update_knob` freezes it whenever the region is off.

### A new gesture

Implement `GestureDetector` in `src/gesture/` and push it into the
`GestureEngine` in `src/app.rs`. Express thresholds as multiples of
`Hand::scale()` (palm width) so the gesture behaves the same near and far from
the camera, and give it hysteresis — separate enter/exit thresholds — or it will
flicker at the boundary.

### A new region shape

Implement `RegionSource` in `src/region/`, and build quads with
`Region::quad()` rather than `Region::Quad` directly: it takes the convex hull
of your corners, so the shape survives corners arriving in any order. That
matters — flipping one hand swings its thumb above its index, and a fixed
corner order would trace a self-intersecting bow-tie that the mask cannot
resolve.

## Tests

```sh
cargo test
```

57 tests covering the ROI geometry, anchor decoding, NMS, the smoothing filter,
gesture hysteresis, region convexity, the knob's curl calibration, and eye
aspect ratio and blink timing. Two tests run against a real GPU device — one compiles every effect
shader, the other drives the egui paint path — so rendering errors surface here
rather than when the window opens. Two further tests run the full detection → crop → landmark chain against
a real photograph, including a sweep over hand orientations; they are skipped
unless you point them at an image:

```sh
PAWCONTROL_TEST_IMAGE=/path/to/hand.jpg PAWCONTROL_FACE_IMAGE=/path/to/face.jpg cargo test
```

## Requirements

- Windows (capture uses Media Foundation; inference uses DirectML)
- A GPU — AMD, NVIDIA or integrated. `WGPU_BACKEND=dx12|vulkan` forces the
  rendering backend if you need to.
