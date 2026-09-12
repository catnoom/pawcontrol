# pawcontrol

Real-time camera effects driven by hand and finger tracking, in Rust.

The built-in effect mirrors the reference clip: hold both hands up and a quad is
drawn between your index and thumb tips, with everything inside it pixelated.
Touch thumb to pinky to cycle to the next effect; pinch thumb to index to dial
the current effect's intensity.

## Running

```sh
cargo run --release
```

| Flag | Meaning |
| --- | --- |
| `--camera N` | use capture device N (default: first non-virtual device) |
| `--list-cameras` | print detected devices and exit |
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
| thumb + index | adjust current effect's intensity |
| `space` | next effect (keyboard fallback) |
| `d` | toggle the debug skeleton |
| `o` | toggle the region outline |
| `m` | toggle mirroring |
| `esc` | quit |

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
the shader as `g.params`; `EffectCtx::pinch_knob()` gives you a 0..1 value from
the user's pinch.

### A new gesture

Implement `GestureDetector` in `src/gesture/` and push it into the
`GestureEngine` in `src/app.rs`. Express thresholds as multiples of
`Hand::scale()` (palm width) so the gesture behaves the same near and far from
the camera, and give it hysteresis — separate enter/exit thresholds — or it will
flicker at the boundary.

### A new region shape

Implement `RegionSource` in `src/region/`. `Region::Quad` is masked by the
shader via a signed distance field that works with either winding order.
`SingleHandBox` is included as a one-handed example.

## Tests

```sh
cargo test
```

26 tests covering the ROI geometry, anchor decoding, NMS, the smoothing filter,
gesture hysteresis and region winding. One further test runs the full
detection → crop → landmark chain against a real photograph; it is skipped
unless you point it at an image:

```sh
PAWCONTROL_TEST_IMAGE=/path/to/hand.jpg cargo test
```

## Requirements

- Windows (capture uses Media Foundation; inference uses DirectML)
- A GPU — AMD, NVIDIA or integrated. `WGPU_BACKEND=dx12|vulkan` forces the
  rendering backend if you need to.
