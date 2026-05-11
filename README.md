# HandTracking Demo — Quantum Waves

A live hand-tracking demo built as a concept for **GLOW Eindhoven 2026 — "Connect" / Quantum Waves**.

A webcam tracks your hand, and a browser tab renders your hand as a hydrogen wave function — a probability-density cloud of an electron orbital. Smaller "wave-value" orbs drift around with elastic collisions; the main orb eats them and climbs an 18-state quantum-number chain (1s → 2s → 2p_z → 2p_xy → 3s → … → 4f). If you stop eating, the orbital collapses one rung at a time, with a slow back-and-forth morph between the current and next-simpler shape as warning.

## What it does

- macOS or Windows webcam → 21 hand landmarks at ~30 fps (Go + OpenCV + MediaPipe palm-detection and hand-landmark TFLite models, ported to Go)
- Adaptive [One-Euro filter](https://gery.casiez.net/1euro/) smoothing on landmarks (snappy on fast motion, calm at rest)
- An embedded HTTP server broadcasts hand state to the browser via Server-Sent Events
- The web app renders `|ψ_nlm|²` (hydrogen orbital probability density) with a magma colormap, follows your index fingertip via a critically-damped spring, and slowly morphs between orbital states on eat / decay

## Run

After setup (see below), from the repo root:

```bash
make build       # macOS
./handtracking
```

```cmd
go build -o handtracking.exe .   :: Windows (see Windows setup for env vars)
handtracking.exe
```

Then open <http://localhost:8080>. ESC in the OpenCV preview window quits the app.

---

## Setup — macOS

Tested on Apple Silicon, macOS 14+. Any Mac with a webcam should work.

```bash
# 1. Go 1.23+ (skip if already installed)
brew install go

# 2. OpenCV + pkg-config — gocv depends on these (~500 MB, ~2 min)
./scripts/install-deps.sh

# 3. TFLite C library — built from TensorFlow source via CMake (~10–15 min, one-time)
brew install cmake
./scripts/install-tflite.sh
```

That populates `_third_party/tflite/` with `libtensorflowlite_c.dylib` and headers. Then:

```bash
make build
./handtracking
```

The first run prompts for camera access — grant it.

---

## Setup — Windows

Tested on Windows 11 with a built-in webcam.

### Prerequisites

1. **Go 1.23+** — <https://go.dev/dl/>
2. **Visual Studio Build Tools 2022** with the "Desktop development with C++" workload — <https://aka.ms/vs/17/release/vs_BuildTools.exe>
3. **CMake** — <https://cmake.org/download/> (tick "Add to PATH" during install)
4. **Git** — <https://git-scm.com/download/win>
5. **MinGW-w64** — easiest via [MSYS2](https://www.msys2.org/), then `pacman -S mingw-w64-x86_64-toolchain`. gocv on Windows links through MinGW.

### 1. OpenCV (via gocv installer)

[gocv](https://gocv.io) ships a PowerShell helper that downloads and builds a pinned OpenCV. From an **admin PowerShell**:

```powershell
git clone https://github.com/hybridgroup/gocv.git $env:USERPROFILE\gocv
cd $env:USERPROFILE\gocv
.\win_build_opencv.cmd
```

It places DLLs under `C:\opencv\build\install\x64\mingw\bin`. Add that directory to your **User PATH**.

### 2. TFLite C library

The macOS shell script doesn't work on Windows, but the same CMake recipe does. From a **"x64 Native Tools Command Prompt for VS 2022"**:

```cmd
cd %USERPROFILE%
git clone --depth 1 --branch v2.16.1 https://github.com/tensorflow/tensorflow.git tf
mkdir tflite-build && cd tflite-build
cmake %USERPROFILE%\tf\tensorflow\lite\c -DCMAKE_BUILD_TYPE=Release -DTFLITE_ENABLE_XNNPACK=ON
cmake --build . --config Release -j
```

When it finishes:

1. Copy `Release\tensorflowlite_c.dll` (and `tensorflowlite_c.lib`) to `<repo>\_third_party\tflite\lib\`.
2. Copy the header tree: every `*.h` under `%USERPROFILE%\tf\tensorflow\lite\` into `<repo>\_third_party\tflite\include\tensorflow\lite\`, preserving directory layout.

### 3. Build the demo

From a Developer Command Prompt at the repo root:

```cmd
set CGO_CFLAGS=-I%CD%\_third_party\tflite\include
set CGO_LDFLAGS=-L%CD%\_third_party\tflite\lib -ltensorflowlite_c
go build -o handtracking.exe .
```

Either copy `tensorflowlite_c.dll` next to `handtracking.exe`, or add `_third_party\tflite\lib` to PATH. Then:

```cmd
handtracking.exe
```

Open <http://localhost:8080> in any browser.

---

## What you'll see

- **OpenCV preview window** — your webcam mirrored, with hand skeleton, bounding box, hand-sign and finger-gesture labels, FPS, and a small HUD. ESC to quit.
- **Browser tab at `localhost:8080`** — the Quantum Waves visualization. Top-left HUD shows the current orbital (`1s`, `2p_z`, `3d_x²-y²`, …), the decay timer, and the classified hand sign and gesture. Bottom-right legend explains the four orb types: cyan +1, green +2, yellow +3, pink +4 (quantum-number jumps along the complexity chain).

Move your hand around to swoop the orb through small wave orbs. Hold position for a few seconds and watch the orbital morph and collapse back toward 1s. Eat an orb during the pulse warning to cancel the collapse and bump the orbital higher.

## Flags

| Flag | Default | Description |
| --- | --- | --- |
| `-http-addr` | `:8080` | Web app bind address; empty string disables the web side |
| `-device` | `0` | Webcam index (first connected = 0) |
| `-width` `-height` | 960 × 540 | Capture resolution |
| `-min-detection-confidence` | `0.85` | Palm-detection presence threshold |
| `-min-tracking-confidence` | `0.7` | Hand-landmark presence threshold |
| `-smooth-min-cutoff` | `0.5` | One-Euro min cutoff (Hz); lower = smoother at rest. `0` disables |
| `-smooth-beta` | `5.0` | One-Euro speed coefficient; higher = snappier on fast motion |
| `-debug` | `false` | Stderr diagnostics per frame |

## Keys (OpenCV window)

| Key | Action |
| --- | --- |
| `ESC` | Quit |
| `n` | Normal mode |
| `k` | Log keypoint training data → `model/keypoint_classifier/keypoint.csv` |
| `h` | Log point-history training data → `model/point_history_classifier/point_history.csv` |
| `0`..`9` | Class id while a logging mode is active |

## Project layout

```
.
├── main.go                          # capture loop, classifier glue, CSV logging, web broadcast
├── fps.go                           # sliding-window FPS
├── internal/
│   ├── pipeline/                    # Go port of MediaPipe Hands (palm + landmark + ROI tracking)
│   ├── classifier/                  # KeyPointClassifier + PointHistoryClassifier (TFLite MLPs)
│   ├── render/                      # gocv-based drawing for the OpenCV preview
│   ├── smooth/                      # One-Euro adaptive filter for landmark smoothing
│   └── server/                      # SSE web server + embedded Quantum Waves web app
│       ├── server.go
│       └── web/index.html           # Hydrogen orbital visualization (Canvas + SSE)
├── model/                           # MediaPipe + classifier TFLite models
├── scripts/                         # install-deps.sh, install-tflite.sh (macOS)
└── _third_party/tflite/             # libtensorflowlite_c + headers (gitignored — built locally)
```

## Tests

```bash
make test
```

Covers the pure-math parts of the pipeline: SSD anchor generation, box decoding (with `reverse_output_order`), NMS weighted merge, ROI rotation/square-long logic, and the landmark normalization that feeds the MLPs.

## Acknowledgements

- Pipeline derived from [Kazuhito00/hand-gesture-recognition-using-mediapipe](https://github.com/Kazuhito00/hand-gesture-recognition-using-mediapipe) (Apache 2.0) via the [GLOW-Delta-2026/HandTracking](https://github.com/GLOW-Delta-2026/HandTracking) fork.
- MediaPipe palm + hand-landmark models from Google's [MediaPipe](https://github.com/google-ai-edge/mediapipe) (Apache 2.0).
- [gocv](https://gocv.io) — Go bindings for OpenCV.
- [mattn/go-tflite](https://github.com/mattn/go-tflite) — Go bindings for TFLite.
- One-Euro Filter: Casiez et al., *1€ Filter*, CHI 2012.
- Hydrogen orbital math from the standard Schrödinger-equation solutions; visualization styled after the perceptually-uniform [magma](https://matplotlib.org/stable/users/explain/colors/colormaps.html) colormap.
