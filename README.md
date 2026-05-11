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

> ### ⚠️ Read this first — the shell matters
>
> **Every single command in this Windows section must be run inside the "MSYS2 MINGW64" shell.** Not PowerShell. Not Command Prompt. Not "MSYS2 MSYS", "MSYS2 UCRT64", "MSYS2 CLANG64", or "MSYS2 CLANGARM64".
>
> If you don't already have it: install [MSYS2](https://www.msys2.org/) (run the installer, accept the defaults). Then launch **"MSYS2 MINGW64"** from the Start menu. The prompt should look like this, with `MINGW64` in **green**:
>
> ```
> user@machine MINGW64 ~
> $
> ```
>
> If your prompt says `MSYS`, `UCRT64`, `CLANG64`, `CLANGARM64`, or `PS C:\…>`, **stop** and open the correct shell before running anything below. Running these commands in any other terminal will fail with errors like `go: command not found`, missing libraries, or broken builds.

### Step 0 — Refresh MSYS2 (only the first time after installing it)

A fresh MSYS2 ships with a stale package database. Inside **MSYS2 MINGW64**:

```bash
pacman -Syu
```

If it asks you to close the terminal mid-way, close it, reopen **MSYS2 MINGW64**, and run it again:

```bash
pacman -Syu
```

### Step 1 — Install Go + OpenCV + toolchain (~3–5 min)

Still inside **MSYS2 MINGW64**:

```bash
pacman -S --needed \
  mingw-w64-x86_64-go \
  mingw-w64-x86_64-toolchain \
  mingw-w64-x86_64-cmake \
  mingw-w64-x86_64-opencv \
  mingw-w64-x86_64-pkgconf \
  make \
  git
```

> **Do not install Go via the Windows `.msi` from go.dev.** The MSYS2 MINGW64 shell has its own PATH and won't see a Windows-side Go install — `go` would still report "command not found". The `mingw-w64-x86_64-go` package above is the one that works here.

### Step 2 — Clone the repo and download TFLite (~30 seconds)

> **The repo path must not contain spaces.** Some of the local fallback build scripts (FP16, etc.) don't handle paths with spaces well, so clone somewhere like `C:/glow/handTrackingDemo` or your home directory — **not** under a path like `Semester 6/...`.

Still inside **MSYS2 MINGW64**:

```bash
cd /c/glow              # or any path without spaces
git clone https://github.com/GLOW-Delta-2026/handTrackingDemo.git
cd handTrackingDemo
./scripts/install-tflite.sh
```

`install-tflite.sh` on Windows downloads a prebuilt `libtensorflowlite_c.dll` (built by GitHub Actions on a clean MSYS2 MINGW64 runner) plus the matching headers from the latest [GitHub release](https://github.com/GLOW-Delta-2026/handTrackingDemo/releases). This avoids the ~15-minute TensorFlow source build that's brittle on MinGW (CMake 4.x policy issues, MSVC-only flags in third-party CMakeLists, etc.). If the download fails for any reason, the script automatically falls back to a from-source build with all known workarounds applied. To force a from-source build: `TFLITE_BUILD_FROM_SOURCE=1 ./scripts/install-tflite.sh`.

### Step 3 — Build the demo

Still inside **MSYS2 MINGW64**:

```bash
make build
```

This uses the cross-platform Makefile, which on Windows sets the right `-tags customenv` + cgo env vars for gocv and go-tflite, runs `go build`, and copies `libtensorflowlite_c.dll` next to the produced `handtracking.exe` (Windows has no rpath, so the DLL has to live in the same folder as the .exe).

> **Don't run a bare `go build` on Windows.** It will fail with `tensorflow/lite/c/c_api.h: No such file or directory` and `opencv2/opencv.hpp: No such file or directory` — gocv's default Windows cgo directives expect OpenCV at `C:/opencv/...`, not MSYS2's `/mingw64/...`, and `CGO_CPPFLAGS` is not set without the Makefile. Use `make build`.

### Step 4 — Run

Still inside **MSYS2 MINGW64**:

```bash
./handtracking.exe
```

Then open <http://localhost:8080> in any browser.

### Troubleshooting

- **`go: command not found` or `'go' is not recognized as a cmdlet…`** — Almost always means you're not in the MSYS2 MINGW64 shell. The prompt must read `MINGW64` (in green) — if it says `PS C:\…>` you're in PowerShell. Close it, open **MSYS2 MINGW64** from the Start menu, and re-run from there. If you confirmed you're in MINGW64 and it still says command not found, run `pacman -S mingw-w64-x86_64-go`. Installing Go through the Windows `.msi` from go.dev does **not** help here — MSYS2 has its own PATH.
- **`pkg-config: command not found`** — Same cause; you missed `mingw-w64-x86_64-pkgconf` in the pacman line.
- **`could not satisfy dependencies … mingw-w64-x86_64-pkg-config`** — Old package name. The current MSYS2 name is `mingw-w64-x86_64-pkgconf` (no hyphen). Use that.
- **`failed to prepare transaction` / `could not satisfy dependencies`** — A fresh MSYS2 install ships with a stale package database. Run `pacman -Syu` first (twice if it asks you to close the terminal mid-way), then re-run the install line.
- **gocv `cannot find -lopencv_*`** — `mingw-w64-x86_64-opencv` didn't install. Re-run the pacman line.
- **CMake error during `./scripts/install-tflite.sh`: `Compatibility with CMake < 3.5 has been removed`** (typically in `FP16-source/CMakeLists.txt`) — modern CMake (4.x, the version MSYS2 ships now) refuses TensorFlow v2.16.1's older third-party deps. The install script already passes `-DCMAKE_POLICY_VERSION_MINIMUM=3.5` to work around this; if you cloned before that fix, run `git pull` and re-run `./scripts/install-tflite.sh`.
- **CMake error: `ADD_SUBDIRECTORY given source "…/psimd-source" which is not an existing directory`** — almost always means **your repo path contains a space**. TensorFlow's older third-party CMake scripts (FP16's `DownloadPSimd.cmake`) don't quote paths, so a space anywhere in the path breaks the nested CMake invocation that fetches `psimd`. Move the repo to a path without spaces (`C:/glow/handTrackingDemo`, your home dir, etc.) and re-run `./scripts/install-tflite.sh`. The install script now refuses to run from a path with spaces, so you get a clear error early instead of this confusing one.
- **`tensorflow/lite/c/c_api.h: No such file or directory`** and/or **`opencv2/opencv.hpp: No such file or directory`** during `go build` — you ran a bare `go build` instead of `make build`. The Makefile is what sets the cgo env vars (`CGO_CPPFLAGS`, `CGO_LDFLAGS`) and the `-tags customenv` flag that lets gocv look up OpenCV via MSYS2's `pkg-config`. Run `make build` instead. If `make: command not found`, install it with `pacman -S make`.

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
