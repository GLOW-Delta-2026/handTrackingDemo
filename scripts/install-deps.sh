#!/usr/bin/env bash
# Install native dependencies for the handtracking Go app on macOS.
# Run once before `go run .`
set -euo pipefail

if ! command -v brew >/dev/null 2>&1; then
  echo "Homebrew not found. Install from https://brew.sh and re-run." >&2
  exit 1
fi

echo "==> Installing OpenCV + pkg-config (for gocv)"
brew install opencv pkg-config

echo
echo "==> OpenCV install summary"
pkg-config --modversion opencv4 || true

echo
echo "==> TensorFlow Lite C library (libtensorflowlite_c)"
echo "    The handtracking app uses go-tflite, which links against"
echo "    libtensorflowlite_c.dylib. There is no official brew formula."
echo "    Choose ONE of the following:"
echo
echo "    1) Build from TF source (slow, ~30 min): bazel build for tensorflow/lite/c"
echo "    2) Use a prebuilt binary: see scripts/install-tflite.sh (not yet provided)"
echo
echo "    Phase 1 (webcam loop + FPS overlay) does NOT need TFLite."
echo "    Skip this until you start on the hand-detection phase."
