#!/usr/bin/env bash
# Build libtensorflowlite_c.dylib from source via CMake, install into vendor/tflite/.
# One-time, ~10–15 minutes on Apple Silicon. No Bazel needed.
#
# Output:
#   vendor/tflite/lib/libtensorflowlite_c.dylib
#   vendor/tflite/include/tensorflow/lite/c/c_api.h  (+ deps)
#
# Then `go build` finds it via the cgo directives in pipeline/tflite_cgo.go.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENDOR="$REPO_ROOT/_third_party/tflite"
BUILD_DIR="$REPO_ROOT/_third_party/tflite-build"
TF_VERSION="${TF_VERSION:-v2.16.1}"

if ! command -v cmake >/dev/null 2>&1; then
  echo "cmake not found. Run: brew install cmake" >&2
  exit 1
fi

mkdir -p "$VENDOR/lib" "$VENDOR/include" "$BUILD_DIR"

if [ ! -d "$BUILD_DIR/tensorflow" ]; then
  echo "==> Cloning TensorFlow $TF_VERSION (shallow)"
  git clone --depth 1 --branch "$TF_VERSION" https://github.com/tensorflow/tensorflow.git "$BUILD_DIR/tensorflow"
fi

echo "==> Configuring TFLite C library"
mkdir -p "$BUILD_DIR/out"
cd "$BUILD_DIR/out"
cmake "$BUILD_DIR/tensorflow/tensorflow/lite/c" \
  -DCMAKE_BUILD_TYPE=Release \
  -DTFLITE_ENABLE_XNNPACK=ON

echo "==> Building (this is the slow part)"
cmake --build . -j "$(sysctl -n hw.ncpu)"

echo "==> Installing into $VENDOR"
cp "$BUILD_DIR/out/libtensorflowlite_c.dylib" "$VENDOR/lib/"
# Headers — copy the full tensorflow/lite tree, preserving directory structure.
# The TFLite C API transitively includes builtin_ops.h and many other top-level
# headers, so a partial copy isn't enough.
mkdir -p "$VENDOR/include/tensorflow"
(cd "$BUILD_DIR/tensorflow/tensorflow" && tar cf - $(find lite -name "*.h" -type f)) \
  | (cd "$VENDOR/include/tensorflow" && tar xf -)

echo
echo "==> Done. Library at: $VENDOR/lib/libtensorflowlite_c.dylib"
echo "    Headers at:       $VENDOR/include"
echo
echo "Set CGO flags before building (these are baked into pipeline/tflite_cgo.go too):"
echo "  export CGO_CFLAGS=\"-I$VENDOR/include\""
echo "  export CGO_LDFLAGS=\"-L$VENDOR/lib -ltensorflowlite_c -Wl,-rpath,$VENDOR/lib\""
