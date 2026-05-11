#!/usr/bin/env bash
# Build libtensorflowlite_c from source via CMake, install into _third_party/tflite/.
# One-time, ~10–15 minutes. Works on macOS (produces .dylib) and on Windows
# under MSYS2 MINGW64 (produces .dll).
#
# Output:
#   _third_party/tflite/lib/libtensorflowlite_c.{dylib,dll,so}
#   _third_party/tflite/include/tensorflow/lite/c/c_api.h  (+ deps)
#
# Then `go build` finds it via the cgo directives in cgo_tflite_*.go.
set -euo pipefail

case "$(uname -s)" in
  Darwin*)        LIB_NAME="libtensorflowlite_c.dylib" ;;
  MINGW*|MSYS*)   LIB_NAME="libtensorflowlite_c.dll" ;;
  Linux*)         LIB_NAME="libtensorflowlite_c.so" ;;
  *)              echo "Unsupported platform: $(uname -s)" >&2; exit 1 ;;
esac

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
  -DTFLITE_ENABLE_XNNPACK=ON \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5

echo "==> Building (this is the slow part)"
if command -v nproc >/dev/null 2>&1; then NJOBS="$(nproc)"
elif command -v sysctl >/dev/null 2>&1; then NJOBS="$(sysctl -n hw.ncpu)"
else NJOBS=4; fi
cmake --build . -j "$NJOBS" --config Release

echo "==> Installing into $VENDOR"
# CMake on different platforms drops the artifact in different subdirs.
LIB_SRC=""
for candidate in \
  "$BUILD_DIR/out/$LIB_NAME" \
  "$BUILD_DIR/out/Release/$LIB_NAME" \
  "$BUILD_DIR/out/Release/tensorflowlite_c.dll"; do
  if [ -f "$candidate" ]; then LIB_SRC="$candidate"; break; fi
done
if [ -z "$LIB_SRC" ]; then
  echo "Could not locate built library; expected one of:" >&2
  echo "  $BUILD_DIR/out/$LIB_NAME" >&2
  echo "  $BUILD_DIR/out/Release/$LIB_NAME" >&2
  exit 1
fi
cp "$LIB_SRC" "$VENDOR/lib/$LIB_NAME"
# Headers — copy the full tensorflow/lite tree, preserving directory structure.
# The TFLite C API transitively includes builtin_ops.h and many other top-level
# headers, so a partial copy isn't enough.
mkdir -p "$VENDOR/include/tensorflow"
(cd "$BUILD_DIR/tensorflow/tensorflow" && tar cf - $(find lite -name "*.h" -type f)) \
  | (cd "$VENDOR/include/tensorflow" && tar xf -)

echo
echo "==> Done. Library at: $VENDOR/lib/$LIB_NAME"
echo "    Headers at:       $VENDOR/include"
echo
echo "The cgo bridges under internal/{pipeline,classifier} already point at"
echo "$VENDOR — just run \`go build .\` next."
case "$(uname -s)" in
  MINGW*|MSYS*)
    echo
    echo "Note (Windows): also copy the DLL next to the .exe before running:"
    echo "  cp $VENDOR/lib/$LIB_NAME ./"
    ;;
esac
