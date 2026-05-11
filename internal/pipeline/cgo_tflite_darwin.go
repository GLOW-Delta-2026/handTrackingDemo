package pipeline

// Wire the locally-built libtensorflowlite_c.dylib into the linker.
// The path is resolved at build time relative to this file's directory
// (${SRCDIR}). Build the lib first with `./scripts/install-tflite.sh`.

// #cgo darwin CFLAGS: -I${SRCDIR}/../../_third_party/tflite/include
// #cgo darwin LDFLAGS: -L${SRCDIR}/../../_third_party/tflite/lib -ltensorflowlite_c -Wl,-rpath,${SRCDIR}/../../_third_party/tflite/lib
import "C"
