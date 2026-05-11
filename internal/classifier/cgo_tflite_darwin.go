package classifier

// CGo CFLAGS don't propagate across packages, so any package that imports
// go-tflite needs its own copy of the include path. Mirror of
// internal/pipeline/cgo_tflite_darwin.go — keep both in sync.

// #cgo darwin CFLAGS: -I${SRCDIR}/../../_third_party/tflite/include
// #cgo darwin LDFLAGS: -L${SRCDIR}/../../_third_party/tflite/lib -ltensorflowlite_c -Wl,-rpath,${SRCDIR}/../../_third_party/tflite/lib
import "C"
