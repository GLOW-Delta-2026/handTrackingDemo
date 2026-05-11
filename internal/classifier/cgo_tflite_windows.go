package classifier

// Mirror of internal/pipeline/cgo_tflite_windows.go — keep both in sync.

// #cgo windows CFLAGS: -I${SRCDIR}/../../_third_party/tflite/include
// #cgo windows LDFLAGS: -L${SRCDIR}/../../_third_party/tflite/lib -ltensorflowlite_c
import "C"
