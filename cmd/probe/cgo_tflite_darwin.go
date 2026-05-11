package main

// #cgo darwin CFLAGS: -I${SRCDIR}/../../_third_party/tflite/include
// #cgo darwin LDFLAGS: -L${SRCDIR}/../../_third_party/tflite/lib -ltensorflowlite_c -Wl,-rpath,${SRCDIR}/../../_third_party/tflite/lib
import "C"
