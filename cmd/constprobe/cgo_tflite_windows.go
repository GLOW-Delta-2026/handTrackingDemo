package main

// #cgo windows CFLAGS: -I${SRCDIR}/../../_third_party/tflite/include
// #cgo windows LDFLAGS: -L${SRCDIR}/../../_third_party/tflite/lib -ltensorflowlite_c
import "C"
