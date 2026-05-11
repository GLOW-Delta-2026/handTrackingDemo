package pipeline

// Wire the locally-built libtensorflowlite_c.dll into the linker. Windows
// has no rpath equivalent — at runtime the DLL must sit next to the .exe
// or live on PATH. Build the lib first with `scripts/install-tflite.sh`
// from the MSYS2 MINGW64 shell.

// #cgo windows CFLAGS: -I${SRCDIR}/../../_third_party/tflite/include
// #cgo windows LDFLAGS: -L${SRCDIR}/../../_third_party/tflite/lib -ltensorflowlite_c
import "C"
