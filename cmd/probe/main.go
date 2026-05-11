// Probe a TFLite model and print its input/output tensor shapes and types.
// Run: go run ./cmd/probe model/mediapipe/palm_detection_full.tflite
package main

import (
	"fmt"
	"os"

	"github.com/mattn/go-tflite"
)

func main() {
	if len(os.Args) < 2 {
		fmt.Fprintln(os.Stderr, "usage: probe <model.tflite> [more.tflite ...]")
		os.Exit(2)
	}
	for _, path := range os.Args[1:] {
		fmt.Printf("== %s ==\n", path)
		dump(path)
		fmt.Println()
	}
}

func dump(path string) {
	m := tflite.NewModelFromFile(path)
	if m == nil {
		fmt.Println("  load FAILED")
		return
	}
	defer m.Delete()
	opts := tflite.NewInterpreterOptions()
	defer opts.Delete()
	it := tflite.NewInterpreter(m, opts)
	if it == nil {
		fmt.Println("  interpreter FAILED")
		return
	}
	defer it.Delete()
	if it.AllocateTensors() != tflite.OK {
		fmt.Println("  allocate FAILED")
		return
	}
	nin := it.GetInputTensorCount()
	nout := it.GetOutputTensorCount()
	fmt.Printf("  %d input(s), %d output(s)\n", nin, nout)
	for i := 0; i < nin; i++ {
		t := it.GetInputTensor(i)
		fmt.Printf("  in[%d]: name=%q type=%v shape=%v\n", i, t.Name(), t.Type(), t.Shape())
	}
	for i := 0; i < nout; i++ {
		t := it.GetOutputTensor(i)
		fmt.Printf("  out[%d]: name=%q type=%v shape=%v\n", i, t.Name(), t.Type(), t.Shape())
	}
}
