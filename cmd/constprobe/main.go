// constprobe feeds the palm detector a synthetic constant image and prints
// the output distribution. If outputs are still in the thousands, the model
// file or our output reading is the problem. If outputs are small, our
// real-image preprocessing is the problem.
//
// Run: go run ./cmd/constprobe
package main

import (
	"fmt"
	"math"
	"os"
	"sort"

	"github.com/mattn/go-tflite"
)

func main() {
	modelPath := "model/mediapipe/palm_detection_full.tflite"
	if len(os.Args) > 1 {
		modelPath = os.Args[1]
	}

	model := tflite.NewModelFromFile(modelPath)
	if model == nil {
		fmt.Fprintln(os.Stderr, "load failed")
		os.Exit(1)
	}
	defer model.Delete()
	opts := tflite.NewInterpreterOptions()
	defer opts.Delete()
	it := tflite.NewInterpreter(model, opts)
	if it == nil {
		fmt.Fprintln(os.Stderr, "interpreter failed")
		os.Exit(1)
	}
	defer it.Delete()
	if it.AllocateTensors() != tflite.OK {
		fmt.Fprintln(os.Stderr, "allocate failed")
		os.Exit(1)
	}

	// Three test inputs: all zeros, all 0.5, all ones (after [-1,1] mapping).
	tests := []struct {
		name string
		val  float32
	}{
		{"zero (=-1.0 normalized)", -1.0},
		{"gray (=0.0 normalized)", 0.0},
		{"white (=+1.0 normalized)", 1.0},
	}

	in := it.GetInputTensor(0)
	shape := in.Shape()
	n := 1
	for _, d := range shape {
		n *= d
	}
	buf := make([]float32, n)

	for _, t := range tests {
		fmt.Printf("== input = %s ==\n", t.name)
		for i := range buf {
			buf[i] = t.val
		}
		setFloat32(in, buf)
		if it.Invoke() != tflite.OK {
			fmt.Println("invoke failed")
			continue
		}
		scores := readFloat32(it.GetOutputTensor(1))
		boxes := readFloat32(it.GetOutputTensor(0))

		ss := append([]float32(nil), scores...)
		sort.Slice(ss, func(i, j int) bool { return ss[i] > ss[j] })
		fmt.Printf("  scores: min=%.3f median=%.3f max=%.3f, top-5: %v\n",
			ss[len(ss)-1], ss[len(ss)/2], ss[0], ss[:5])

		// Sample a few box rows (first anchor and the top-scoring one).
		top := argmax(scores)
		fmt.Printf("  boxes[0..3]   = %v\n", boxes[:4])
		fmt.Printf("  boxes[top=%d] = %v\n", top, boxes[top*18:top*18+4])
		fmt.Println()
	}
}

func setFloat32(t *tflite.Tensor, data []float32) {
	buf := make([]byte, len(data)*4)
	for i, v := range data {
		bits := math.Float32bits(v)
		buf[i*4+0] = byte(bits)
		buf[i*4+1] = byte(bits >> 8)
		buf[i*4+2] = byte(bits >> 16)
		buf[i*4+3] = byte(bits >> 24)
	}
	t.CopyFromBuffer(buf)
}

func readFloat32(t *tflite.Tensor) []float32 {
	shape := t.Shape()
	n := 1
	for _, d := range shape {
		n *= d
	}
	buf := make([]byte, n*4)
	t.CopyToBuffer(buf)
	out := make([]float32, n)
	for i := 0; i < n; i++ {
		bits := uint32(buf[i*4]) |
			uint32(buf[i*4+1])<<8 |
			uint32(buf[i*4+2])<<16 |
			uint32(buf[i*4+3])<<24
		out[i] = math.Float32frombits(bits)
	}
	return out
}

func argmax(s []float32) int {
	best := float32(math.Inf(-1))
	idx := 0
	for i, v := range s {
		if v > best {
			best = v
			idx = i
		}
	}
	return idx
}
