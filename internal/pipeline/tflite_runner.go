package pipeline

import (
	"fmt"
	"unsafe"

	"github.com/mattn/go-tflite"
)

// runner is a thin wrapper around a TFLite interpreter with float32 in/out.
// It owns the model + interpreter and frees them on Close.
type runner struct {
	model       *tflite.Model
	options     *tflite.InterpreterOptions
	interpreter *tflite.Interpreter
}

func newRunner(modelPath string, numThreads int) (*runner, error) {
	model := tflite.NewModelFromFile(modelPath)
	if model == nil {
		return nil, fmt.Errorf("load model %s", modelPath)
	}
	opts := tflite.NewInterpreterOptions()
	if numThreads > 0 {
		opts.SetNumThread(numThreads)
	}
	interp := tflite.NewInterpreter(model, opts)
	if interp == nil {
		opts.Delete()
		model.Delete()
		return nil, fmt.Errorf("create interpreter for %s", modelPath)
	}
	if status := interp.AllocateTensors(); status != tflite.OK {
		interp.Delete()
		opts.Delete()
		model.Delete()
		return nil, fmt.Errorf("allocate tensors for %s: status=%v", modelPath, status)
	}
	return &runner{model: model, options: opts, interpreter: interp}, nil
}

func (r *runner) close() {
	if r.interpreter != nil {
		r.interpreter.Delete()
	}
	if r.options != nil {
		r.options.Delete()
	}
	if r.model != nil {
		r.model.Delete()
	}
}

// setInputFloat32 copies float32 data into the i-th input tensor. The tensor
// itself may be float32 or uint8 (quantized); we handle both.
func (r *runner) setInputFloat32(i int, data []float32) error {
	t := r.interpreter.GetInputTensor(i)
	if t == nil {
		return fmt.Errorf("no input tensor at index %d", i)
	}
	switch t.Type() {
	case tflite.Float32:
		if s := t.CopyFromBuffer(float32SliceAsBytes(data)); s != tflite.OK {
			return fmt.Errorf("input tensor copy: status=%v", s)
		}
		return nil
	case tflite.UInt8:
		q := t.QuantizationParams()
		buf := make([]byte, len(data))
		for k, v := range data {
			qv := int32(v/float32(q.Scale)) + int32(q.ZeroPoint)
			if qv < 0 {
				qv = 0
			} else if qv > 255 {
				qv = 255
			}
			buf[k] = byte(qv)
		}
		if s := t.CopyFromBuffer(buf); s != tflite.OK {
			return fmt.Errorf("input tensor copy: status=%v", s)
		}
		return nil
	default:
		return fmt.Errorf("unsupported input tensor type %v", t.Type())
	}
}

// invoke runs the interpreter.
func (r *runner) invoke() error {
	if status := r.interpreter.Invoke(); status != tflite.OK {
		return fmt.Errorf("invoke: status=%v", status)
	}
	return nil
}

// outputFloat32 returns the i-th output tensor as a freshly allocated float32
// slice. Handles both float32 and uint8 (dequantized) outputs.
func (r *runner) outputFloat32(i int) ([]float32, error) {
	t := r.interpreter.GetOutputTensor(i)
	if t == nil {
		return nil, fmt.Errorf("no output tensor at index %d", i)
	}
	shape := t.Shape()
	n := 1
	for _, d := range shape {
		n *= d
	}
	switch t.Type() {
	case tflite.Float32:
		out := make([]float32, n)
		if s := t.CopyToBuffer(float32SliceAsBytes(out)); s != tflite.OK {
			return nil, fmt.Errorf("output tensor copy: status=%v", s)
		}
		return out, nil
	case tflite.UInt8:
		buf := make([]byte, n)
		if s := t.CopyToBuffer(buf); s != tflite.OK {
			return nil, fmt.Errorf("output tensor copy: status=%v", s)
		}
		q := t.QuantizationParams()
		out := make([]float32, n)
		for k, b := range buf {
			out[k] = (float32(int32(b)) - float32(q.ZeroPoint)) * float32(q.Scale)
		}
		return out, nil
	default:
		return nil, fmt.Errorf("unsupported output tensor type %v", t.Type())
	}
}

// float32SliceAsBytes reinterprets a []float32 as []byte with no copy.
func float32SliceAsBytes(s []float32) []byte {
	if len(s) == 0 {
		return nil
	}
	return unsafe.Slice((*byte)(unsafe.Pointer(&s[0])), len(s)*4)
}
