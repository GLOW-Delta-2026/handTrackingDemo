// Package classifier wraps the two small TFLite MLPs shipped in the original
// Python project: a static-pose classifier (KeyPointClassifier) and a
// finger-gesture classifier driven by index-fingertip trajectory
// (PointHistoryClassifier).
package classifier

import (
	"bufio"
	"encoding/csv"
	"fmt"
	"math"
	"os"
	"strings"
	"unsafe"

	"github.com/mattn/go-tflite"
)

// Classifier wraps a single-input single-output TFLite MLP that takes a 1-D
// float32 feature vector and returns class probabilities.
type Classifier struct {
	model       *tflite.Model
	options     *tflite.InterpreterOptions
	interpreter *tflite.Interpreter
	labels      []string

	scoreThresh  float32
	invalidValue int
}

// New loads a TFLite MLP and its sibling label CSV. labelPath may be empty —
// in that case labels are auto-named "class_0", "class_1", ...
func New(modelPath, labelPath string) (*Classifier, error) {
	model := tflite.NewModelFromFile(modelPath)
	if model == nil {
		return nil, fmt.Errorf("classifier: load %s", modelPath)
	}
	opts := tflite.NewInterpreterOptions()
	opts.SetNumThread(1) // MLPs are tiny; threading hurts.
	interp := tflite.NewInterpreter(model, opts)
	if interp == nil {
		opts.Delete()
		model.Delete()
		return nil, fmt.Errorf("classifier: build interpreter for %s", modelPath)
	}
	if interp.AllocateTensors() != tflite.OK {
		interp.Delete()
		opts.Delete()
		model.Delete()
		return nil, fmt.Errorf("classifier: allocate tensors for %s", modelPath)
	}

	labels, err := readLabels(labelPath)
	if err != nil {
		interp.Delete()
		opts.Delete()
		model.Delete()
		return nil, err
	}

	return &Classifier{
		model:        model,
		options:      opts,
		interpreter:  interp,
		labels:       labels,
		scoreThresh:  0.0, // 0 = no thresholding (KeyPointClassifier behavior).
		invalidValue: 0,
	}, nil
}

// SetScoreThreshold matches PointHistoryClassifier's behavior: if the
// best-class probability is below thresh, return invalidValue instead.
func (c *Classifier) SetScoreThreshold(thresh float32, invalidValue int) {
	c.scoreThresh = thresh
	c.invalidValue = invalidValue
}

// Labels returns the loaded label list.
func (c *Classifier) Labels() []string { return c.labels }

// Close releases the underlying interpreter, options and model.
func (c *Classifier) Close() {
	if c.interpreter != nil {
		c.interpreter.Delete()
	}
	if c.options != nil {
		c.options.Delete()
	}
	if c.model != nil {
		c.model.Delete()
	}
}

// Classify runs the MLP on features and returns the argmax class index and
// its probability.
func (c *Classifier) Classify(features []float32) (int, float32, error) {
	in := c.interpreter.GetInputTensor(0)
	if in == nil {
		return 0, 0, fmt.Errorf("classifier: no input tensor")
	}
	if s := in.CopyFromBuffer(float32SliceAsBytes(features)); s != tflite.OK {
		return 0, 0, fmt.Errorf("classifier: input copy status=%v", s)
	}
	if c.interpreter.Invoke() != tflite.OK {
		return 0, 0, fmt.Errorf("classifier: invoke failed")
	}
	out := c.interpreter.GetOutputTensor(0)
	if out == nil {
		return 0, 0, fmt.Errorf("classifier: no output tensor")
	}
	shape := out.Shape()
	n := 1
	for _, d := range shape {
		n *= d
	}
	scores := make([]float32, n)
	if s := out.CopyToBuffer(float32SliceAsBytes(scores)); s != tflite.OK {
		return 0, 0, fmt.Errorf("classifier: output copy status=%v", s)
	}
	idx, best := argmax(scores)
	if c.scoreThresh > 0 && best < c.scoreThresh {
		return c.invalidValue, best, nil
	}
	return idx, best, nil
}

func argmax(s []float32) (int, float32) {
	best := float32(math.Inf(-1))
	bestIdx := 0
	for i, v := range s {
		if v > best {
			best = v
			bestIdx = i
		}
	}
	return bestIdx, best
}

func readLabels(path string) ([]string, error) {
	if path == "" {
		return nil, nil
	}
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("classifier: open labels %s: %w", path, err)
	}
	defer f.Close()
	// The upstream Python opens these as utf-8-sig, so we strip a leading BOM
	// if present.
	br := bufio.NewReader(f)
	bom, _ := br.Peek(3)
	if len(bom) == 3 && bom[0] == 0xEF && bom[1] == 0xBB && bom[2] == 0xBF {
		_, _ = br.Discard(3)
	}
	r := csv.NewReader(br)
	r.FieldsPerRecord = -1
	rows, err := r.ReadAll()
	if err != nil {
		return nil, fmt.Errorf("classifier: parse labels: %w", err)
	}
	var labels []string
	for _, row := range rows {
		if len(row) == 0 {
			continue
		}
		labels = append(labels, strings.TrimSpace(row[0]))
	}
	return labels, nil
}

func float32SliceAsBytes(s []float32) []byte {
	if len(s) == 0 {
		return nil
	}
	return unsafe.Slice((*byte)(unsafe.Pointer(&s[0])), len(s)*4)
}
