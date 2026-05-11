package pipeline

import (
	"fmt"

	"gocv.io/x/gocv"
)

// PalmDetector wraps the MediaPipe palm_detection model.
type PalmDetector struct {
	runner       *runner
	anchors      []Anchor
	decodeOpts   DecodeOptions
	inputW       int
	inputH       int
	nmsThreshold float32

	// Scratch buffers, reused across frames.
	resized gocv.Mat
	rgb     gocv.Mat
	input   []float32

	// Per-pixel normalization: (pixel + normBias) * normScale. Default
	// is [0, 1] — the range palm_detection_full.tflite (mediapipe-assets)
	// actually expects.
	normBias  float32
	normScale float32

	// Letterbox transform from the last Detect() call. lastScale is the
	// uniform scale from source to model input; (lastOffX, lastOffY) is the
	// top-left of the resized source inside the model-input square.
	lastOffX  int
	lastOffY  int
	lastScale float32

	// DEBUG mirrors of the last inference's raw tensors.
	lastRawBoxes  []float32
	lastRawScores []float32
}

// SetNormalization overrides the default (bias=-127.5, scale=1/127.5 →
// [-1, 1]). Used by diagnostic tools to test alternate ranges.
func (p *PalmDetector) SetNormalization(bias, scale float32) {
	p.normBias = bias
	p.normScale = scale
}

// NewPalmDetector loads palm_detection_full.tflite (or the lite variant — they
// share anchor layout and decode options).
func NewPalmDetector(modelPath string) (*PalmDetector, error) {
	r, err := newRunner(modelPath, 4)
	if err != nil {
		return nil, fmt.Errorf("palm detector: %w", err)
	}
	// Probe the input tensor for dimensions — the lite + full variants are
	// both 192x192 but we don't want to silently break if Google ships a
	// different shape down the road.
	in := r.interpreter.GetInputTensor(0)
	if in == nil {
		r.close()
		return nil, fmt.Errorf("palm detector: missing input tensor")
	}
	shape := in.Shape()
	if len(shape) != 4 || shape[3] != 3 {
		r.close()
		return nil, fmt.Errorf("palm detector: unexpected input shape %v", shape)
	}
	h, w := shape[1], shape[2]
	opts := PalmDetectionAnchorOptions()
	opts.InputSizeWidth = w
	opts.InputSizeHeight = h
	decOpts := PalmDetectionDecodeOptions()
	decOpts.XScale = float32(w)
	decOpts.YScale = float32(h)
	decOpts.WScale = float32(w)
	decOpts.HScale = float32(h)
	return &PalmDetector{
		runner:       r,
		anchors:      GenerateAnchors(opts),
		decodeOpts:   decOpts,
		inputW:       w,
		inputH:       h,
		nmsThreshold: 0.3,
		resized:      gocv.NewMat(),
		rgb:          gocv.NewMat(),
		input:        make([]float32, w*h*3),
		// palm_detection_full.tflite (mediapipe-assets) uses [0, 1] in
		// practice, despite older MediaPipe pbtxt docs saying [-1, 1].
		// Empirically verified — feeding [-1, 1] explodes outputs to
		// the thousands; [0, 1] keeps logits in the [-20, 0] range.
		normBias:  0,
		normScale: 1.0 / 255.0,
	}, nil
}

// Close releases the underlying TFLite resources.
func (p *PalmDetector) Close() {
	p.resized.Close()
	p.rgb.Close()
	p.runner.close()
}

// Detect runs palm detection on a BGR uint8 source frame and returns
// post-NMS detections in normalized [0,1] coordinates of the source frame.
func (p *PalmDetector) Detect(srcBGR gocv.Mat) ([]Detection, error) {
	// Letterbox-resize to preserve aspect ratio (MediaPipe does the same).
	// Stash the transform so post-decode coordinates can be mapped back to
	// source-frame pixels.
	p.resized.Close()
	offX, offY, scale := letterboxFit(srcBGR, &p.resized, p.inputW, p.inputH)
	p.lastOffX, p.lastOffY, p.lastScale = offX, offY, scale
	gocv.CvtColor(p.resized, &p.rgb, gocv.ColorBGRToRGB)
	matToFloat(p.rgb, p.input, p.normBias, p.normScale)

	if err := p.runner.setInputFloat32(0, p.input); err != nil {
		return nil, err
	}
	if err := p.runner.invoke(); err != nil {
		return nil, err
	}

	// Output 0: raw boxes  [1, num_anchors, 18]
	// Output 1: raw scores [1, num_anchors,  1]
	rawBoxes, err := p.runner.outputFloat32(0)
	if err != nil {
		return nil, err
	}
	rawScores, err := p.runner.outputFloat32(1)
	if err != nil {
		return nil, err
	}
	if got, want := len(rawScores), len(p.anchors); got != want {
		return nil, fmt.Errorf("palm detector: anchor mismatch — model emits %d scores, generator made %d", got, want)
	}

	// DEBUG: stash the raw outputs and the resized BGR input for the caller
	// to inspect when something looks off.
	p.lastRawBoxes = rawBoxes
	p.lastRawScores = rawScores

	dets := Decode(p.anchors, rawBoxes, rawScores, p.decodeOpts)
	return WeightedNMS(dets, p.nmsThreshold), nil
}

// LastRawBoxes returns the raw box tensor from the most recent Detect call —
// for diagnostics only.
func (p *PalmDetector) LastRawBoxes() []float32 { return p.lastRawBoxes }

// LastRawScores returns the raw score tensor from the most recent Detect call.
func (p *PalmDetector) LastRawScores() []float32 { return p.lastRawScores }

// LastInputBGR returns the 192x192 BGR Mat actually fed to the model
// (post-resize, pre-normalization). The returned Mat shares memory with the
// detector's scratch buffer — clone it before saving or holding.
func (p *PalmDetector) LastInputBGR() gocv.Mat { return p.resized }

// LastNormalizedInput returns the float32 buffer that was copied into the
// model's input tensor. For diagnostics only.
func (p *PalmDetector) LastNormalizedInput() []float32 { return p.input }

// LastLetterbox returns the (offX, offY, scale) from the most recent Detect.
// Use it to map model-space coordinates ([0,1] inside the 192x192 input) back
// to source-frame pixel coordinates:
//
//	srcX = (modelX*192 - offX) / scale
//	srcY = (modelY*192 - offY) / scale
func (p *PalmDetector) LastLetterbox() (offX, offY int, scale float32) {
	return p.lastOffX, p.lastOffY, p.lastScale
}

// InputSize returns the model's expected input resolution.
func (p *PalmDetector) InputSize() (w, h int) { return p.inputW, p.inputH }
