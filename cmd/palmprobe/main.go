// palmprobe captures one frame from the webcam, runs palm detection, and
// dumps:
//   - the 192x192 BGR patch that was fed to the model       → debug-palm-input.png
//   - the source frame                                       → debug-source.png
//   - score distribution (min/max/median/top-10) to stderr
//   - top-10 raw box rows (xc, yc, w, h, kp0, kp1) to stderr
//
// Run: go run ./cmd/palmprobe -device=0
package main

import (
	"flag"
	"fmt"
	"log"
	"os"
	"sort"

	"gocv.io/x/gocv"

	"github.com/glow-delta-2026/handtracking/internal/pipeline"
)

func main() {
	device := flag.Int("device", 0, "camera device")
	width := flag.Int("width", 960, "capture width")
	height := flag.Int("height", 540, "capture height")
	palmModel := flag.String("palm-model", "model/mediapipe/palm_detection_full.tflite", "model path")
	bias := flag.Float64("bias", -127.5, "per-pixel bias added before scale")
	scale := flag.Float64("scale", 1.0/127.5, "per-pixel scale applied after bias")
	flag.Parse()

	cap, err := gocv.OpenVideoCapture(*device)
	if err != nil {
		log.Fatal(err)
	}
	defer cap.Close()
	cap.Set(gocv.VideoCaptureFrameWidth, float64(*width))
	cap.Set(gocv.VideoCaptureFrameHeight, float64(*height))

	frame := gocv.NewMat()
	defer frame.Close()

	// Warm up — first few frames from a webcam are sometimes blank.
	for i := 0; i < 5; i++ {
		cap.Read(&frame)
	}
	if !cap.Read(&frame) || frame.Empty() {
		log.Fatal("could not read frame")
	}
	mirrored := gocv.NewMat()
	defer mirrored.Close()
	gocv.Flip(frame, &mirrored, 1)

	det, err := pipeline.NewPalmDetector(*palmModel)
	if err != nil {
		log.Fatal(err)
	}
	defer det.Close()
	det.SetNormalization(float32(*bias), float32(*scale))

	dets, err := det.Detect(mirrored)
	if err != nil {
		log.Fatal(err)
	}

	// Dump first 30 floats of the normalized input — they should ALL be in
	// [-1, 1]. If any exceeds that range, matToFloat has a bug.
	in := det.LastNormalizedInput()
	fmt.Fprintln(os.Stderr, "first 12 normalized input values (should be in [-1, 1]):")
	for i := 0; i < 12 && i < len(in); i++ {
		fmt.Fprintf(os.Stderr, "  in[%d] = %.4f\n", i, in[i])
	}
	var minV, maxV float32 = in[0], in[0]
	for _, v := range in {
		if v < minV {
			minV = v
		}
		if v > maxV {
			maxV = v
		}
	}
	fmt.Fprintf(os.Stderr, "input range: min=%.4f max=%.4f (expect [-1, 1])\n\n", minV, maxV)

	// Save the source frame + the model-input patch.
	if ok := gocv.IMWrite("debug-source.png", mirrored); !ok {
		log.Println("save source failed")
	}
	if ok := gocv.IMWrite("debug-palm-input.png", det.LastInputBGR()); !ok {
		log.Println("save palm input failed")
	}

	rawScores := det.LastRawScores()
	rawBoxes := det.LastRawBoxes()

	// Score distribution.
	scores := make([]float32, len(rawScores))
	copy(scores, rawScores)
	sort.Slice(scores, func(i, j int) bool { return scores[i] > scores[j] })
	min := scores[len(scores)-1]
	max := scores[0]
	median := scores[len(scores)/2]
	fmt.Fprintf(os.Stderr, "scores: min=%.3f median=%.3f max=%.3f (raw logits)\n", min, median, max)
	fmt.Fprintln(os.Stderr, "top-10 raw scores:")
	for i := 0; i < 10 && i < len(scores); i++ {
		fmt.Fprintf(os.Stderr, "  [%d] = %.4f (sigmoid = %.4f)\n", i, scores[i], sigmoid(scores[i]))
	}

	// Top-10 raw boxes (by score), with their anchor index.
	type indexed struct {
		idx   int
		score float32
	}
	idxs := make([]indexed, len(rawScores))
	for i, s := range rawScores {
		idxs[i] = indexed{i, s}
	}
	sort.Slice(idxs, func(i, j int) bool { return idxs[i].score > idxs[j].score })
	anchors := pipeline.GenerateAnchors(pipeline.PalmDetectionAnchorOptions())
	fmt.Fprintln(os.Stderr, "top-10 raw boxes (anchor index, raw x,y,w,h pre-decode, then decoded):")
	for i := 0; i < 10 && i < len(idxs); i++ {
		ai := idxs[i].idx
		base := ai * 18
		rx := rawBoxes[base+0]
		ry := rawBoxes[base+1]
		rw := rawBoxes[base+2]
		rh := rawBoxes[base+3]
		a := anchors[ai]
		dx := rx/192*a.W + a.XCenter
		dy := ry/192*a.H + a.YCenter
		dw := rw / 192 * a.W
		dh := rh / 192 * a.H
		fmt.Fprintf(os.Stderr, "  anchor[%4d]@(%.2f,%.2f) raw=(%7.2f, %7.2f, %7.2f, %7.2f) → (%.3f, %.3f, %.3f, %.3f)  score=%.3f\n",
			ai, a.XCenter, a.YCenter, rx, ry, rw, rh, dx, dy, dw, dh, idxs[i].score)
	}

	fmt.Fprintf(os.Stderr, "\npost-NMS detections: %d\n", len(dets))
}

func sigmoid(x float32) float32 {
	if x > 100 {
		x = 100
	} else if x < -100 {
		x = -100
	}
	return 1.0 / (1.0 + float32_exp(-x))
}

func float32_exp(x float32) float32 {
	return float32(safe_exp(float64(x)))
}

func safe_exp(x float64) float64 {
	if x > 100 {
		return 2.6881171418161356e+43
	}
	if x < -100 {
		return 0
	}
	return mathExp(x)
}

// avoid pulling in math just to call Exp.
func mathExp(x float64) float64 {
	// Use accurate Taylor-ish via math package.
	return mathExpStdlib(x)
}
