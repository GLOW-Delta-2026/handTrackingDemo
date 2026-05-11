package pipeline

import (
	"fmt"
	"image"
	"math"

	"gocv.io/x/gocv"
)

// HandLandmarker wraps the MediaPipe hand_landmark model. Input is a
// 224x224 cropped ROI; output is 21 (x,y,z) landmarks in input space plus
// a hand-presence score and a handedness score.
type HandLandmarker struct {
	runner *runner
	inputW int
	inputH int

	// Scratch buffers.
	cropped gocv.Mat
	rgb     gocv.Mat
	input   []float32

	// Output channel sizes — discovered at load time.
	outLandmarks int // expected 21 * 3 = 63
}

// NewHandLandmarker loads hand_landmark_full.tflite (or the lite variant).
func NewHandLandmarker(modelPath string) (*HandLandmarker, error) {
	r, err := newRunner(modelPath, 4)
	if err != nil {
		return nil, fmt.Errorf("hand landmarker: %w", err)
	}
	in := r.interpreter.GetInputTensor(0)
	if in == nil {
		r.close()
		return nil, fmt.Errorf("hand landmarker: missing input tensor")
	}
	shape := in.Shape()
	if len(shape) != 4 || shape[3] != 3 {
		r.close()
		return nil, fmt.Errorf("hand landmarker: unexpected input shape %v", shape)
	}
	h, w := shape[1], shape[2]
	return &HandLandmarker{
		runner:       r,
		inputW:       w,
		inputH:       h,
		cropped:      gocv.NewMat(),
		rgb:          gocv.NewMat(),
		input:        make([]float32, w*h*3),
		outLandmarks: LandmarkCount * 3,
	}, nil
}

// Close releases TFLite resources.
func (l *HandLandmarker) Close() {
	l.cropped.Close()
	l.rgb.Close()
	l.runner.close()
}

// Run extracts the ROI from srcBGR via an affine warp, runs the landmark
// model, and returns the recovered Hand in source-image coordinates. roi
// is in normalized [0,1] coordinates; the source frame is srcW × srcH pixels.
//
// Returns nil if the model's hand-presence score is below scoreThresh.
func (l *HandLandmarker) Run(srcBGR gocv.Mat, roi ROI, srcW, srcH int, scoreThresh float32) (*Hand, error) {
	// 1) Build the affine transform that maps the ROI corners to the model's
	// input rectangle. We use top-left, top-right, bottom-left corners.
	corners := roi.Corners()
	srcPts := []image.Point{
		{X: int(corners[0].X * float32(srcW)), Y: int(corners[0].Y * float32(srcH))},
		{X: int(corners[1].X * float32(srcW)), Y: int(corners[1].Y * float32(srcH))},
		{X: int(corners[3].X * float32(srcW)), Y: int(corners[3].Y * float32(srcH))},
	}
	dstPts := []image.Point{
		{X: 0, Y: 0},
		{X: l.inputW - 1, Y: 0},
		{X: 0, Y: l.inputH - 1},
	}
	srcPV := gocv.NewPointVectorFromPoints(srcPts)
	defer srcPV.Close()
	dstPV := gocv.NewPointVectorFromPoints(dstPts)
	defer dstPV.Close()
	M := gocv.GetAffineTransform(srcPV, dstPV)
	defer M.Close()

	gocv.WarpAffine(srcBGR, &l.cropped, M, image.Pt(l.inputW, l.inputH))

	// 2) Color convert + normalize to [0,1] float32 (hand_landmark expects [0,1]).
	gocv.CvtColor(l.cropped, &l.rgb, gocv.ColorBGRToRGB)
	matToFloat(l.rgb, l.input, 0, 1.0/255.0)

	if err := l.runner.setInputFloat32(0, l.input); err != nil {
		return nil, err
	}
	if err := l.runner.invoke(); err != nil {
		return nil, err
	}

	// 3) Read outputs. Output index conventions for hand_landmark_full:
	//      0: 21 landmarks (x,y,z) in [1, 63] — input-pixel space
	//      1: hand presence score [1, 1] (sigmoid-ready)
	//      2: handedness [1, 1] (0 = left, 1 = right after sigmoid)
	//      3: (full only) world landmarks — ignored
	lmRaw, err := l.runner.outputFloat32(0)
	if err != nil {
		return nil, err
	}
	if len(lmRaw) < l.outLandmarks {
		return nil, fmt.Errorf("hand landmarker: landmark output too small (%d)", len(lmRaw))
	}
	presence, err := l.runner.outputFloat32(1)
	if err != nil {
		return nil, err
	}
	handedness, err := l.runner.outputFloat32(2)
	if err != nil {
		return nil, err
	}

	score := sigmoid(presence[0])
	if score < scoreThresh {
		return nil, nil
	}

	// 4) Recover landmarks: undo the affine to map (x,y) from input space back
	// into source-pixel space. z stays in input-pixel space; that's fine for
	// the classifier since it only uses (x,y).
	hand := &Hand{Score: score}
	if sigmoid(handedness[0]) > 0.5 {
		hand.Handedness = HandednessRight
	} else {
		hand.Handedness = HandednessLeft
	}

	// Use ROI center + rotation to map each landmark back. This is equivalent
	// to the inverse of the affine we used above but avoids a matrix
	// invocation per point.
	cos := float32(math.Cos(float64(roi.Rotation)))
	sin := float32(math.Sin(float64(roi.Rotation)))
	roiCX := roi.CenterX * float32(srcW)
	roiCY := roi.CenterY * float32(srcH)
	roiW := roi.Width * float32(srcW)
	roiH := roi.Height * float32(srcH)

	xMin, yMin := float32(math.MaxFloat32), float32(math.MaxFloat32)
	xMax, yMax := float32(-math.MaxFloat32), float32(-math.MaxFloat32)
	for i := 0; i < LandmarkCount; i++ {
		// Landmarks in input space [0, inputW] × [0, inputH].
		nx := lmRaw[i*3+0]/float32(l.inputW) - 0.5
		ny := lmRaw[i*3+1]/float32(l.inputH) - 0.5
		nz := lmRaw[i*3+2]
		// Scale by ROI dimensions then rotate around ROI center.
		lx := nx * roiW
		ly := ny * roiH
		wx := roiCX + lx*cos - ly*sin
		wy := roiCY + lx*sin + ly*cos
		hand.Landmarks[i] = Landmark{X: wx, Y: wy, Z: nz}

		if wx < xMin {
			xMin = wx
		}
		if wy < yMin {
			yMin = wy
		}
		if wx > xMax {
			xMax = wx
		}
		if wy > yMax {
			yMax = wy
		}
	}
	hand.BBox = Rect{X: int(xMin), Y: int(yMin), W: int(xMax - xMin), H: int(yMax - yMin)}
	return hand, nil
}
