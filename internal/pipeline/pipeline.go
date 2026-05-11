package pipeline

import (
	"fmt"

	"gocv.io/x/gocv"
)

func (p *Pipeline) debugf(format string, args ...any) {
	if p.Debug != nil {
		p.Debug(fmt.Sprintf(format, args...))
	}
}

// Pipeline runs the full MediaPipe Hands inference graph end-to-end:
// palm detection (or tracker reuse) → ROI extraction → hand landmark
// refinement.
type Pipeline struct {
	palm     *PalmDetector
	landmark *HandLandmarker

	// Tracking state — when the last frame produced a confident hand we skip
	// palm detection and reuse the ROI computed from those landmarks.
	prevROI    *ROI
	hasPrevROI bool

	// Tunables.
	MinPalmScore     float32 // 0.7 default; below this we drop the palm detection
	MinLandmarkScore float32 // 0.5 default; below this we drop the hand and re-run palm next frame

	// Plausibility bounds for the landmark-derived bbox, as a fraction of the
	// source frame area. Hands smaller than MinBBoxFrac or larger than
	// MaxBBoxFrac get rejected — these usually mean the model is hallucinating
	// landmarks on a face, arm, or large background object.
	MinBBoxFrac float32
	MaxBBoxFrac float32

	// Debug: if Debug != nil it receives a one-line summary per frame.
	Debug func(string)
}

// NewPipeline loads both TFLite models. modelDir should contain
// palm_detection_full.tflite and hand_landmark_full.tflite (the names baked
// into Pipeline.LoadDefault below).
func NewPipeline(palmPath, landmarkPath string) (*Pipeline, error) {
	p, err := NewPalmDetector(palmPath)
	if err != nil {
		return nil, err
	}
	l, err := NewHandLandmarker(landmarkPath)
	if err != nil {
		p.Close()
		return nil, err
	}
	return &Pipeline{
		palm:             p,
		landmark:         l,
		MinPalmScore:     0.7,
		MinLandmarkScore: 0.5,
		MinBBoxFrac:      0.005, // ~0.5% of frame; smaller than a fingertip-sized blob
		MaxBBoxFrac:      0.6,   // a single hand realistically can't cover >60% of frame
	}, nil
}

// Close releases TFLite + gocv resources.
func (p *Pipeline) Close() {
	if p.palm != nil {
		p.palm.Close()
	}
	if p.landmark != nil {
		p.landmark.Close()
	}
}

// Process runs the pipeline on a single BGR frame. Returns one Hand per
// detected hand (currently at most 1 — we only feed the highest-confidence
// palm detection to the landmark model, matching MediaPipe's single-hand
// mode).
func (p *Pipeline) Process(frame gocv.Mat) ([]Hand, error) {
	w, h := frame.Cols(), frame.Rows()

	var roi ROI
	var haveROI bool

	// 1) If we have a recent confident ROI, reuse it (skip palm detection).
	if p.hasPrevROI {
		roi = *p.prevROI
		haveROI = true
	}

	// 2) Otherwise (or if tracker reuse fails downstream), run palm detection.
	if !haveROI {
		dets, err := p.palm.Detect(frame)
		if err != nil {
			return nil, fmt.Errorf("palm detect: %w", err)
		}
		// Detections come out in letterboxed-model-space [0, 1]. Map back
		// to source-frame normalized coords so the rest of the pipeline
		// (ROI, landmark, draw) sees correct positions.
		offX, offY, lScale := p.palm.LastLetterbox()
		mw, mh := p.palm.InputSize()
		for i := range dets {
			dets[i] = unletterbox(dets[i], offX, offY, lScale, mw, mh, w, h)
		}
		var topScore float32
		for i := range dets {
			if dets[i].Score > topScore {
				topScore = dets[i].Score
			}
		}
		var best *Detection
		for i := range dets {
			if dets[i].Score < p.MinPalmScore {
				continue
			}
			if best == nil || dets[i].Score > best.Score {
				best = &dets[i]
			}
		}
		if best != nil {
			p.debugf("palm: %d dets, top=%.3f kept; best @ (%.2f,%.2f) size=%.2fx%.2f kp0=(%.2f,%.2f) kp2=(%.2f,%.2f)",
				len(dets), topScore,
				best.XCenter, best.YCenter, best.Width, best.Height,
				best.Keypoints[0].X, best.Keypoints[0].Y,
				best.Keypoints[2].X, best.Keypoints[2].Y)
		} else {
			p.debugf("palm: %d dets, top=%.3f, thresh=%.3f, none kept", len(dets), topScore, p.MinPalmScore)
		}
		if best == nil {
			p.hasPrevROI = false
			p.prevROI = nil
			return nil, nil
		}
		roi = DetectionToROI(*best, PalmToHandROIOptions())
		haveROI = true
		p.debugf("roi: center=(%.2f,%.2f) size=%.2fx%.2f rot=%.0fdeg",
			roi.CenterX, roi.CenterY, roi.Width, roi.Height, roi.Rotation*57.2958)
	} else {
		p.debugf("palm: SKIPPED (reusing tracker ROI)")
	}

	// 3) Run hand landmark on the ROI.
	hand, err := p.landmark.Run(frame, roi, w, h, p.MinLandmarkScore)
	if err != nil {
		return nil, fmt.Errorf("hand landmark: %w", err)
	}
	if hand == nil {
		p.debugf("landmark: presence below threshold %.3f", p.MinLandmarkScore)
		p.hasPrevROI = false
		p.prevROI = nil
		return nil, nil
	}
	p.debugf("landmark: presence=%.3f handedness=%s", hand.Score, hand.Handedness)

	// Plausibility check: bbox area as a fraction of frame area. Catches the
	// common failure mode where the landmark model hallucinates a "hand" on a
	// face or arm — the bbox tends to be either very small or very large.
	if p.MinBBoxFrac > 0 || p.MaxBBoxFrac > 0 {
		frac := float32(hand.BBox.W*hand.BBox.H) / float32(w*h)
		if (p.MinBBoxFrac > 0 && frac < p.MinBBoxFrac) || (p.MaxBBoxFrac > 0 && frac > p.MaxBBoxFrac) {
			p.debugf("landmark: bbox frac %.3f outside [%.3f, %.3f] — rejecting", frac, p.MinBBoxFrac, p.MaxBBoxFrac)
			p.hasPrevROI = false
			p.prevROI = nil
			return nil, nil
		}
	}

	// 4) Update tracker state — keep the ROI as long as the landmark model
	// stays above MinLandmarkScore. We previously required a +0.2 margin to
	// avoid a death-spiral, but with correct input normalization that never
	// fires, and the extra margin just made the tracker flap (palm
	// re-detection every few frames, visible as ROI jitter).
	next := roiFromLandmarks(hand, w, h)
	p.prevROI = &next
	p.hasPrevROI = true

	return []Hand{*hand}, nil
}

// unletterbox maps a Detection from letterboxed-model-space [0, 1] back to
// source-frame normalized coords. offX/offY is the top-left of the resized
// source inside the (mw, mh) model input square; lScale is the uniform
// resize factor from source to model.
func unletterbox(d Detection, offX, offY int, lScale float32, mw, mh, srcW, srcH int) Detection {
	mapX := func(x float32) float32 {
		px := x*float32(mw) - float32(offX)
		return px / lScale / float32(srcW)
	}
	mapY := func(y float32) float32 {
		py := y*float32(mh) - float32(offY)
		return py / lScale / float32(srcH)
	}
	mapW := func(w float32) float32 {
		return w * float32(mw) / lScale / float32(srcW)
	}
	mapH := func(h float32) float32 {
		return h * float32(mh) / lScale / float32(srcH)
	}
	out := Detection{
		Score:   d.Score,
		XCenter: mapX(d.XCenter),
		YCenter: mapY(d.YCenter),
		Width:   mapW(d.Width),
		Height:  mapH(d.Height),
	}
	for i := range d.Keypoints {
		out.Keypoints[i] = Vec2{X: mapX(d.Keypoints[i].X), Y: mapY(d.Keypoints[i].Y)}
	}
	return out
}

// roiFromLandmarks builds the next-frame ROI by synthesizing a palm
// "detection" from the existing landmarks. Mirrors MediaPipe's
// LandmarksToDetectionCalculator + DetectionsToRectsCalculator chain.
func roiFromLandmarks(h *Hand, srcW, srcH int) ROI {
	// Bbox enclosing all 21 landmarks in normalized coords.
	var xMin, yMin, xMax, yMax float32 = 1e9, 1e9, -1e9, -1e9
	for i := 0; i < LandmarkCount; i++ {
		nx := h.Landmarks[i].X / float32(srcW)
		ny := h.Landmarks[i].Y / float32(srcH)
		if nx < xMin {
			xMin = nx
		}
		if ny < yMin {
			yMin = ny
		}
		if nx > xMax {
			xMax = nx
		}
		if ny > yMax {
			yMax = ny
		}
	}
	det := Detection{
		Score:   h.Score,
		XCenter: (xMin + xMax) / 2,
		YCenter: (yMin + yMax) / 2,
		Width:   xMax - xMin,
		Height:  yMax - yMin,
	}
	// Reuse wrist (landmark 0) and middle MCP (landmark 9) for the rotation
	// vector — MediaPipe uses these exact indices in
	// hand_landmark_landmarks_to_roi.pbtxt.
	det.Keypoints[0] = Vec2{X: h.Landmarks[0].X / float32(srcW), Y: h.Landmarks[0].Y / float32(srcH)}
	det.Keypoints[2] = Vec2{X: h.Landmarks[9].X / float32(srcW), Y: h.Landmarks[9].Y / float32(srcH)}

	opts := ROIOptions{
		StartKeypointIdx:   0,
		EndKeypointIdx:     2,
		TargetAngleDegrees: 90,
		ShiftX:             0,
		ShiftY:             -0.1, // landmark-derived ROI shifts less than palm-derived
		ScaleX:             2.0,
		ScaleY:             2.0,
		SquareLong:         true,
	}
	return DetectionToROI(det, opts)
}
