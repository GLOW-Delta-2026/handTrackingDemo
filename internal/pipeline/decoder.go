package pipeline

import "math"

// DecodeOptions captures the palm-detector-specific knobs of MediaPipe's
// TensorsToDetectionsCalculator. Defaults below match
// palm_detection_full_inference.pbtxt.
type DecodeOptions struct {
	NumCoords            int     // 18 for palm detection (4 bbox + 7 keypoints × 2)
	NumKeypoints         int     // 7
	NumValuesPerKeypoint int     // 2
	BoxCoordOffset       int     // 0
	KeypointCoordOffset  int     // 4
	XScale, YScale       float32 // 192, 192
	WScale, HScale       float32 // 192, 192
	MinScoreThresh       float32 // 0.5
	ScoreClippingThresh  float32 // 100.0
	SigmoidScore         bool    // true
	ReverseOutputOrder   bool    // palm_detection: true → channel layout is x,y,w,h (boxes) and x,y (keypoints)
}

// PalmDetectionDecodeOptions returns the exact decoder config baked into
// palm_detection_full.tflite.
func PalmDetectionDecodeOptions() DecodeOptions {
	return DecodeOptions{
		NumCoords:            18,
		NumKeypoints:         PalmKeypointCount,
		NumValuesPerKeypoint: 2,
		BoxCoordOffset:       0,
		KeypointCoordOffset:  4,
		XScale:               192,
		YScale:               192,
		WScale:               192,
		HScale:               192,
		MinScoreThresh:       0.5,
		ScoreClippingThresh:  100.0,
		SigmoidScore:         true,
		ReverseOutputOrder:   true,
	}
}

// Decode turns raw model output into score-thresholded Detection structs in
// normalized [0,1] model-input coordinates. rawBoxes is a flat slice of length
// len(anchors)*opt.NumCoords; rawScores is a flat slice of length len(anchors).
//
// Note the y/x and h/w order in MediaPipe's raw output: the first coord per
// box is y_center, not x_center.
func Decode(anchors []Anchor, rawBoxes, rawScores []float32, opt DecodeOptions) []Detection {
	out := make([]Detection, 0, 32)
	for i, a := range anchors {
		score := rawScores[i]
		if opt.SigmoidScore {
			if score < -opt.ScoreClippingThresh {
				score = -opt.ScoreClippingThresh
			} else if score > opt.ScoreClippingThresh {
				score = opt.ScoreClippingThresh
			}
			score = sigmoid(score)
		}
		if score < opt.MinScoreThresh {
			continue
		}

		base := i * opt.NumCoords
		var xc, yc, w, h float32
		if opt.ReverseOutputOrder {
			xc = rawBoxes[base+opt.BoxCoordOffset+0]
			yc = rawBoxes[base+opt.BoxCoordOffset+1]
			w = rawBoxes[base+opt.BoxCoordOffset+2]
			h = rawBoxes[base+opt.BoxCoordOffset+3]
		} else {
			yc = rawBoxes[base+opt.BoxCoordOffset+0]
			xc = rawBoxes[base+opt.BoxCoordOffset+1]
			h = rawBoxes[base+opt.BoxCoordOffset+2]
			w = rawBoxes[base+opt.BoxCoordOffset+3]
		}

		xc = xc/opt.XScale*a.W + a.XCenter
		yc = yc/opt.YScale*a.H + a.YCenter
		w = w / opt.WScale * a.W
		h = h / opt.HScale * a.H

		d := Detection{
			Score:   score,
			XCenter: xc, YCenter: yc, Width: w, Height: h,
		}
		for k := 0; k < opt.NumKeypoints && k < PalmKeypointCount; k++ {
			koff := base + opt.KeypointCoordOffset + k*opt.NumValuesPerKeypoint
			var kx, ky float32
			if opt.ReverseOutputOrder {
				kx = rawBoxes[koff+0]
				ky = rawBoxes[koff+1]
			} else {
				ky = rawBoxes[koff+0]
				kx = rawBoxes[koff+1]
			}
			d.Keypoints[k] = Vec2{
				X: kx/opt.XScale*a.W + a.XCenter,
				Y: ky/opt.YScale*a.H + a.YCenter,
			}
		}
		out = append(out, d)
	}
	return out
}

func sigmoid(x float32) float32 {
	return float32(1.0 / (1.0 + math.Exp(float64(-x))))
}
