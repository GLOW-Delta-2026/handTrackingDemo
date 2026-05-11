package classifier

import "math"

// PreProcessLandmarks reproduces app.py's pre_process_landmark:
//
//  1. Translate every (x,y) so the wrist (landmark 0) is at the origin.
//  2. Flatten to a 1-D list of length 42.
//  3. Divide every value by max(|value|) so the result lives in [-1, 1].
//
// landmarks is a slice of [x, y] pixel coords in source-image space.
func PreProcessLandmarks(landmarks [][2]float32) []float32 {
	if len(landmarks) == 0 {
		return nil
	}
	out := make([]float32, 0, len(landmarks)*2)
	baseX, baseY := landmarks[0][0], landmarks[0][1]
	var maxAbs float32
	for _, p := range landmarks {
		dx := p[0] - baseX
		dy := p[1] - baseY
		out = append(out, dx, dy)
		if a := absF(dx); a > maxAbs {
			maxAbs = a
		}
		if a := absF(dy); a > maxAbs {
			maxAbs = a
		}
	}
	if maxAbs > 0 {
		inv := 1.0 / maxAbs
		for i := range out {
			out[i] *= inv
		}
	}
	return out
}

// PreProcessPointHistory reproduces app.py's pre_process_point_history:
//
//  1. Make every (x,y) relative to the first point.
//  2. Divide x by imageWidth and y by imageHeight to normalize.
//  3. Flatten to a 1-D list of length 32 (16 history × 2).
func PreProcessPointHistory(history [][2]float32, imageWidth, imageHeight int) []float32 {
	if len(history) == 0 {
		return nil
	}
	out := make([]float32, 0, len(history)*2)
	baseX, baseY := history[0][0], history[0][1]
	w := float32(imageWidth)
	h := float32(imageHeight)
	for _, p := range history {
		out = append(out, (p[0]-baseX)/w, (p[1]-baseY)/h)
	}
	return out
}

func absF(x float32) float32 {
	return float32(math.Abs(float64(x)))
}
