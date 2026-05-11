package pipeline

import "math"

// ROIOptions matches MediaPipe's DetectionsToRectsCalculator config for
// palm-detection → hand-landmark in hand_landmark_cpu.pbtxt:
//
//	rotation_vector_start_keypoint_index: 0   // wrist
//	rotation_vector_end_keypoint_index:   2   // middle-finger MCP joint
//	rotation_vector_target_angle_degrees: 90
//	shift_x: 0.0
//	shift_y: -0.5
//	scale_x: 2.0
//	scale_y: 2.0
//	square_long: true
type ROIOptions struct {
	StartKeypointIdx    int
	EndKeypointIdx      int
	TargetAngleDegrees  float32
	ShiftX, ShiftY      float32
	ScaleX, ScaleY      float32
	SquareLong          bool
}

// PalmToHandROIOptions returns the exact config used to go from a palm
// detection to the rotated, padded ROI fed to the hand landmark model.
func PalmToHandROIOptions() ROIOptions {
	return ROIOptions{
		StartKeypointIdx:   0,
		EndKeypointIdx:     2,
		TargetAngleDegrees: 90,
		ShiftX:             0.0,
		ShiftY:             -0.5,
		ScaleX:             2.0,
		ScaleY:             2.0,
		SquareLong:         true,
	}
}

// ROI is a rotated, oriented rectangle in normalized [0,1] model-input
// coordinates. To use it against a source frame of WxH pixels, multiply
// CenterX,Width by W and CenterY,Height by H.
type ROI struct {
	CenterX  float32
	CenterY  float32
	Width    float32
	Height   float32
	Rotation float32 // radians, counter-clockwise
}

// DetectionToROI is a port of MediaPipe's DetectionsToRectsCalculator (with
// rotation) followed by RectTransformationCalculator (shift, scale,
// square_long). Both happen between palm detection and hand landmark.
func DetectionToROI(d Detection, opt ROIOptions) ROI {
	// 1) Initial rect = detection bbox.
	cx, cy := d.XCenter, d.YCenter
	w, h := d.Width, d.Height

	// 2) Rotation derived from start→end palm keypoints.
	target := float32(opt.TargetAngleDegrees) * math.Pi / 180.0
	startKP := d.Keypoints[opt.StartKeypointIdx]
	endKP := d.Keypoints[opt.EndKeypointIdx]
	angle := normalizeRadians(target - float32(math.Atan2(float64(-(endKP.Y-startKP.Y)), float64(endKP.X-startKP.X))))

	// 3) Shift in the rotated rect's local frame, then scale, then square.
	if opt.ShiftX != 0 || opt.ShiftY != 0 {
		xShift := w*opt.ShiftX*float32(math.Cos(float64(angle))) - h*opt.ShiftY*float32(math.Sin(float64(angle)))
		yShift := w*opt.ShiftX*float32(math.Sin(float64(angle))) + h*opt.ShiftY*float32(math.Cos(float64(angle)))
		cx += xShift
		cy += yShift
	}

	if opt.SquareLong {
		long := w
		if h > long {
			long = h
		}
		w, h = long, long
	}
	w *= opt.ScaleX
	h *= opt.ScaleY

	return ROI{
		CenterX:  cx,
		CenterY:  cy,
		Width:    w,
		Height:   h,
		Rotation: angle,
	}
}

// normalizeRadians wraps an angle to (-π, π].
func normalizeRadians(a float32) float32 {
	twoPi := float32(2 * math.Pi)
	a = a - twoPi*float32(math.Floor(float64((a+float32(math.Pi))/twoPi)))
	return a
}

// Corners returns the four corners of a ROI in normalized coordinates,
// ordered top-left, top-right, bottom-right, bottom-left in the rotated
// frame. Use this to compute the affine transform that crops the ROI from
// the source frame into the 224×224 hand-landmark input.
func (r ROI) Corners() [4]Vec2 {
	cos := float32(math.Cos(float64(r.Rotation)))
	sin := float32(math.Sin(float64(r.Rotation)))
	hw, hh := r.Width/2, r.Height/2
	// Local-frame corners then rotated and translated to world frame.
	local := [4]Vec2{
		{-hw, -hh}, // top-left
		{+hw, -hh}, // top-right
		{+hw, +hh}, // bottom-right
		{-hw, +hh}, // bottom-left
	}
	var out [4]Vec2
	for i, p := range local {
		out[i] = Vec2{
			X: r.CenterX + p.X*cos - p.Y*sin,
			Y: r.CenterY + p.X*sin + p.Y*cos,
		}
	}
	return out
}
