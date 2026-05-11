package pipeline

import (
	"math"
	"testing"
)

// TestROIRotationVertical: when the wrist-to-middle-finger vector points
// straight up in image coords (i.e. middle finger above wrist), the rotation
// must be zero — fingers already align with the target_angle (90°).
//
// Image coords have +Y pointing down, so "middle finger above wrist" means
// endKP.y < startKP.y.
func TestROIRotationVertical(t *testing.T) {
	d := Detection{XCenter: 0.5, YCenter: 0.5, Width: 0.2, Height: 0.2}
	d.Keypoints[0] = Vec2{X: 0.5, Y: 0.6} // wrist (lower in image)
	d.Keypoints[2] = Vec2{X: 0.5, Y: 0.4} // middle MCP (higher in image)

	r := DetectionToROI(d, PalmToHandROIOptions())
	if math.Abs(float64(r.Rotation)) > 1e-5 {
		t.Fatalf("rotation = %v, want ~0", r.Rotation)
	}
}

// TestROISquareLong: when scale_x = scale_y and square_long is true, width
// equals height after squaring. Verifies the squaring step happens before
// the scale, matching MediaPipe's RectTransformationCalculator order.
func TestROISquareLong(t *testing.T) {
	d := Detection{XCenter: 0.5, YCenter: 0.5, Width: 0.1, Height: 0.3}
	d.Keypoints[0] = Vec2{X: 0.5, Y: 0.6}
	d.Keypoints[2] = Vec2{X: 0.5, Y: 0.4}

	r := DetectionToROI(d, PalmToHandROIOptions())
	if math.Abs(float64(r.Width-r.Height)) > 1e-5 {
		t.Fatalf("w=%v h=%v, want equal (square_long)", r.Width, r.Height)
	}
	// long edge = 0.3, scaled by 2.0
	if math.Abs(float64(r.Width-0.6)) > 1e-5 {
		t.Fatalf("Width = %v, want 0.6 (= long*scale = 0.3*2.0)", r.Width)
	}
}

// TestROIRotation90: when the wrist→middle vector is horizontal (pointing
// right), the rotation must be +90° (= π/2) so the hand ends up pointing up
// in the rotated frame.
func TestROIRotation90(t *testing.T) {
	d := Detection{XCenter: 0.5, YCenter: 0.5, Width: 0.2, Height: 0.2}
	d.Keypoints[0] = Vec2{X: 0.4, Y: 0.5} // wrist
	d.Keypoints[2] = Vec2{X: 0.6, Y: 0.5} // middle MCP to the right

	r := DetectionToROI(d, PalmToHandROIOptions())
	if math.Abs(float64(r.Rotation)-math.Pi/2) > 1e-5 {
		t.Fatalf("rotation = %v, want %v (π/2)", r.Rotation, math.Pi/2)
	}
}
