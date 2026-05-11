package pipeline

import (
	"math"
	"testing"
)

// TestDecodeIdentityAnchor: with a single anchor centered at (0.5, 0.5) and
// raw box outputs of all zero, the decoded box must collapse to the anchor
// center with zero width/height. Catches off-by-one in the bbox decode math.
func TestDecodeIdentityAnchor(t *testing.T) {
	anchors := []Anchor{{XCenter: 0.5, YCenter: 0.5, W: 1.0, H: 1.0}}
	rawBoxes := make([]float32, 18)
	// Pre-sigmoid score → large positive → sigmoid ≈ 1.
	rawScores := []float32{10.0}

	opt := PalmDetectionDecodeOptions()
	dets := Decode(anchors, rawBoxes, rawScores, opt)

	if len(dets) != 1 {
		t.Fatalf("got %d detections, want 1", len(dets))
	}
	d := dets[0]
	if math.Abs(float64(d.XCenter-0.5)) > 1e-6 || math.Abs(float64(d.YCenter-0.5)) > 1e-6 {
		t.Fatalf("center = (%v,%v), want (0.5,0.5)", d.XCenter, d.YCenter)
	}
	if d.Width != 0 || d.Height != 0 {
		t.Fatalf("w,h = (%v,%v), want (0,0)", d.Width, d.Height)
	}
	if d.Score < 0.999 {
		t.Fatalf("score = %v, want ~1.0", d.Score)
	}
}

// TestDecodeBelowScoreThreshold: a low score must be filtered out.
func TestDecodeBelowScoreThreshold(t *testing.T) {
	anchors := []Anchor{{XCenter: 0.5, YCenter: 0.5, W: 1.0, H: 1.0}}
	rawBoxes := make([]float32, 18)
	rawScores := []float32{-10.0} // sigmoid ≈ 0

	dets := Decode(anchors, rawBoxes, rawScores, PalmDetectionDecodeOptions())
	if len(dets) != 0 {
		t.Fatalf("got %d detections, want 0", len(dets))
	}
}

// TestDecodePalmXYOrder pins palm_detection_full.tflite's output layout:
// reverse_output_order=true → raw[0..3] = x_center, y_center, w, h (not
// y, x, h, w like older MediaPipe SSD heads). Getting this wrong produces
// detections rotated 90° from the actual hand.
func TestDecodePalmXYOrder(t *testing.T) {
	anchors := []Anchor{{XCenter: 0.5, YCenter: 0.5, W: 1.0, H: 1.0}}
	rawBoxes := make([]float32, 18)
	rawBoxes[0] = 19.2 // x offset: +0.1 (= 19.2/192) on top of anchor.x_center=0.5
	rawBoxes[1] = 0    // y offset: 0
	rawBoxes[2] = 96.0 // w: 0.5
	rawBoxes[3] = 38.4 // h: 0.2
	rawScores := []float32{10.0}

	dets := Decode(anchors, rawBoxes, rawScores, PalmDetectionDecodeOptions())
	if len(dets) != 1 {
		t.Fatalf("got %d detections, want 1", len(dets))
	}
	d := dets[0]
	if math.Abs(float64(d.XCenter-0.6)) > 1e-5 {
		t.Fatalf("XCenter = %v, want 0.6 (proves x-first ordering with reverse_output_order)", d.XCenter)
	}
	if math.Abs(float64(d.YCenter-0.5)) > 1e-5 {
		t.Fatalf("YCenter = %v, want 0.5", d.YCenter)
	}
	if math.Abs(float64(d.Width-0.5)) > 1e-5 {
		t.Fatalf("Width = %v, want 0.5", d.Width)
	}
	if math.Abs(float64(d.Height-0.2)) > 1e-5 {
		t.Fatalf("Height = %v, want 0.2", d.Height)
	}
}
