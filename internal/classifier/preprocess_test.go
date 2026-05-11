package classifier

import (
	"math"
	"testing"
)

// TestPreProcessLandmarks_KnownInput pins the algorithm against a small
// hand-checkable example. Wrist at (10,10), middle MCP at (10,30):
// relative coords are (0,0), (0,20) and max|v|=20, so normalized output
// is (0,0), (0,1).
func TestPreProcessLandmarks_KnownInput(t *testing.T) {
	in := [][2]float32{
		{10, 10},
		{10, 30},
	}
	got := PreProcessLandmarks(in)
	want := []float32{0, 0, 0, 1}
	if len(got) != len(want) {
		t.Fatalf("len=%d, want %d", len(got), len(want))
	}
	for i := range want {
		if math.Abs(float64(got[i]-want[i])) > 1e-6 {
			t.Fatalf("got[%d]=%v, want %v", i, got[i], want[i])
		}
	}
}

// TestPreProcessLandmarks_AllSamePoint guards the divide-by-zero edge case.
func TestPreProcessLandmarks_AllSamePoint(t *testing.T) {
	in := [][2]float32{{5, 5}, {5, 5}}
	got := PreProcessLandmarks(in)
	for i, v := range got {
		if v != 0 {
			t.Fatalf("expected all zeros, got[%d]=%v", i, v)
		}
	}
}

// TestPreProcessPointHistory normalizes by image dimensions, not by max-abs.
func TestPreProcessPointHistory(t *testing.T) {
	in := [][2]float32{
		{100, 200},
		{200, 400},
	}
	got := PreProcessPointHistory(in, 1000, 1000)
	want := []float32{0, 0, 0.1, 0.2}
	for i := range want {
		if math.Abs(float64(got[i]-want[i])) > 1e-6 {
			t.Fatalf("got[%d]=%v, want %v", i, got[i], want[i])
		}
	}
}
