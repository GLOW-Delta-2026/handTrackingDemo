package pipeline

import (
	"math"
	"testing"
)

// TestNMSCollapsesOverlap: two near-identical detections must merge into one.
func TestNMSCollapsesOverlap(t *testing.T) {
	a := Detection{Score: 0.9, XCenter: 0.5, YCenter: 0.5, Width: 0.2, Height: 0.2}
	b := Detection{Score: 0.8, XCenter: 0.51, YCenter: 0.51, Width: 0.2, Height: 0.2}

	got := WeightedNMS([]Detection{a, b}, 0.3)
	if len(got) != 1 {
		t.Fatalf("got %d detections, want 1", len(got))
	}
	// Score should be the top detection's, not the average.
	if math.Abs(float64(got[0].Score-0.9)) > 1e-6 {
		t.Fatalf("merged score = %v, want 0.9", got[0].Score)
	}
	// Center should be weighted average ≈ (0.5*0.9 + 0.51*0.8) / (0.9+0.8) ≈ 0.5047
	want := float32((0.5*0.9 + 0.51*0.8) / (0.9 + 0.8))
	if math.Abs(float64(got[0].XCenter-want)) > 1e-5 {
		t.Fatalf("merged x_center = %v, want %v", got[0].XCenter, want)
	}
}

// TestNMSKeepsDisjoint: two non-overlapping detections survive.
func TestNMSKeepsDisjoint(t *testing.T) {
	a := Detection{Score: 0.9, XCenter: 0.2, YCenter: 0.2, Width: 0.1, Height: 0.1}
	b := Detection{Score: 0.8, XCenter: 0.8, YCenter: 0.8, Width: 0.1, Height: 0.1}

	got := WeightedNMS([]Detection{a, b}, 0.3)
	if len(got) != 2 {
		t.Fatalf("got %d detections, want 2", len(got))
	}
}

// TestNMSEmpty: the empty-input edge case must not panic.
func TestNMSEmpty(t *testing.T) {
	got := WeightedNMS(nil, 0.3)
	if got != nil {
		t.Fatalf("nil input should yield nil output, got %v", got)
	}
}
