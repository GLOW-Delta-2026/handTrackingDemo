package pipeline

import "testing"

// TestPalmAnchorCount pins the MediaPipe palm detector anchor count to 2016.
// If this changes, the SSD math drifted from the upstream reference and the
// model's output tensor will index into the wrong rows.
func TestPalmAnchorCount(t *testing.T) {
	anchors := GenerateAnchors(PalmDetectionAnchorOptions())
	if got, want := len(anchors), 2016; got != want {
		t.Fatalf("anchor count = %d, want %d", got, want)
	}
}

// TestPalmAnchorLayerBreakdown verifies the per-layer distribution:
//
//	stride 8  → 24×24 feature map, 2 anchors/cell (ar=1.0 + interpolated) = 1152
//	strides 16,16,16 grouped → 12×12 feature map, 6 anchors/cell = 864
//	total = 2016
func TestPalmAnchorLayerBreakdown(t *testing.T) {
	anchors := GenerateAnchors(PalmDetectionAnchorOptions())

	const layer0End = 1152
	if got := len(anchors); got < layer0End+1 {
		t.Fatalf("not enough anchors (%d) to inspect layer split", got)
	}

	// First 1152 should live on the stride-8 feature map: 24 distinct
	// y-centers, each at (i+0.5)/24.
	seenY := map[float32]int{}
	for i := 0; i < layer0End; i++ {
		seenY[anchors[i].YCenter]++
	}
	if got, want := len(seenY), 24; got != want {
		t.Fatalf("stride-8 distinct y-centers = %d, want %d", got, want)
	}

	// Anchors after layer0End are on the stride-16 map: 12 distinct y-centers.
	seenY = map[float32]int{}
	for i := layer0End; i < len(anchors); i++ {
		seenY[anchors[i].YCenter]++
	}
	if got, want := len(seenY), 12; got != want {
		t.Fatalf("stride-16 distinct y-centers = %d, want %d", got, want)
	}
}

// TestPalmAnchorFixedSize verifies all anchors have unit width/height — the
// palm detector uses fixed_anchor_size, so box decoding reads w,h directly
// from the model output rather than scaling by the anchor.
func TestPalmAnchorFixedSize(t *testing.T) {
	anchors := GenerateAnchors(PalmDetectionAnchorOptions())
	for i, a := range anchors {
		if a.W != 1.0 || a.H != 1.0 {
			t.Fatalf("anchor[%d] w,h = %v,%v, want 1.0,1.0", i, a.W, a.H)
		}
	}
}
