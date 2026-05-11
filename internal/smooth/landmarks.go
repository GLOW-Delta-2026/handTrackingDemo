package smooth

import (
	"time"

	"github.com/glow-delta-2026/handtracking/internal/pipeline"
)

// LandmarkFilter applies an independent OneEuro filter to each (x, y, z)
// coordinate of every landmark. 21 landmarks × 3 dims = 63 filters per hand.
//
// Parameters tuned for typical hand tracking at 30 FPS:
//   - minCutoff 0.5 keeps the hand from looking rubbery at rest
//   - beta 5.0 lets snap motion through with minimal lag
//   - dCutoff 1.0 is the standard derivative smoothing
type LandmarkFilter struct {
	filters [pipeline.LandmarkCount * 3]*OneEuro
}

func NewLandmarkFilter(minCutoff, beta, dCutoff float64) *LandmarkFilter {
	lf := &LandmarkFilter{}
	for i := range lf.filters {
		lf.filters[i] = NewOneEuro(minCutoff, beta, dCutoff)
	}
	return lf
}

// Apply filters every coordinate of h.Landmarks in-place. The bounding box
// is recomputed from the smoothed landmarks.
func (lf *LandmarkFilter) Apply(h *pipeline.Hand, now time.Time) {
	if h == nil {
		return
	}
	for i := 0; i < pipeline.LandmarkCount; i++ {
		fx, fy, fz := lf.filters[i*3+0], lf.filters[i*3+1], lf.filters[i*3+2]
		x := float32(fx.Filter(float64(h.Landmarks[i].X), now))
		y := float32(fy.Filter(float64(h.Landmarks[i].Y), now))
		z := float32(fz.Filter(float64(h.Landmarks[i].Z), now))
		h.Landmarks[i] = pipeline.Landmark{X: x, Y: y, Z: z}
	}
	// Recompute bbox from the smoothed coords so the rect on screen lines
	// up with the smoothed skeleton.
	var xMin, yMin, xMax, yMax float32
	xMin, yMin = h.Landmarks[0].X, h.Landmarks[0].Y
	xMax, yMax = xMin, yMin
	for _, lm := range h.Landmarks[1:] {
		if lm.X < xMin {
			xMin = lm.X
		}
		if lm.X > xMax {
			xMax = lm.X
		}
		if lm.Y < yMin {
			yMin = lm.Y
		}
		if lm.Y > yMax {
			yMax = lm.Y
		}
	}
	h.BBox = pipeline.Rect{X: int(xMin), Y: int(yMin), W: int(xMax - xMin), H: int(yMax - yMin)}
}

// Reset clears all filter state. Call when tracking is lost so the next
// hand doesn't blend with the previous one's last position.
func (lf *LandmarkFilter) Reset() {
	for _, f := range lf.filters {
		f.Reset()
	}
}
