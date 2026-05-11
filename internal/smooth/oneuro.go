// Package smooth implements the One-Euro filter — an adaptive low-pass filter
// that smooths slow motion (reducing jitter) while staying responsive to
// fast motion (preserving snap). MediaPipe uses it to smooth hand and pose
// landmarks; we mirror that here.
//
// Reference: Casiez et al., "1€ Filter: A Simple Speed-based Low-pass Filter
// for Noisy Input in Interactive Systems", CHI 2012.
package smooth

import (
	"math"
	"time"
)

// OneEuro filters a scalar stream over time. Zero value is not usable;
// construct with NewOneEuro.
type OneEuro struct {
	minCutoff float64
	beta      float64
	dCutoff   float64

	prevValue float64
	prevDeriv float64
	prevTime  time.Time
	hasPrev   bool
}

// NewOneEuro returns a OneEuro with the given parameters:
//
//   - minCutoff: filter cutoff frequency at zero velocity (Hz). Lower =
//     smoother but more lag at rest. Hands: ~0.5.
//   - beta: speed coefficient. Higher = filter reacts faster to motion.
//     Hands: ~5.0–80 depending on application.
//   - dCutoff: cutoff for the derivative filter (Hz). Typically 1.0.
func NewOneEuro(minCutoff, beta, dCutoff float64) *OneEuro {
	return &OneEuro{minCutoff: minCutoff, beta: beta, dCutoff: dCutoff}
}

// Filter consumes the next sample at the given time and returns the smoothed
// value.
func (f *OneEuro) Filter(value float64, now time.Time) float64 {
	if !f.hasPrev {
		f.prevValue = value
		f.prevTime = now
		f.hasPrev = true
		return value
	}
	dt := now.Sub(f.prevTime).Seconds()
	if dt <= 0 {
		return f.prevValue
	}
	f.prevTime = now

	deriv := (value - f.prevValue) / dt
	alphaD := smoothingAlpha(f.dCutoff, dt)
	smoothedDeriv := f.prevDeriv + alphaD*(deriv-f.prevDeriv)
	f.prevDeriv = smoothedDeriv

	cutoff := f.minCutoff + f.beta*math.Abs(smoothedDeriv)
	alpha := smoothingAlpha(cutoff, dt)
	filtered := f.prevValue + alpha*(value-f.prevValue)
	f.prevValue = filtered
	return filtered
}

// Reset clears the filter's state. Use this when tracking is lost and
// re-initialized so the next sample isn't averaged with stale state.
func (f *OneEuro) Reset() {
	f.hasPrev = false
	f.prevValue = 0
	f.prevDeriv = 0
}

func smoothingAlpha(cutoff, dt float64) float64 {
	tau := 1.0 / (2.0 * math.Pi * cutoff)
	return 1.0 / (1.0 + tau/dt)
}
