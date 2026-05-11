package main

import "time"

// fpsCalc is a port of utils.CvFpsCalc — sliding-window average of frame
// intervals. We use Go's monotonic clock instead of cv2.getTickCount.
type fpsCalc struct {
	last      time.Time
	intervals []time.Duration
	idx       int
	full      bool
}

func newFPSCalc(bufferLen int) *fpsCalc {
	if bufferLen < 1 {
		bufferLen = 1
	}
	return &fpsCalc{
		last:      time.Now(),
		intervals: make([]time.Duration, bufferLen),
	}
}

func (f *fpsCalc) tick() float64 {
	now := time.Now()
	dt := now.Sub(f.last)
	f.last = now

	f.intervals[f.idx] = dt
	f.idx = (f.idx + 1) % len(f.intervals)
	if f.idx == 0 {
		f.full = true
	}

	n := len(f.intervals)
	if !f.full {
		n = f.idx
		if n == 0 {
			n = 1
		}
	}
	var sum time.Duration
	for i := 0; i < n; i++ {
		sum += f.intervals[i]
	}
	if sum == 0 {
		return 0
	}
	return float64(n) / sum.Seconds()
}
