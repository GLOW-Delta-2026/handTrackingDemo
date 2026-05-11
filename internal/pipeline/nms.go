package pipeline

import "sort"

// WeightedNMS reproduces MediaPipe's NonMaxSuppressionCalculator with
// algorithm = WEIGHTED, overlap_type = INTERSECTION_OVER_UNION.
//
// For each highest-score detection, every overlapping detection (IoU >=
// overlapThresh) is merged into it by averaging coords weighted by score.
// This is what gives MediaPipe's palm detector its stable bbox even when
// dozens of nearby anchors fire.
func WeightedNMS(dets []Detection, overlapThresh float32) []Detection {
	if len(dets) == 0 {
		return nil
	}
	sorted := make([]Detection, len(dets))
	copy(sorted, dets)
	sort.SliceStable(sorted, func(i, j int) bool {
		return sorted[i].Score > sorted[j].Score
	})

	used := make([]bool, len(sorted))
	out := make([]Detection, 0, len(sorted))

	for i := range sorted {
		if used[i] {
			continue
		}
		used[i] = true
		group := []int{i}
		for j := i + 1; j < len(sorted); j++ {
			if used[j] {
				continue
			}
			if iou(sorted[i], sorted[j]) >= overlapThresh {
				used[j] = true
				group = append(group, j)
			}
		}
		out = append(out, mergeWeighted(sorted, group))
	}
	return out
}

func mergeWeighted(dets []Detection, idxs []int) Detection {
	if len(idxs) == 1 {
		return dets[idxs[0]]
	}
	var totalW float32
	for _, k := range idxs {
		totalW += dets[k].Score
	}
	if totalW == 0 {
		return dets[idxs[0]]
	}
	var m Detection
	m.Score = dets[idxs[0]].Score // highest score wins for the merged result
	for _, k := range idxs {
		w := dets[k].Score / totalW
		m.XCenter += dets[k].XCenter * w
		m.YCenter += dets[k].YCenter * w
		m.Width += dets[k].Width * w
		m.Height += dets[k].Height * w
		for p := 0; p < PalmKeypointCount; p++ {
			m.Keypoints[p].X += dets[k].Keypoints[p].X * w
			m.Keypoints[p].Y += dets[k].Keypoints[p].Y * w
		}
	}
	return m
}

func iou(a, b Detection) float32 {
	ax1 := a.XCenter - a.Width/2
	ay1 := a.YCenter - a.Height/2
	ax2 := a.XCenter + a.Width/2
	ay2 := a.YCenter + a.Height/2
	bx1 := b.XCenter - b.Width/2
	by1 := b.YCenter - b.Height/2
	bx2 := b.XCenter + b.Width/2
	by2 := b.YCenter + b.Height/2

	ix1 := max32(ax1, bx1)
	iy1 := max32(ay1, by1)
	ix2 := min32(ax2, bx2)
	iy2 := min32(ay2, by2)
	iw := ix2 - ix1
	ih := iy2 - iy1
	if iw <= 0 || ih <= 0 {
		return 0
	}
	inter := iw * ih
	areaA := (ax2 - ax1) * (ay2 - ay1)
	areaB := (bx2 - bx1) * (by2 - by1)
	union := areaA + areaB - inter
	if union <= 0 {
		return 0
	}
	return inter / union
}

func min32(a, b float32) float32 {
	if a < b {
		return a
	}
	return b
}

func max32(a, b float32) float32 {
	if a > b {
		return a
	}
	return b
}
