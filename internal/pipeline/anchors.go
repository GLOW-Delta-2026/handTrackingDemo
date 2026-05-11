package pipeline

import "math"

// AnchorOptions captures the subset of MediaPipe's SsdAnchorsCalculatorOptions
// we actually need. Defaults below match palm_detection_full.tflite.
type AnchorOptions struct {
	NumLayers                     int
	MinScale                      float64
	MaxScale                      float64
	InputSizeWidth                int
	InputSizeHeight               int
	AnchorOffsetX                 float64
	AnchorOffsetY                 float64
	Strides                       []int
	AspectRatios                  []float64
	InterpolatedScaleAspectRatio  float64 // 0 to disable
	FixedAnchorSize               bool
	ReduceBoxesInLowestLayer      bool
}

// PalmDetectionAnchorOptions returns the exact config baked into MediaPipe's
// palm_detection_full.tflite (and palm_detection_lite — same anchor layout).
// Total anchors: 2016 — verified by TestAnchorCount.
func PalmDetectionAnchorOptions() AnchorOptions {
	return AnchorOptions{
		NumLayers:                    4,
		MinScale:                     0.1484375,
		MaxScale:                     0.75,
		InputSizeWidth:               192,
		InputSizeHeight:              192,
		AnchorOffsetX:                0.5,
		AnchorOffsetY:                0.5,
		Strides:                      []int{8, 16, 16, 16},
		AspectRatios:                 []float64{1.0},
		InterpolatedScaleAspectRatio: 1.0,
		FixedAnchorSize:              true,
	}
}

// Anchor is one SSD anchor box in normalized model-input coordinates.
type Anchor struct {
	XCenter, YCenter, W, H float32
}

// GenerateAnchors is a faithful port of MediaPipe's SsdAnchorsCalculator
// (mediapipe/calculators/tflite/ssd_anchors_calculator.cc). It emits anchors
// in the exact order the model expects; do not reorder.
func GenerateAnchors(opt AnchorOptions) []Anchor {
	var anchors []Anchor
	layerID := 0
	for layerID < opt.NumLayers {
		var anchorHeights, anchorWidths []float64
		var aspectRatios, scales []float64

		// Group consecutive layers with the same stride — they share a feature
		// map and contribute multiple anchors per cell.
		lastSameStrideLayer := layerID
		for lastSameStrideLayer < opt.NumLayers &&
			opt.Strides[lastSameStrideLayer] == opt.Strides[layerID] {
			scale := calcScale(opt.MinScale, opt.MaxScale, lastSameStrideLayer, len(opt.Strides))

			if lastSameStrideLayer == 0 && opt.ReduceBoxesInLowestLayer {
				aspectRatios = append(aspectRatios, 1.0, 2.0, 0.5)
				scales = append(scales, 0.1, scale, scale)
			} else {
				for _, ar := range opt.AspectRatios {
					aspectRatios = append(aspectRatios, ar)
					scales = append(scales, scale)
				}
				if opt.InterpolatedScaleAspectRatio > 0 {
					var scaleNext float64
					if lastSameStrideLayer == len(opt.Strides)-1 {
						scaleNext = 1.0
					} else {
						scaleNext = calcScale(opt.MinScale, opt.MaxScale, lastSameStrideLayer+1, len(opt.Strides))
					}
					scales = append(scales, math.Sqrt(scale*scaleNext))
					aspectRatios = append(aspectRatios, opt.InterpolatedScaleAspectRatio)
				}
			}
			lastSameStrideLayer++
		}

		for i, ar := range aspectRatios {
			rs := math.Sqrt(ar)
			anchorHeights = append(anchorHeights, scales[i]/rs)
			anchorWidths = append(anchorWidths, scales[i]*rs)
		}

		stride := opt.Strides[layerID]
		fmH := int(math.Ceil(float64(opt.InputSizeHeight) / float64(stride)))
		fmW := int(math.Ceil(float64(opt.InputSizeWidth) / float64(stride)))

		for y := 0; y < fmH; y++ {
			for x := 0; x < fmW; x++ {
				for j := range anchorHeights {
					xCenter := (float64(x) + opt.AnchorOffsetX) / float64(fmW)
					yCenter := (float64(y) + opt.AnchorOffsetY) / float64(fmH)
					var w, h float64
					if opt.FixedAnchorSize {
						w, h = 1.0, 1.0
					} else {
						w, h = anchorWidths[j], anchorHeights[j]
					}
					anchors = append(anchors, Anchor{
						XCenter: float32(xCenter),
						YCenter: float32(yCenter),
						W:       float32(w),
						H:       float32(h),
					})
				}
			}
		}

		layerID = lastSameStrideLayer
	}
	return anchors
}

func calcScale(minScale, maxScale float64, strideIdx, numStrides int) float64 {
	if numStrides == 1 {
		return (minScale + maxScale) / 2.0
	}
	return minScale + (maxScale-minScale)*float64(strideIdx)/float64(numStrides-1)
}
