package pipeline

import (
	"image"

	"gocv.io/x/gocv"
)

// matToFloat copies a uint8 [H,W,3] gocv.Mat into a flat float32 slice with
// per-channel normalization (v + bias) * scale. The channel order is
// preserved — callers are responsible for handing in a Mat that already has
// the order the model expects (do BGR→RGB via gocv.CvtColor first).
//
// For palm_detection_full: bias=-127.5, scale=1/127.5 → [-1, 1].
// For hand_landmark_full:  bias=0,      scale=1/255   → [ 0, 1].
func matToFloat(src gocv.Mat, out []float32, bias, scale float32) {
	rows, cols := src.Rows(), src.Cols()
	buf, _ := src.DataPtrUint8()
	stride := src.Step()
	for y := 0; y < rows; y++ {
		row := buf[y*stride : y*stride+cols*3]
		base := y * cols * 3
		for x := 0; x < cols; x++ {
			c0 := float32(row[x*3+0])
			c1 := float32(row[x*3+1])
			c2 := float32(row[x*3+2])
			out[base+x*3+0] = (c0 + bias) * scale
			out[base+x*3+1] = (c1 + bias) * scale
			out[base+x*3+2] = (c2 + bias) * scale
		}
	}
}

// resizeForModel resizes src into dst at the requested size using bilinear
// interpolation. dst is reused across calls.
func resizeForModel(src gocv.Mat, dst *gocv.Mat, w, h int) {
	gocv.Resize(src, dst, image.Pt(w, h), 0, 0, gocv.InterpolationLinear)
}

// letterboxFit resizes src to fit inside a wxh square, preserving aspect
// ratio, padding with black on the long axis. This matches MediaPipe's
// ImageToTensorCalculator (keep_aspect_ratio=true, border_mode=BORDER_ZERO).
//
// Returns the affine offset (offX, offY) of the resized image's top-left
// inside the output square, and the uniform scale that was applied. Callers
// use these to map back from model-space coordinates to source-frame pixels.
func letterboxFit(src gocv.Mat, dst *gocv.Mat, w, h int) (offX, offY int, scale float32) {
	sw, sh := src.Cols(), src.Rows()
	sx := float32(w) / float32(sw)
	sy := float32(h) / float32(sh)
	scale = sx
	if sy < sx {
		scale = sy
	}
	rw := int(float32(sw) * scale)
	rh := int(float32(sh) * scale)
	offX = (w - rw) / 2
	offY = (h - rh) / 2

	// Allocate dst as a w*h*3 black canvas, then resize src into the
	// correctly-sized inner rect.
	*dst = gocv.NewMatWithSize(h, w, gocv.MatTypeCV8UC3)
	resized := gocv.NewMat()
	defer resized.Close()
	gocv.Resize(src, &resized, image.Pt(rw, rh), 0, 0, gocv.InterpolationLinear)
	roi := dst.Region(image.Rect(offX, offY, offX+rw, offY+rh))
	defer roi.Close()
	resized.CopyTo(&roi)
	return
}

