// Package pipeline implements the MediaPipe Hands inference graph in Go:
// palm detection → ROI extraction → 21-point landmark refinement → frame-to-
// frame tracking. All inference goes through plain TFLite; the rest is
// ordinary linear algebra and image preprocessing.
package pipeline

// Vec2 is a 2D point in image coordinates (pixels) or normalized coordinates,
// depending on context — the field comments make this explicit at each use.
type Vec2 struct {
	X, Y float32
}

// Rect is an axis-aligned bounding box in pixels.
type Rect struct {
	X, Y, W, H int
}

// Handedness mirrors MediaPipe's notion (mirrored relative to the user since
// the source frame is flipped horizontally).
type Handedness uint8

const (
	HandednessUnknown Handedness = iota
	HandednessLeft
	HandednessRight
)

func (h Handedness) String() string {
	switch h {
	case HandednessLeft:
		return "Left"
	case HandednessRight:
		return "Right"
	}
	return "Unknown"
}

// PalmKeypointCount is the number of palm keypoints emitted by the palm
// detector — used to derive a rotated, padded ROI for the landmark model.
// See MediaPipe palm_detection_full.tflite output channels: 4 bbox values
// followed by 7 (x,y) keypoints = 18 channels.
const PalmKeypointCount = 7

// Detection is one raw palm detection (post-NMS), in input-space normalized
// coordinates (i.e. [0,1] relative to the 192x192 model input).
type Detection struct {
	Score     float32
	XCenter   float32
	YCenter   float32
	Width     float32
	Height    float32
	Keypoints [PalmKeypointCount]Vec2
}

// LandmarkCount is fixed by MediaPipe — 21 points per hand.
const LandmarkCount = 21

// Landmark is one of the 21 hand keypoints in pixel coordinates of the source
// frame. Z is depth relative to the wrist (negative = closer to camera).
type Landmark struct {
	X, Y, Z float32
}

// Hand is the per-frame result emitted to the renderer + classifiers.
type Hand struct {
	Score      float32 // landmark model presence score
	Handedness Handedness
	Landmarks  [LandmarkCount]Landmark
	BBox       Rect // axis-aligned bbox derived from landmarks
}
