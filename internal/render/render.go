// Package render draws hand landmarks, bounding rects, and HUD overlays onto
// frames using gocv. Mirrors app.py's draw_landmarks / draw_bounding_rect /
// draw_info_text / draw_point_history / draw_info functions.
package render

import (
	"fmt"
	"image"
	"image/color"

	"gocv.io/x/gocv"

	"github.com/glow-delta-2026/handtracking/internal/pipeline"
)

var (
	black = color.RGBA{0, 0, 0, 255}
	white = color.RGBA{255, 255, 255, 255}
	green = color.RGBA{152, 251, 152, 255} // PaleGreen, like app.py
)

// landmarkConnections lists the bones MediaPipe draws between the 21
// landmarks. Same edges as app.py's draw_landmarks.
var landmarkConnections = [...][2]int{
	// thumb
	{2, 3}, {3, 4},
	// index
	{5, 6}, {6, 7}, {7, 8},
	// middle
	{9, 10}, {10, 11}, {11, 12},
	// ring
	{13, 14}, {14, 15}, {15, 16},
	// pinky
	{17, 18}, {18, 19}, {19, 20},
	// palm
	{0, 1}, {1, 2}, {2, 5}, {5, 9}, {9, 13}, {13, 17}, {17, 0},
}

// Hand draws all the per-hand annotations: bounding rect, skeleton,
// keypoints, info text.
func Hand(img *gocv.Mat, hand pipeline.Hand, handSign, fingerGesture string) {
	drawBoundingRect(img, hand.BBox)
	drawSkeleton(img, hand.Landmarks)
	drawKeypoints(img, hand.Landmarks)
	drawInfoText(img, hand.BBox, hand.Handedness, handSign, fingerGesture)
}

func drawBoundingRect(img *gocv.Mat, b pipeline.Rect) {
	gocv.Rectangle(img, image.Rect(b.X, b.Y, b.X+b.W, b.Y+b.H), black, 1)
}

func drawSkeleton(img *gocv.Mat, lms [pipeline.LandmarkCount]pipeline.Landmark) {
	for _, c := range landmarkConnections {
		p1 := image.Pt(int(lms[c[0]].X), int(lms[c[0]].Y))
		p2 := image.Pt(int(lms[c[1]].X), int(lms[c[1]].Y))
		// Two-tone: thick black under, thin white over (matches app.py).
		gocv.Line(img, p1, p2, black, 6)
		gocv.Line(img, p1, p2, white, 2)
	}
}

func drawKeypoints(img *gocv.Mat, lms [pipeline.LandmarkCount]pipeline.Landmark) {
	for i, lm := range lms {
		p := image.Pt(int(lm.X), int(lm.Y))
		// Fingertips (indices 4, 8, 12, 16, 20) get a bigger circle.
		r := 5
		switch i {
		case 4, 8, 12, 16, 20:
			r = 8
		}
		gocv.Circle(img, p, r, white, -1)
		gocv.Circle(img, p, r, black, 1)
	}
}

func drawInfoText(img *gocv.Mat, b pipeline.Rect, handedness pipeline.Handedness, handSign, fingerGesture string) {
	// Label bar above the bbox.
	gocv.Rectangle(img, image.Rect(b.X, b.Y-22, b.X+b.W, b.Y), black, -1)
	label := handedness.String()
	if handSign != "" {
		label += ":" + handSign
	}
	gocv.PutText(img, label, image.Pt(b.X+5, b.Y-4),
		gocv.FontHersheySimplex, 0.6, white, 1)

	if fingerGesture != "" {
		txt := "Finger Gesture:" + fingerGesture
		gocv.PutText(img, txt, image.Pt(10, 60),
			gocv.FontHersheySimplex, 1.0, black, 4)
		gocv.PutText(img, txt, image.Pt(10, 60),
			gocv.FontHersheySimplex, 1.0, white, 2)
	}
}

// PointHistory draws the recent index-fingertip trail as growing green dots.
// history is a slice of (x,y) in source-image pixels; zero entries are
// skipped (matches app.py's draw_point_history).
func PointHistory(img *gocv.Mat, history [][2]float32) {
	for i, p := range history {
		if p[0] == 0 && p[1] == 0 {
			continue
		}
		r := 1 + i/2
		gocv.Circle(img, image.Pt(int(p[0]), int(p[1])), r, green, 2)
	}
}

// HUD draws the FPS counter and (when in a logging mode) the mode label and
// pending class number.
func HUD(img *gocv.Mat, fps float64, mode, number int) {
	fpsText := fmt.Sprintf("FPS:%.2f", fps)
	gocv.PutText(img, fpsText, image.Pt(10, 30), gocv.FontHersheySimplex, 1.0, black, 4)
	gocv.PutText(img, fpsText, image.Pt(10, 30), gocv.FontHersheySimplex, 1.0, white, 2)

	if mode == 1 || mode == 2 {
		modeText := "MODE:Logging Key Point"
		if mode == 2 {
			modeText = "MODE:Logging Point History"
		}
		gocv.PutText(img, modeText, image.Pt(10, 90), gocv.FontHersheySimplex, 0.6, white, 1)
		if number >= 0 && number <= 9 {
			gocv.PutText(img, fmt.Sprintf("NUM:%d", number), image.Pt(10, 110),
				gocv.FontHersheySimplex, 0.6, white, 1)
		}
	}
}
