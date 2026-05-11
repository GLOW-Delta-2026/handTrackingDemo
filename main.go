package main

import (
	"container/list"
	"context"
	"encoding/csv"
	"flag"
	"fmt"
	"log"
	"os"
	"strconv"
	"time"

	"gocv.io/x/gocv"

	"github.com/glow-delta-2026/handtracking/internal/classifier"
	"github.com/glow-delta-2026/handtracking/internal/pipeline"
	"github.com/glow-delta-2026/handtracking/internal/render"
	"github.com/glow-delta-2026/handtracking/internal/server"
	"github.com/glow-delta-2026/handtracking/internal/smooth"
)

type config struct {
	deviceID               int
	width, height          int
	minDetectionConfidence float64
	minTrackingConfidence  float64
	staticImageMode        bool

	palmModel       string
	landmarkModel   string
	keypointModel   string
	keypointLabels  string
	gestureModel    string
	gestureLabels   string
	keypointDataCSV  string
	gestureDataCSV   string
	debug            bool
	smoothMinCutoff  float64
	smoothBeta       float64
	httpAddr         string
}

func parseFlags() config {
	c := config{}
	flag.IntVar(&c.deviceID, "device", 0, "camera device index")
	flag.IntVar(&c.width, "width", 960, "capture width")
	flag.IntVar(&c.height, "height", 540, "capture height")
	flag.Float64Var(&c.minDetectionConfidence, "min-detection-confidence", 0.85, "palm detection presence threshold")
	flag.Float64Var(&c.minTrackingConfidence, "min-tracking-confidence", 0.7, "hand landmark presence threshold")
	flag.BoolVar(&c.staticImageMode, "static", false, "treat each frame as independent (disables ROI tracking)")

	flag.StringVar(&c.palmModel, "palm-model", "model/mediapipe/palm_detection_full.tflite", "palm detector .tflite path")
	flag.StringVar(&c.landmarkModel, "landmark-model", "model/mediapipe/hand_landmark_full.tflite", "hand landmark .tflite path")
	flag.StringVar(&c.keypointModel, "keypoint-model", "model/keypoint_classifier/keypoint_classifier.tflite", "hand sign MLP")
	flag.StringVar(&c.keypointLabels, "keypoint-labels", "model/keypoint_classifier/keypoint_classifier_label.csv", "hand sign label CSV")
	flag.StringVar(&c.gestureModel, "gesture-model", "model/point_history_classifier/point_history_classifier.tflite", "finger gesture MLP")
	flag.StringVar(&c.gestureLabels, "gesture-labels", "model/point_history_classifier/point_history_classifier_label.csv", "finger gesture label CSV")
	flag.StringVar(&c.keypointDataCSV, "keypoint-data", "model/keypoint_classifier/keypoint.csv", "keypoint training CSV (write target for mode 1)")
	flag.StringVar(&c.gestureDataCSV, "gesture-data", "model/point_history_classifier/point_history.csv", "point-history training CSV (write target for mode 2)")
	flag.BoolVar(&c.debug, "debug", false, "stderr debug logging per frame")
	flag.Float64Var(&c.smoothMinCutoff, "smooth-min-cutoff", 0.5, "OneEuro min cutoff (Hz); lower = smoother at rest. 0 disables smoothing")
	flag.Float64Var(&c.smoothBeta, "smooth-beta", 5.0, "OneEuro speed coefficient; higher = snappier on fast motion")
	flag.StringVar(&c.httpAddr, "http-addr", ":8080", "HTTP address for the live web app; empty to disable")
	flag.Parse()
	return c
}

const historyLength = 16

func main() {
	cfg := parseFlags()
	if err := run(cfg); err != nil {
		log.Fatal(err)
	}
}

func run(cfg config) error {
	// Load models first — failing here is the most informative error mode
	// (missing .tflite file, missing libtensorflowlite_c, etc).
	pipe, err := pipeline.NewPipeline(cfg.palmModel, cfg.landmarkModel)
	if err != nil {
		return fmt.Errorf("pipeline: %w", err)
	}
	defer pipe.Close()
	pipe.MinPalmScore = float32(cfg.minDetectionConfidence)
	pipe.MinLandmarkScore = float32(cfg.minTrackingConfidence)
	if cfg.debug {
		var lastDebug string
		var lastN int
		pipe.Debug = func(s string) {
			// Collapse identical consecutive messages: "msg (xN)".
			if s == lastDebug {
				lastN++
				return
			}
			if lastDebug != "" && lastN > 0 {
				fmt.Fprintf(os.Stderr, "[pipe] %s (x%d)\n", lastDebug, lastN+1)
			} else if lastDebug != "" {
				fmt.Fprintf(os.Stderr, "[pipe] %s\n", lastDebug)
			}
			lastDebug = s
			lastN = 0
		}
	}

	keypointClf, err := classifier.New(cfg.keypointModel, cfg.keypointLabels)
	if err != nil {
		return fmt.Errorf("keypoint classifier: %w", err)
	}
	defer keypointClf.Close()

	gestureClf, err := classifier.New(cfg.gestureModel, cfg.gestureLabels)
	if err != nil {
		return fmt.Errorf("gesture classifier: %w", err)
	}
	defer gestureClf.Close()
	gestureClf.SetScoreThreshold(0.5, 0) // matches PointHistoryClassifier(score_th=0.5)

	cap, err := gocv.OpenVideoCapture(cfg.deviceID)
	if err != nil {
		return fmt.Errorf("open camera %d: %w", cfg.deviceID, err)
	}
	defer cap.Close()
	cap.Set(gocv.VideoCaptureFrameWidth, float64(cfg.width))
	cap.Set(gocv.VideoCaptureFrameHeight, float64(cfg.height))

	window := gocv.NewWindow("Hand Gesture Recognition")
	defer window.Close()

	frame := gocv.NewMat()
	defer frame.Close()
	mirrored := gocv.NewMat()
	defer mirrored.Close()

	fpsCalc := newFPSCalc(10)
	mode := 0
	number := -1

	pointHistory := make([][2]float32, 0, historyLength)
	gestureHistory := list.New() // recent finger-gesture IDs for majority vote

	var landmarkFilter *smooth.LandmarkFilter
	if cfg.smoothMinCutoff > 0 {
		landmarkFilter = smooth.NewLandmarkFilter(cfg.smoothMinCutoff, cfg.smoothBeta, 1.0)
	}

	var srv *server.Server
	if cfg.httpAddr != "" {
		srv = server.New(cfg.httpAddr)
		shutdown, err := srv.Start()
		if err != nil {
			return fmt.Errorf("http server: %w", err)
		}
		defer func() {
			ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
			defer cancel()
			_ = shutdown(ctx)
		}()
		fmt.Fprintf(os.Stderr, "[handtracking] web app: http://localhost%s\n", cfg.httpAddr)
	}

	fmt.Fprintln(os.Stderr, "[handtracking] press ESC to quit; n=normal, k=log keypoints, h=log point history, 0-9=class id")

	for {
		if ok := cap.Read(&frame); !ok || frame.Empty() {
			log.Println("frame read failed; retrying")
			continue
		}

		gocv.Flip(frame, &mirrored, 1)
		fps := fpsCalc.tick()

		hands, err := pipe.Process(mirrored)
		if err != nil {
			log.Printf("pipeline: %v", err)
		}

		w := mirrored.Cols()
		h := mirrored.Rows()

		var (
			handSignText      string
			fingerGestureText string
		)

		if len(hands) > 0 {
			hand := hands[0]
			if landmarkFilter != nil {
				landmarkFilter.Apply(&hand, time.Now())
			}

			// Hand-sign classification (3 classes: Open/Close/Pointer).
			lmList := make([][2]float32, pipeline.LandmarkCount)
			for i, l := range hand.Landmarks {
				lmList[i] = [2]float32{l.X, l.Y}
			}
			feat := classifier.PreProcessLandmarks(lmList)
			signID, _, err := keypointClf.Classify(feat)
			if err != nil {
				log.Printf("keypoint classify: %v", err)
			} else if labels := keypointClf.Labels(); signID >= 0 && signID < len(labels) {
				handSignText = labels[signID]
			}

			// Pointer (class 2) drives the index-fingertip trail.
			if signID == 2 {
				pointHistory = appendHistory(pointHistory, [2]float32{hand.Landmarks[8].X, hand.Landmarks[8].Y}, historyLength)
			} else {
				pointHistory = appendHistory(pointHistory, [2]float32{0, 0}, historyLength)
			}

			// Finger-gesture classification (4 classes) — only when we have a
			// full history buffer.
			fingerGestureID := 0
			feat2 := classifier.PreProcessPointHistory(pointHistory, w, h)
			if len(feat2) == historyLength*2 {
				id, _, err := gestureClf.Classify(feat2)
				if err != nil {
					log.Printf("gesture classify: %v", err)
				} else {
					fingerGestureID = id
				}
			}
			pushGesture(gestureHistory, fingerGestureID, historyLength)
			most := mostCommonGesture(gestureHistory)
			if labels := gestureClf.Labels(); most >= 0 && most < len(labels) {
				fingerGestureText = labels[most]
			}

			// Logging modes — append to training CSVs when mode is set and a
			// class number key was just pressed.
			if mode == 1 && number >= 0 && number <= 9 {
				if err := appendCSV(cfg.keypointDataCSV, number, feat); err != nil {
					log.Printf("write keypoint CSV: %v", err)
				}
			}
			if mode == 2 && number >= 0 && number <= 9 && len(feat2) == historyLength*2 {
				if err := appendCSV(cfg.gestureDataCSV, number, feat2); err != nil {
					log.Printf("write gesture CSV: %v", err)
				}
			}

			render.Hand(&mirrored, hand, handSignText, fingerGestureText)
		} else {
			pointHistory = appendHistory(pointHistory, [2]float32{0, 0}, historyLength)
			if landmarkFilter != nil {
				landmarkFilter.Reset()
			}
		}

		render.PointHistory(&mirrored, pointHistory)
		render.HUD(&mirrored, fps, mode, number)

		if srv != nil {
			st := server.State{FPS: fps, Sign: handSignText, Gesture: fingerGestureText}
			if len(hands) > 0 {
				hand := hands[0]
				st.Present = true
				st.Tip = [2]float32{hand.Landmarks[8].X / float32(w), hand.Landmarks[8].Y / float32(h)}
				st.Palm = [2]float32{hand.Landmarks[9].X / float32(w), hand.Landmarks[9].Y / float32(h)}
			}
			srv.Broadcast(st)
		}

		window.IMShow(mirrored)
		key := window.WaitKey(10)
		if key == 27 { // ESC
			break
		}
		number, mode = selectMode(key, mode, number)
	}
	return nil
}

func selectMode(key, mode, prev int) (number, newMode int) {
	number = -1
	if key >= '0' && key <= '9' {
		number = key - '0'
	}
	switch key {
	case 'n':
		mode = 0
	case 'k':
		mode = 1
	case 'h':
		mode = 2
	}
	_ = prev
	return number, mode
}

func appendHistory(buf [][2]float32, p [2]float32, maxLen int) [][2]float32 {
	buf = append(buf, p)
	if len(buf) > maxLen {
		buf = buf[len(buf)-maxLen:]
	}
	return buf
}

func pushGesture(l *list.List, id, maxLen int) {
	l.PushBack(id)
	for l.Len() > maxLen {
		l.Remove(l.Front())
	}
}

func mostCommonGesture(l *list.List) int {
	counts := map[int]int{}
	for e := l.Front(); e != nil; e = e.Next() {
		counts[e.Value.(int)]++
	}
	bestID := 0
	bestCount := -1
	for id, c := range counts {
		if c > bestCount {
			bestCount = c
			bestID = id
		}
	}
	return bestID
}

func appendCSV(path string, classID int, features []float32) error {
	f, err := os.OpenFile(path, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		return err
	}
	defer f.Close()
	w := csv.NewWriter(f)
	row := make([]string, 0, len(features)+1)
	row = append(row, strconv.Itoa(classID))
	for _, v := range features {
		row = append(row, strconv.FormatFloat(float64(v), 'f', 6, 32))
	}
	if err := w.Write(row); err != nil {
		return err
	}
	w.Flush()
	return w.Error()
}
