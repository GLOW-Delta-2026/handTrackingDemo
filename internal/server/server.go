// Package server exposes the live hand-tracking state over HTTP. The main
// loop calls Broadcast each frame; connected browsers receive the state via
// Server-Sent Events at /stream. A static index page is served at /.
package server

import (
	"context"
	"embed"
	"encoding/json"
	"fmt"
	"io/fs"
	"log"
	"net/http"
	"sync"
	"time"
)

//go:embed web/*
var staticFS embed.FS

// State is the JSON payload sent to browsers each frame. Coordinates are
// normalized to [0, 1] of the source frame, already mirrored so x=0 is the
// left side of the displayed image.
type State struct {
	Present bool       `json:"present"`
	Tip     [2]float32 `json:"tip"`     // index fingertip (landmark 8)
	Palm    [2]float32 `json:"palm"`    // middle-finger MCP (landmark 9) — stable hand anchor
	Sign    string     `json:"sign"`    // "Open" / "Close" / "Pointer"
	Gesture string     `json:"gesture"` // "Stop" / "Clockwise" / "Counter Clockwise" / "Move"
	FPS     float64    `json:"fps"`
	Seq     uint64     `json:"seq"`
}

// Server fans out State updates to all connected SSE clients.
type Server struct {
	addr string

	mu      sync.Mutex
	clients map[chan State]struct{}
	seq     uint64
}

func New(addr string) *Server {
	return &Server{
		addr:    addr,
		clients: make(map[chan State]struct{}),
	}
}

// Broadcast non-blockingly sends state to every connected client. Slow
// clients miss updates rather than backing up the pipeline.
func (s *Server) Broadcast(st State) {
	s.mu.Lock()
	s.seq++
	st.Seq = s.seq
	for ch := range s.clients {
		select {
		case ch <- st:
		default:
			// client buffer full — drop this frame for them
		}
	}
	s.mu.Unlock()
}

func (s *Server) addClient() chan State {
	ch := make(chan State, 4)
	s.mu.Lock()
	s.clients[ch] = struct{}{}
	s.mu.Unlock()
	return ch
}

func (s *Server) removeClient(ch chan State) {
	s.mu.Lock()
	delete(s.clients, ch)
	s.mu.Unlock()
	close(ch)
}

// Start launches the HTTP server in a goroutine and returns immediately. The
// returned shutdown func should be called on exit.
func (s *Server) Start() (shutdown func(context.Context) error, err error) {
	mux := http.NewServeMux()

	sub, err := fs.Sub(staticFS, "web")
	if err != nil {
		return nil, fmt.Errorf("static fs: %w", err)
	}
	mux.Handle("/", http.FileServer(http.FS(sub)))
	mux.HandleFunc("/stream", s.handleStream)

	srv := &http.Server{Addr: s.addr, Handler: mux}
	go func() {
		if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Printf("http server: %v", err)
		}
	}()
	return srv.Shutdown, nil
}

func (s *Server) handleStream(w http.ResponseWriter, r *http.Request) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "streaming unsupported", http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("Access-Control-Allow-Origin", "*")

	ch := s.addClient()
	defer s.removeClient(ch)

	// Heartbeat keeps proxies from closing the connection during idle gaps.
	tick := time.NewTicker(15 * time.Second)
	defer tick.Stop()

	enc := json.NewEncoder(w)
	for {
		select {
		case <-r.Context().Done():
			return
		case <-tick.C:
			fmt.Fprintf(w, ": ping\n\n")
			flusher.Flush()
		case st := <-ch:
			fmt.Fprintf(w, "data: ")
			if err := enc.Encode(st); err != nil {
				return
			}
			fmt.Fprintf(w, "\n")
			flusher.Flush()
		}
	}
}
