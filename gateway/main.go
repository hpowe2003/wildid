// WildID gateway: serves the web UI, validates uploads, calls the ML service,
// and stores sightings. Stdlib only.
package main

import (
	"bytes"
	"embed"
	"encoding/json"
	"io"
	"io/fs"
	"log"
	"mime/multipart"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"sync"
	"time"
)

//go:embed web/*
var web embed.FS

type Sighting struct {
	ID           int       `json:"id"`
	Species      string    `json:"species"`
	Lat          float64   `json:"lat"`
	Lon          float64   `json:"lon"`
	IndividualID string    `json:"individual_id"`
	Matched      bool      `json:"matched_existing"`
	Similarity   float64   `json:"similarity"`
	NeedsReview  bool      `json:"needs_review"`
	At           time.Time `json:"at"`
}

var (
	mu        sync.Mutex
	sightings []Sighting
	mlURL     = env("ML_URL", "http://localhost:8000")
	dataFile  = filepath.Join(env("DATA_DIR", "./data"), "sightings.json")
	client    = &http.Client{Timeout: 15 * time.Second}
)

func env(k, d string) string {
	if v := os.Getenv(k); v != "" {
		return v
	}
	return d
}

func writeJSON(w http.ResponseWriter, code int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	json.NewEncoder(w).Encode(v)
}

func addSighting(w http.ResponseWriter, r *http.Request) {
	r.Body = http.MaxBytesReader(w, r.Body, 8<<20)
	if err := r.ParseMultipartForm(8 << 20); err != nil {
		writeJSON(w, 400, map[string]string{"error": "image must be under 8 MB"})
		return
	}
	species := r.FormValue("species")
	lat, e1 := strconv.ParseFloat(r.FormValue("lat"), 64)
	lon, e2 := strconv.ParseFloat(r.FormValue("lon"), 64)
	f, _, err := r.FormFile("file")
	if species == "" || err != nil || e1 != nil || e2 != nil {
		writeJSON(w, 400, map[string]string{"error": "species, lat, lon and an image file are required"})
		return
	}
	defer f.Close()

	var buf bytes.Buffer
	mw := multipart.NewWriter(&buf)
	mw.WriteField("species", species)
	part, _ := mw.CreateFormFile("file", "upload")
	io.Copy(part, f)
	mw.Close()
	resp, err := client.Post(mlURL+"/identify", mw.FormDataContentType(), &buf)
	if err != nil {
		writeJSON(w, 502, map[string]string{"error": "identification service unavailable"})
		return
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		w.WriteHeader(resp.StatusCode)
		io.Copy(w, resp.Body)
		return
	}
	var s Sighting
	json.NewDecoder(resp.Body).Decode(&s)
	s.Species, s.Lat, s.Lon, s.At = species, lat, lon, time.Now().UTC()

	mu.Lock()
	s.ID = len(sightings) + 1
	sightings = append(sightings, s)
	b, _ := json.Marshal(sightings)
	os.MkdirAll(filepath.Dir(dataFile), 0o755)
	os.WriteFile(dataFile, b, 0o644)
	mu.Unlock()
	writeJSON(w, 201, s)
}

func listSightings(w http.ResponseWriter, _ *http.Request) {
	mu.Lock()
	defer mu.Unlock()
	if sightings == nil {
		writeJSON(w, 200, []Sighting{})
		return
	}
	writeJSON(w, 200, sightings)
}

func proxy(path string) http.HandlerFunc {
	return func(w http.ResponseWriter, _ *http.Request) {
		resp, err := client.Get(mlURL + path)
		if err != nil {
			writeJSON(w, 502, map[string]string{"error": "identification service unavailable"})
			return
		}
		defer resp.Body.Close()
		w.Header().Set("Content-Type", "application/json")
		io.Copy(w, resp.Body)
	}
}

func main() {
	if b, err := os.ReadFile(dataFile); err == nil {
		json.Unmarshal(b, &sightings)
	}
	sub, _ := fs.Sub(web, "web")
	mux := http.NewServeMux()
	mux.Handle("/", http.FileServer(http.FS(sub)))
	mux.HandleFunc("POST /api/sightings", addSighting)
	mux.HandleFunc("GET /api/sightings", listSightings)
	mux.HandleFunc("GET /api/model", proxy("/model"))
	mux.HandleFunc("GET /api/monitor", proxy("/monitor"))
	mux.HandleFunc("GET /healthz", func(w http.ResponseWriter, _ *http.Request) { writeJSON(w, 200, map[string]bool{"ok": true}) })
	log.Println("gateway listening on :8080")
	log.Fatal(http.ListenAndServe(":8080", mux))
}
