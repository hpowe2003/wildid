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
	if s.NeedsReview || s.Matched {
		go triage(s)
	}
	writeJSON(w, 201, s)
}

func listSightings(w http.ResponseWriter, r *http.Request) {
	mu.Lock()
	defer mu.Unlock()
	out := []Sighting{}
	id := r.URL.Query().Get("individual_id")
	for _, s := range sightings {
		if id == "" || s.IndividualID == id {
			out = append(out, s)
		}
	}
	writeJSON(w, 200, out)
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

type Alert struct {
	ID         int       `json:"id"`
	SightingID int       `json:"sighting_id"`
	Severity   string    `json:"severity"`
	Summary    string    `json:"summary"`
	Action     string    `json:"recommended_action"`
	At         time.Time `json:"at"`
}

var (
	alerts      = []Alert{}
	agentURL    = env("AGENT_URL", "http://agent:8000")
	alertFile   = filepath.Join(env("DATA_DIR", "./data"), "alerts.json")
	agentClient = &http.Client{Timeout: 90 * time.Second}
)

// triage asks the LLM agent to investigate a sighting. Failures never block uploads.
func triage(s Sighting) {
	b, _ := json.Marshal(s)
	resp, err := agentClient.Post(agentURL+"/triage", "application/json", bytes.NewReader(b))
	if err != nil {
		log.Println("triage skipped:", err)
		return
	}
	resp.Body.Close()
}

func addAlert(w http.ResponseWriter, r *http.Request) {
	if t := os.Getenv("AGENT_TOKEN"); t != "" && r.Header.Get("X-Agent-Token") != t {
		writeJSON(w, 401, map[string]string{"error": "unauthorized"})
		return
	}
	r.Body = http.MaxBytesReader(w, r.Body, 1<<16)
	var a Alert
	if json.NewDecoder(r.Body).Decode(&a) != nil || a.Summary == "" ||
		(a.Severity != "low" && a.Severity != "medium" && a.Severity != "high") {
		writeJSON(w, 400, map[string]string{"error": "invalid alert"})
		return
	}
	mu.Lock()
	a.ID, a.At = len(alerts)+1, time.Now().UTC()
	alerts = append(alerts, a)
	b, _ := json.Marshal(alerts)
	os.MkdirAll(filepath.Dir(alertFile), 0o755)
	os.WriteFile(alertFile, b, 0o644)
	mu.Unlock()
	writeJSON(w, 201, a)
}

func listAlerts(w http.ResponseWriter, _ *http.Request) {
	mu.Lock()
	defer mu.Unlock()
	writeJSON(w, 200, alerts)
}

func ask(w http.ResponseWriter, r *http.Request) {
	r.Body = http.MaxBytesReader(w, r.Body, 1<<14)
	resp, err := agentClient.Post(agentURL+"/ask", "application/json", r.Body)
	if err != nil {
		writeJSON(w, 502, map[string]string{"error": "agent unavailable"})
		return
	}
	defer resp.Body.Close()
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(resp.StatusCode)
	io.Copy(w, resp.Body)
}

func main() {
	if b, err := os.ReadFile(dataFile); err == nil {
		json.Unmarshal(b, &sightings)
	}
	if b, err := os.ReadFile(alertFile); err == nil {
		json.Unmarshal(b, &alerts)
	}
	sub, _ := fs.Sub(web, "web")
	mux := http.NewServeMux()
	mux.Handle("/", http.FileServer(http.FS(sub)))
	mux.HandleFunc("POST /api/sightings", addSighting)
	mux.HandleFunc("GET /api/sightings", listSightings)
	mux.HandleFunc("POST /api/alerts", addAlert)
	mux.HandleFunc("GET /api/alerts", listAlerts)
	mux.HandleFunc("POST /api/ask", ask)
	mux.HandleFunc("GET /api/model", proxy("/model"))
	mux.HandleFunc("GET /api/monitor", proxy("/monitor"))
	mux.HandleFunc("GET /healthz", func(w http.ResponseWriter, _ *http.Request) { writeJSON(w, 200, map[string]bool{"ok": true}) })
	log.Println("gateway listening on :8080")
	log.Fatal(http.ListenAndServe(":8080", mux))
}
