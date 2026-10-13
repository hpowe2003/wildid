"""Pattern re-identification service: embeds coat/stripe/spot patterns, matches
against a gallery of known individuals, and exposes a model card + drift monitor."""
import io, json, os, threading
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

MODEL_CARD = {
    "name": "gradient-pattern-v1", "version": "1.0.0",
    "type": "Orientation-histogram (HOG-style) pattern embedding, 128-d",
    "match_threshold": 0.92, "review_band": 0.04,
    "intended_use": "Re-identify individuals with unique coats (cheetah spots, zebra/tiger stripes) from cropped photos.",
    "limitations": "Sensitive to pose, lighting and crop. Baseline descriptor: swap in a learned embedding. "
                   "Matches within the review band are flagged for human confirmation.",
}
DATA = os.environ.get("DATA_DIR", "/data")
PATH = os.path.join(DATA, "gallery.json")
lock = threading.Lock()
app = FastAPI(title="WildID ML")
gallery: dict = {}   # id -> {"species": str, "vec": list[float]}
scores: list = []    # best-match similarity per request (for drift monitoring)
reviews: list = []   # 1 if flagged for review

if os.path.exists(PATH):
    gallery = json.load(open(PATH))

def embed(img: Image.Image) -> np.ndarray:
    g = np.asarray(img.convert("L").resize((128, 128)), dtype=np.float32) / 255
    gy, gx = np.gradient(g)
    mag, ang = np.hypot(gx, gy), np.arctan2(gy, gx) % np.pi
    bins = np.minimum((ang / np.pi * 8).astype(int), 7)
    feats = [np.bincount(bins[i*32:(i+1)*32, j*32:(j+1)*32].ravel(),
                         weights=mag[i*32:(i+1)*32, j*32:(j+1)*32].ravel(), minlength=8)
             for i in range(4) for j in range(4)]
    v = np.concatenate(feats)
    return v / (np.linalg.norm(v) + 1e-9)

@app.get("/healthz")
def healthz(): return {"ok": True}

@app.get("/model")
def model(): return {**MODEL_CARD, "gallery_size": len(gallery)}

@app.post("/identify")
async def identify(species: str = Form(...), file: UploadFile = File(...)):
    try:
        img = Image.open(io.BytesIO(await file.read()))
        vec = embed(img)
    except (UnidentifiedImageError, OSError):
        raise HTTPException(400, "Upload a valid JPEG or PNG image")
    with lock:
        best_id, best = None, -1.0
        for iid, rec in gallery.items():
            if rec["species"] != species: continue
            s = float(np.dot(vec, np.array(rec["vec"])))
            if s > best: best_id, best = iid, s
        thr = MODEL_CARD["match_threshold"]
        matched = best_id is not None and best >= thr
        if not matched:
            best_id = f"{species[:3].upper()}-{sum(r['species']==species for r in gallery.values())+1:03d}"
            gallery[best_id] = {"species": species, "vec": vec.tolist()}
            os.makedirs(DATA, exist_ok=True); json.dump(gallery, open(PATH, "w"))
        needs_review = matched and best < thr + MODEL_CARD["review_band"]
        if best >= 0: scores.append(best); reviews.append(int(needs_review))
    return {"individual_id": best_id, "matched_existing": matched,
            "similarity": round(max(best, 0), 4), "needs_review": needs_review,
            "model_version": MODEL_CARD["version"]}

@app.get("/monitor")
def monitor():
    n = len(scores)
    out = {"requests_scored": n, "drift_flag": False, "review_rate": round(sum(reviews)/n, 3) if n else 0}
    if n >= 100:  # compare first 50 (baseline) vs last 50 (recent) similarity scores
        b, r = float(np.mean(scores[:50])), float(np.mean(scores[-50:]))
        out.update(baseline_mean=round(b, 4), recent_mean=round(r, 4), drift_flag=abs(b - r) > 0.1)
    return out
