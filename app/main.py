# app/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
import sqlite3, pickle, pandas as pd, numpy as np

APP_TITLE = "RecoFilm API — KNN-CF (cosine)"
DB_PATH   = Path("data/reco.db")
MODEL_PKL = Path("models/knncf.pkl")

app = FastAPI(title=APP_TITLE)

# ---------- I/O schemas ----------
class PredictIn(BaseModel):
    user_id: int
    top_k: int = 10

class PredictOut(BaseModel):
    user_id: int
    top_k: int
    titles: list[str]

# ---------- bootstrap ----------
@app.on_event("startup")
def _load_model():
    global payload
    if not MODEL_PKL.exists():
        raise RuntimeError(f"Model not found: {MODEL_PKL}")
    payload = pickle.load(open(MODEL_PKL, "rb"))

def _recommend_titles(user_id: int, top_k: int) -> list[str]:
    """Same logic as CLI: KNN-CF with popularity fallback."""
    knn      = payload["knn"]
    uid2row  = payload["uid2row"]
    mid2col  = payload["mid2col"]
    col2mid  = payload["col2mid"]
    pop_list = payload["popularity"]

    if not DB_PATH.exists():
        raise RuntimeError(f"DB not found: {DB_PATH}")

    con = sqlite3.connect(DB_PATH)
    ratings = pd.read_sql(
        "SELECT movieId, rating FROM ml_ratings WHERE userId = ? ORDER BY rating DESC, ROWID DESC",
        con, params=[user_id]
    )
    movies  = pd.read_sql("SELECT movieId, title FROM ml_movies", con)

    liked = ratings[ratings["rating"] >= 3.5]["movieId"].tolist()[:20]

    scores = {}
    if liked:
        for mid in liked:
            if mid not in mid2col:
                continue
            col = mid2col[mid]
            dist, idxs = knn.kneighbors(knn._fit_X[col], n_neighbors=51, return_distance=True)
            for d, i in zip(dist[0], idxs[0]):
                if i == col:
                    continue
                cand = col2mid[i]
                sim = 1.0 - float(d)
                scores[cand] = scores.get(cand, 0.0) + sim

        for m in liked:
            scores.pop(m, None)

        ranked = [m for m,_ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]
    else:
        ranked = []

    # fallback to popularity
    if len(ranked) < top_k:
        ranked += [m for m in pop_list if m not in ranked and m not in liked]

    rec_ids = ranked[:top_k]
    if len(rec_ids) == 0 and len(pop_list) == 0:
        con.close()
        return []

    df = movies[movies["movieId"].isin(rec_ids)].copy()
    order = {m:i for i,m in enumerate(rec_ids)}
    df["ord"] = df["movieId"].map(order)
    df = df.sort_values("ord")
    titles = df["title"].tolist()
    con.close()
    return titles

# ---------- endpoints ----------
@app.get("/health")
def health():
    return {"status": "ok", "model": str(MODEL_PKL), "db": str(DB_PATH)}

@app.post("/predict", response_model=PredictOut)
def predict(body: PredictIn):
    try:
        titles = _recommend_titles(body.user_id, body.top_k)
        return PredictOut(user_id=body.user_id, top_k=body.top_k, titles=titles)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/training")
def training():
    """
    Fire-and-forget local retrain (Phase 1).
    Notice: this spawns a subprocess; after training, you should reload the server to pick up the new model.
    """
    import subprocess
    try:
        subprocess.run(["python", "training.py"], check=True)
        return {"status": "trained"}
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {e}")
