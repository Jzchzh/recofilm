from pathlib import Path
import argparse, pickle, sqlite3, pandas as pd, numpy as np

ROOT = Path(__file__).resolve().parents[2]
DB   = ROOT / "data" / "reco.db"
MODEL= ROOT / "models" / "knncf.pkl"

def recommend_for_user(user_id: int, top_k: int, payload):
    knn      = payload["knn"]
    uid2row  = payload["uid2row"]
    mid2col  = payload["mid2col"]
    col2mid  = payload["col2mid"]
    pop_list = payload["popularity"]

    con = sqlite3.connect(DB)
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

    if len(ranked) < top_k:
        ranked += [m for m in pop_list if m not in ranked and m not in liked]

    rec_ids = ranked[:top_k]
    df = movies[movies["movieId"].isin(rec_ids)].copy()
    order = {m:i for i,m in enumerate(rec_ids)}
    df["ord"] = df["movieId"].map(order)
    df = df.sort_values("ord")
    con.close()
    return df["title"].tolist()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user_id", type=int, required=True)
    ap.add_argument("--top_k", type=int, default=10)
    args = ap.parse_args()

    payload = pickle.load(open(MODEL, "rb"))
    titles = recommend_for_user(args.user_id, args.top_k, payload)
    print({"user_id": args.user_id, "recommendations": titles})

if __name__ == "__main__":
    main()
