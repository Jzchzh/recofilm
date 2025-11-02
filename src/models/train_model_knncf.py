from pathlib import Path
import sqlite3, pickle
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parents[2]
DB   = ROOT / "data" / "reco.db"
OUT  = ROOT / "models" / "knncf.pkl"

# simple popularity fallback
def compute_popularity(ratings: pd.DataFrame, top=2000):
    return ratings.groupby("movieId")["rating"].count().sort_values(ascending=False).head(top).index.to_list()

def main():
    con = sqlite3.connect(DB)
    ratings = pd.read_sql("SELECT userId, movieId, rating FROM ml_ratings", con)
    con.close()

  
    liked = ratings[ratings["rating"] >= 3.5].copy()

   
    users  = np.sort(liked["userId"].unique())
    movies = np.sort(liked["movieId"].unique())
    uid2row = {u:i for i,u in enumerate(users)}
    mid2col = {m:i for i,m in enumerate(movies)}
    col2mid = {i:m for m,i in mid2col.items()}

  
    rows = liked["userId"].map(uid2row).values
    cols = liked["movieId"].map(mid2col).values
    data = np.ones(len(liked), dtype=np.float32)
    X = csr_matrix((data, (rows, cols)), shape=(len(users), len(movies)))


    knn = NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=51)  
    knn.fit(X.T)

    
    popularity = compute_popularity(ratings, top=2000)

    
    OUT.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "knn": knn,
        "uid2row": uid2row,
        "mid2col": mid2col,
        "col2mid": col2mid,
        "popularity": popularity,
    }
    pickle.dump(payload, open(OUT, "wb"))
    print(f"✅ saved KNN-CF model to {OUT}")

if __name__ == "__main__":
    main()
