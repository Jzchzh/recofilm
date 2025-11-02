# scripts/load_data.py
from pathlib import Path
import gzip, shutil, sqlite3
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"                  # all your files are here
DB   = DATA / "reco.db"

def gunzip_if_needed(gz_path: Path) -> Path:
    """If path is a .tsv.gz, decompress to .tsv (idempotent) and return the .tsv path."""
    if gz_path.suffixes[-2:] == ['.tsv', '.gz']:
        tsv_path = gz_path.with_suffix("")  # drop the .gz suffix
        if tsv_path.exists():
            print(f"✓ Exists: {tsv_path.name}")
            return tsv_path
        print(f"⊙ Decompress: {gz_path.name} -> {tsv_path.name}")
        with gzip.open(gz_path, "rb") as f_in, open(tsv_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
        return tsv_path
    return gz_path

def to_sql(path: Path, table: str, sep=",", chunksize=500_000):
    """Append CSV/TSV data to a SQLite table using chunked reads (memory friendly)."""
    if not path.exists():
        print(f"⚠️  Skip (missing): {path.name}")
        return
    DB.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB)
    print(f"→ Load {path.name} into table {table}")
    for chunk in pd.read_csv(path, sep=sep, low_memory=False, na_values="\\N", chunksize=chunksize):
        chunk.to_sql(table, con, if_exists="append", index=False)
    con.execute("ANALYZE")
    con.commit()
    con.close()

def main():
    # 1) MovieLens CSVs (as placed directly under data/)
    to_sql(DATA / "ratings.csv", "ml_ratings")
    to_sql(DATA / "movies.csv",  "ml_movies")
    to_sql(DATA / "tags.csv",    "ml_tags")
    # genome-* can be added later if needed

    # 2) IMDb: gunzip .tsv.gz then load (optional)
    for gz_name, table in [
        ("title.basics.tsv.gz",   "imdb_title_basics"),
        ("title.ratings.tsv.gz",  "imdb_title_ratings"),
        # add more if needed:
        # ("title.akas.tsv.gz",     "imdb_title_akas"),
        # ("title.principals.tsv.gz","imdb_title_principals"),
        # ("title.crew.tsv.gz",     "imdb_title_crew"),
        # ("title.episode.tsv.gz",  "imdb_title_episode"),
    ]:
        gz = DATA / gz_name
        if gz.exists():
            tsv = gunzip_if_needed(gz)
            to_sql(tsv, table, sep="\t")

    # 3) Useful indexes
    con = sqlite3.connect(DB)
    con.executescript("""
      CREATE INDEX IF NOT EXISTS idx_ml_user   ON ml_ratings(userId);
      CREATE INDEX IF NOT EXISTS idx_ml_movie  ON ml_ratings(movieId);
      CREATE INDEX IF NOT EXISTS idx_movies_id ON ml_movies(movieId);
    """)
    con.commit(); con.close()
    print(f"✅ Done. SQLite DB at: {DB}")

if __name__ == "__main__":
    main()
