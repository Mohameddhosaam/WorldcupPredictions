"""
=============================================================
STEP 1: DATA COLLECTION
=============================================================

🧠 AI CONCEPT: "Data is the fuel of AI"
   A model is only as good as the data it learns from.
   Garbage in = garbage out. Always start here.

We pull from two public GitHub repos:

  1. martj42/international_results
     → 49,000+ international matches since 1872
     → This is our PRIMARY TRAINING DATA
     → The model learns "what makes a team win" from this

  2. jfjelstul/worldcup (squads & tournaments)
     → World Cup specific data: which teams qualified, when
     → Helps us weight World Cup performance more heavily

=============================================================
"""

import requests
import pandas as pd
import os

# Where we'll save the downloaded data
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)


# ── Helper ────────────────────────────────────────────────
def download_csv(url: str, filename: str) -> pd.DataFrame:
    """
    Download a CSV from a URL and save it locally.

    Why save locally?
    - Don't hammer GitHub on every run (be a good API citizen)
    - Lets you work offline after the first run
    - Reproducibility: you know exactly which data you trained on
    """
    filepath = os.path.join(DATA_DIR, filename)

    # Skip download if we already have the file
    if os.path.exists(filepath):
        print(f"  ✓ Already have {filename}, loading from disk...")
        return pd.read_csv(filepath)

    print(f"  ↓ Downloading {filename}...")
    response = requests.get(url, timeout=30)
    response.raise_for_status()  # Crash loudly if something goes wrong

    with open(filepath, "wb") as f:
        f.write(response.content)

    df = pd.read_csv(filepath)
    print(f"    → {len(df):,} rows loaded")
    return df


# ── Data Sources ──────────────────────────────────────────
SOURCES = {
    # 49k+ international results from 1872 onward
    # Columns: date, home_team, away_team, home_score, away_score,
    #          tournament, city, country, neutral
    "matches.csv": (
        "https://raw.githubusercontent.com/martj42/"
        "international_results/master/results.csv"
    ),

    # All World Cup tournaments (year, host, winner, etc.)
    "worldcups.csv": (
        "https://raw.githubusercontent.com/jfjelstul/"
        "worldcup/master/data-csv/tournaments.csv"
    ),
}


# ── Main ──────────────────────────────────────────────────
def main():
    print("\n🔽 STEP 1: Fetching data from GitHub...\n")

    dataframes = {}
    for filename, url in SOURCES.items():
        try:
            df = download_csv(url, filename)
            dataframes[filename] = df
        except Exception as e:
            print(f"  ✗ Failed to download {filename}: {e}")

    # ── Quick sanity checks ───────────────────────────────
    # Always inspect your raw data before doing anything with it.
    # This is called "data validation" — catching problems early.

    if "matches.csv" in dataframes:
        matches = dataframes["matches.csv"]
        print("\n📋 matches.csv preview:")
        print(matches.head(3).to_string())
        print(f"\n  Shape: {matches.shape}")          # rows × columns
        print(f"  Date range: {matches['date'].min()} → {matches['date'].max()}")
        print(f"  Unique teams: {matches['home_team'].nunique()}")
        print(f"  Null values:\n{matches.isnull().sum()}")

    print("\n✅ Data fetched successfully! Run: python 2_explore_data.py\n")


if __name__ == "__main__":
    main()
