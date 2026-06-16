"""
=============================================================
STEP 2: EXPLORATORY DATA ANALYSIS (EDA)
=============================================================

🧠 AI CONCEPT: "Know your data before you model it"
   Most beginners skip this and jump straight to training.
   That's a mistake. EDA tells you:
     - Are there outliers that will confuse the model?
     - Is the data balanced? (do we have equal wins/losses?)
     - Which raw variables might become useful features?
     - Are there data quality issues to fix?

   Think of this as a doctor reading an X-ray before surgery.

=============================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

DATA_DIR = "data"


def load_data() -> pd.DataFrame:
    path = os.path.join(DATA_DIR, "matches.csv")
    if not os.path.exists(path):
        raise FileNotFoundError("Run 1_fetch_data.py first!")
    df = pd.read_csv(path, parse_dates=["date"])
    return df


def explore(df: pd.DataFrame):
    print("\n📊 STEP 2: Exploratory Data Analysis\n")
    print("=" * 50)

    # ── 1. Basic shape ────────────────────────────────────
    # Always start here. How big is your dataset?
    print(f"\n1️⃣  Dataset shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"   Columns: {list(df.columns)}")

    # ── 2. Class balance ──────────────────────────────────
    # 🧠 CONCEPT: Class imbalance
    #    In classification, if 90% of your data is "home wins",
    #    the model can just always predict "home win" and be 90% accurate
    #    — without actually learning anything. We need to check this.

    df["result"] = np.where(
        df["home_score"] > df["away_score"], "home_win",
        np.where(df["home_score"] < df["away_score"], "away_win", "draw")
    )

    result_counts = df["result"].value_counts(normalize=True) * 100
    print(f"\n2️⃣  Match outcomes (are they balanced?):")
    for outcome, pct in result_counts.items():
        bar = "█" * int(pct / 2)
        print(f"   {outcome:<12} {bar} {pct:.1f}%")

    # ── 3. Goals distribution ─────────────────────────────
    # 🧠 CONCEPT: Understanding the distribution of your target variable
    #    Knowing the typical score range helps us spot outliers
    #    and choose the right model type.

    avg_home = df["home_score"].mean()
    avg_away = df["away_score"].mean()
    print(f"\n3️⃣  Average goals per game:")
    print(f"   Home team: {avg_home:.2f} goals")
    print(f"   Away team: {avg_away:.2f} goals")
    print(f"   Home advantage: +{avg_home - avg_away:.2f} goals")

    # ── 4. Data over time ─────────────────────────────────
    # 🧠 CONCEPT: Temporal patterns
    #    Football has changed a lot since 1872. A model trained on
    #    all data equally might weight 1900s matches the same as 2024 ones.
    #    We'll handle this in feature engineering with "recent form" windows.

    df["year"] = df["date"].dt.year
    decade_counts = df.groupby(df["year"] // 10 * 10).size()
    print(f"\n4️⃣  Matches per decade (data coverage):")
    for decade, count in decade_counts.items():
        bar = "█" * (count // 200)
        print(f"   {decade}s  {bar} ({count:,})")

    # ── 5. Tournament types ───────────────────────────────
    # Not all matches are equal: a World Cup game ≠ a friendly
    # We want the model to learn from competitive games more.

    print(f"\n5️⃣  Top tournament types in dataset:")
    top_tournaments = df["tournament"].value_counts().head(10)
    for t, count in top_tournaments.items():
        print(f"   {t:<40} {count:>5,}")

    # ── 6. Most successful teams ──────────────────────────
    print(f"\n6️⃣  Teams with most wins (all time):")
    wins = df[df["result"] == "home_win"]["home_team"].value_counts()
    wins += df[df["result"] == "away_win"]["away_team"].value_counts()
    wins = wins.fillna(0).sort_values(ascending=False).head(10)
    for team, w in wins.items():
        print(f"   {team:<25} {int(w):>5,} wins")

    # ── 7. Save a visual ──────────────────────────────────
    # 🧠 CONCEPT: Visualizing distributions
    #    A histogram of goal counts shows us the "shape" of scoring.
    #    Most ML features should ideally follow a roughly normal distribution.

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("EDA: Understanding the Raw Data", fontsize=14, fontweight="bold")

    # Goals histogram
    axes[0].hist(df["home_score"], bins=range(0, 15), alpha=0.7,
                 label="Home", color="#2196F3")
    axes[0].hist(df["away_score"], bins=range(0, 15), alpha=0.7,
                 label="Away", color="#FF5722")
    axes[0].set_title("Goals Scored Distribution")
    axes[0].set_xlabel("Goals in a match")
    axes[0].set_ylabel("Number of matches")
    axes[0].legend()

    # Outcome pie chart
    axes[1].pie(
        result_counts.values,
        labels=result_counts.index,
        autopct="%1.1f%%",
        colors=["#4CAF50", "#FF5722", "#9E9E9E"]
    )
    axes[1].set_title("Match Outcome Balance")

    plt.tight_layout()
    plt.savefig(os.path.join(DATA_DIR, "eda_plots.png"), dpi=120)
    print(f"\n   📈 EDA plots saved to data/eda_plots.png")

    print("\n✅ EDA complete! Run: python 3_feature_engineering.py\n")


def main():
    df = load_data()
    explore(df)


if __name__ == "__main__":
    main()
