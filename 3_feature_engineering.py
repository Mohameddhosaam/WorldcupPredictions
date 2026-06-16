"""
=============================================================
STEP 3: FEATURE ENGINEERING
=============================================================

🧠 AI CONCEPT: "Features are what the model actually learns from"
   Raw data looks like:  "Brazil vs Germany, 7-1, 2014-07-08"
   A model can't learn from strings and dates.
   Features are numerical representations the model can work with.

   Example transformation:
     Raw:     "Brazil played 20 games, won 15, drew 3, lost 2"
     Feature: win_rate = 0.75, draw_rate = 0.15, avg_goals = 2.3

   🔑 Rule of thumb: Better features > better model algorithm
      A simple model with great features beats a complex model
      with poor features almost every time.

FEATURES WE'LL ENGINEER:
  Per team, calculated from last N matches before a game:
  - win_rate         : % of games won
  - draw_rate        : % of games drawn
  - avg_goals_scored : average goals scored per game
  - avg_goals_conceded: average goals conceded per game
  - goal_difference  : avg_scored - avg_conceded (overall quality)
  - wc_appearances   : how many World Cups they've been in
  - recent_form      : win rate in last 10 games specifically

=============================================================
"""

import pandas as pd
import numpy as np
import os

DATA_DIR = "data"
LOOKBACK_GAMES = 30   # How many past games to compute stats from
RECENT_GAMES   = 10   # Shorter window for "recent form"


def load_matches() -> pd.DataFrame:
    path = os.path.join(DATA_DIR, "matches.csv")
    if not os.path.exists(path):
        raise FileNotFoundError("Run 1_fetch_data.py first!")
    df = pd.read_csv(path, parse_dates=["date"])
    return df.sort_values("date").reset_index(drop=True)


# ── Core Feature Builder ──────────────────────────────────
def compute_team_stats(
    team: str,
    before_date: pd.Timestamp,
    df: pd.DataFrame,
    n_games: int = LOOKBACK_GAMES
) -> dict:
    """
    For a given team and a cutoff date, compute stats from their
    last `n_games` matches BEFORE that date.

    🧠 CONCEPT: "Look-ahead bias"
       We must NEVER use data from after the match we're predicting.
       If we do, the model "cheats" — it learns patterns that wouldn't
       be available in a real prediction scenario.
       That's why we pass `before_date` and strictly filter.
    """

    # Get all matches where this team played, before the cutoff
    team_matches = df[
        ((df["home_team"] == team) | (df["away_team"] == team)) &
        (df["date"] < before_date)
    ].tail(n_games)  # Take only the most recent N games

    if len(team_matches) == 0:
        # Return neutral/empty stats for teams with no history
        return _empty_stats()

    wins = draws = losses = goals_scored = goals_conceded = 0

    for _, row in team_matches.iterrows():
        if row["home_team"] == team:
            gf, ga = row["home_score"], row["away_score"]
        else:
            gf, ga = row["away_score"], row["home_score"]

        goals_scored   += gf
        goals_conceded += ga

        if gf > ga:
            wins += 1
        elif gf == ga:
            draws += 1
        else:
            losses += 1

    n = len(team_matches)
    return {
        "win_rate":           wins   / n,
        "draw_rate":          draws  / n,
        "loss_rate":          losses / n,
        "avg_goals_scored":   goals_scored   / n,
        "avg_goals_conceded": goals_conceded / n,
        "goal_difference":    (goals_scored - goals_conceded) / n,
        "games_played":       n,
    }


def _empty_stats() -> dict:
    """Default stats when a team has no history."""
    return {
        "win_rate": 0.33, "draw_rate": 0.33, "loss_rate": 0.33,
        "avg_goals_scored": 1.0, "avg_goals_conceded": 1.0,
        "goal_difference": 0.0, "games_played": 0,
    }


# ── Build Training Dataset ────────────────────────────────
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Turn every match in the dataset into a row of features.

    🧠 CONCEPT: Training data format
       For supervised learning, each row = one example.
       The model learns: given these features (X), predict this label (y).
       
       X = [home_win_rate, home_avg_goals, away_win_rate, ...]
       y = 0 (home win), 1 (draw), 2 (away win)
    """

    print("  Building features for each match (this takes ~30s)...")
    print("  (We're computing rolling stats for every game in history)\n")

    rows = []

    # 🧠 CONCEPT: We only use matches from 1990 onward
    #    Modern football is different from 1900s football.
    #    Including too-old data adds noise, not signal.
    recent = df[df["date"] >= "1990-01-01"].copy()

    for idx, (_, match) in enumerate(recent.iterrows()):
        if idx % 2000 == 0:
            print(f"  Processing match {idx:,}/{len(recent):,}...")

        home = match["home_team"]
        away = match["away_team"]
        date = match["date"]

        # Compute stats for BOTH teams before this match
        home_stats = compute_team_stats(home, date, df, LOOKBACK_GAMES)
        away_stats = compute_team_stats(away, date, df, LOOKBACK_GAMES)
        home_form  = compute_team_stats(home, date, df, RECENT_GAMES)
        away_form  = compute_team_stats(away, date, df, RECENT_GAMES)

        # 🧠 CONCEPT: "Difference features"
        #    Instead of just home_win_rate and away_win_rate separately,
        #    we also add the DIFFERENCE. Models often find these
        #    differential features very informative.
        row = {
            # --- Home team stats ---
            "home_win_rate":           home_stats["win_rate"],
            "home_avg_goals_scored":   home_stats["avg_goals_scored"],
            "home_avg_goals_conceded": home_stats["avg_goals_conceded"],
            "home_goal_diff":          home_stats["goal_difference"],
            "home_recent_win_rate":    home_form["win_rate"],

            # --- Away team stats ---
            "away_win_rate":           away_stats["win_rate"],
            "away_avg_goals_scored":   away_stats["avg_goals_scored"],
            "away_avg_goals_conceded": away_stats["avg_goals_conceded"],
            "away_goal_diff":          away_stats["goal_difference"],
            "away_recent_win_rate":    away_form["win_rate"],

            # --- Differential features ---
            "win_rate_diff":   home_stats["win_rate"] - away_stats["win_rate"],
            "goal_diff_diff":  home_stats["goal_difference"] - away_stats["goal_difference"],
            "form_diff":       home_form["win_rate"] - away_form["win_rate"],

            # --- Match context ---
            # 🧠 Neutral ground removes home advantage — important!
            "is_neutral":      int(match.get("neutral", False)),
            "tournament_weight": _tournament_weight(match.get("tournament", "")),

            # --- Label (what we're predicting) ---
            # 0 = home win, 1 = draw, 2 = away win
            "result": _encode_result(match["home_score"], match["away_score"]),
        }
        rows.append(row)

    features_df = pd.DataFrame(rows)
    return features_df


def _encode_result(home_score: int, away_score: int) -> int:
    """Convert match score to a class label."""
    if home_score > away_score:
        return 0  # Home win
    elif home_score == away_score:
        return 1  # Draw
    else:
        return 2  # Away win


def _tournament_weight(tournament: str) -> float:
    """
    🧠 CONCEPT: Domain knowledge as a feature
       Not all matches are equally competitive.
       A World Cup final tells us more about team quality
       than a friendly. We encode this as a numeric weight.
    """
    t = tournament.lower()
    if "fifa world cup" in t:
        return 3.0
    if "uefa" in t or "copa america" in t or "afcon" in t:
        return 2.0
    if "qualifier" in t:
        return 1.5
    if "friendly" in t:
        return 0.5
    return 1.0


# ── Main ──────────────────────────────────────────────────
def main():
    print("\n⚙️  STEP 3: Feature Engineering\n")

    df = load_matches()
    print(f"  Loaded {len(df):,} raw matches")

    features_df = build_features(df)

    # Save the engineered features
    out_path = os.path.join(DATA_DIR, "features.csv")
    features_df.to_csv(out_path, index=False)

    print(f"\n  📋 Features built: {features_df.shape[0]:,} training examples")
    print(f"  📋 Feature columns ({features_df.shape[1]-1} features + 1 label):")
    for col in features_df.columns:
        print(f"     {'[LABEL]' if col == 'result' else '[FEATURE]'} {col}")

    # 🧠 Show class distribution again (on filtered data)
    dist = features_df["result"].value_counts(normalize=True) * 100
    print(f"\n  🎯 Label distribution in training data:")
    labels = {0: "Home win", 1: "Draw", 2: "Away win"}
    for k, v in dist.items():
        print(f"     {labels[k]}: {v:.1f}%")

    print(f"\n  ✅ Saved to {out_path}")
    print("\n✅ Features ready! Run: python 4_train_model.py\n")


if __name__ == "__main__":
    main()
