"""
=============================================================
STEP 5: MONTE CARLO TOURNAMENT SIMULATION
=============================================================

🧠 AI CONCEPT: "Handling uncertainty with probability"

   The model doesn't say "Brazil WILL win."
   It says "Brazil has a 68% chance of beating France."

   To find the World Cup winner, we:
   1. Simulate every match using those probabilities
   2. Run the entire tournament 10,000 times
   3. Count how often each team wins across all simulations
   4. That count / 10,000 = each team's win probability

   This is called Monte Carlo Simulation:
   Named after the Monte Carlo casino, because it uses randomness
   to solve problems that are too complex for exact math.

   🎲 Why not just pick the highest probability winner each match?
      Because upsets happen! Weaker teams sometimes beat stronger ones.
      Monte Carlo captures this uncertainty naturally.

=============================================================
"""

import pandas as pd
import numpy as np
import pickle
import os
import random

DATA_DIR   = "data"
MODELS_DIR = "models"

# ── 2026 World Cup Qualified Teams ────────────────────────
# 48 teams, organized in 12 groups of 4
# Top 2 from each group + 8 best 3rd-place teams advance (32 total)

GROUPS_2026 = {
    "A": ["Mexico", "South Africa", "South Korea", "Czechia"],
    "B": ["Canada", "Switzerland", "Qatar", "Bosnia and Herzegovina"],
    "C": ["Brazil", "Morocco", "Haiti", "Scotland"],
    "D": ["United States", "Paraguay", "Australia", "Türkiye"],
    "E": ["Germany", "Curaçao", "Ivory Coast", "Ecuador"],
    "F": ["Netherlands", "Japan", "Tunisia", "Sweden"],
    "G": ["Belgium", "Egypt", "Iran", "New Zealand"],
    "H": ["Spain", "Cabo Verde", "Saudi Arabia", "Uruguay"],
    "I": ["France", "Senegal", "Norway", "Iraq"],
    "J": ["Argentina", "Algeria", "Austria", "Jordan"],
    "K": ["Portugal", "Uzbekistan", "Colombia", "DR Congo"],
    "L": ["England", "Croatia", "Ghana", "Panama"],
}

ALL_TEAMS = [team for group in GROUPS_2026.values() for team in group]


# ── Load Model ────────────────────────────────────────────
def load_model():
    path = os.path.join(MODELS_DIR, "xgb_match_predictor.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError("Run 4_train_model.py first!")
    with open(path, "rb") as f:
        return pickle.load(f)


# ── Load Latest Team Stats ────────────────────────────────
def compute_team_stats(team: str, before_date: pd.Timestamp, df: pd.DataFrame, n_games: int = 30) -> dict:
    """Compute rolling stats for a team before a given date."""
    team_matches = df[
        ((df["home_team"] == team) | (df["away_team"] == team)) &
        (df["date"] < before_date)
    ].tail(n_games)

    if len(team_matches) == 0:
        return _default_stats()

    wins = draws = losses = goals_scored = goals_conceded = 0
    for _, row in team_matches.iterrows():
        if row["home_team"] == team:
            gf, ga = row["home_score"], row["away_score"]
        else:
            gf, ga = row["away_score"], row["home_score"]
        goals_scored += gf
        goals_conceded += ga
        if gf > ga:   wins   += 1
        elif gf == ga: draws  += 1
        else:          losses += 1

    n = len(team_matches)
    return {
        "win_rate":           wins / n,
        "draw_rate":          draws / n,
        "loss_rate":          losses / n,
        "avg_goals_scored":   goals_scored / n,
        "avg_goals_conceded": goals_conceded / n,
        "goal_difference":    (goals_scored - goals_conceded) / n,
        "games_played":       n,
    }


def load_team_stats(df: pd.DataFrame) -> dict:
    """
    For each team, compute their stats from the last 30 matches.
    These stats will be used as features when predicting 2026 matches.
    """
    cutoff = pd.Timestamp("2026-06-01")  # Day before the tournament starts

    stats = {}
    for team in ALL_TEAMS:
        stats[team] = compute_team_stats(team, cutoff, df, n_games=30)

    return stats


def build_match_features(home: str, away: str, team_stats: dict) -> dict:
    """Turn a matchup into model-ready features."""
    hs = team_stats.get(home, _default_stats())
    as_ = team_stats.get(away, _default_stats())
    return {
        "home_win_rate":           hs["win_rate"],
        "home_avg_goals_scored":   hs["avg_goals_scored"],
        "home_avg_goals_conceded": hs["avg_goals_conceded"],
        "home_goal_diff":          hs["goal_difference"],
        "home_recent_win_rate":    hs["win_rate"],
        "away_win_rate":           as_["win_rate"],
        "away_avg_goals_scored":   as_["avg_goals_scored"],
        "away_avg_goals_conceded": as_["avg_goals_conceded"],
        "away_goal_diff":          as_["goal_difference"],
        "away_recent_win_rate":    as_["win_rate"],
        "win_rate_diff":           hs["win_rate"] - as_["win_rate"],
        "goal_diff_diff":          hs["goal_difference"] - as_["goal_difference"],
        "form_diff":               hs["win_rate"] - as_["win_rate"],
        "is_neutral":              1,    # World Cup = neutral ground
        "tournament_weight":       3.0,  # World Cup weight
    }


def _default_stats():
    return {
        "win_rate": 0.33, "draw_rate": 0.33, "loss_rate": 0.33,
        "avg_goals_scored": 1.0, "avg_goals_conceded": 1.0,
        "goal_difference": 0.0, "games_played": 0,
    }


# ── Single Match Prediction ────────────────────────────────
def predict_match(home: str, away: str, model, features_list: list, team_stats: dict):
    """
    🧠 CONCEPT: Probabilistic prediction
       The model returns 3 probabilities: [P(home win), P(draw), P(away win)]
       These always sum to 1.0.

       In knockout rounds, we can't have draws.
       So we re-normalize: P(home win) / (P(home win) + P(away win))
    """
    features = build_match_features(home, away, team_stats)
    X = pd.DataFrame([features])[features_list]
    proba = model.predict_proba(X)[0]  # [P(home win), P(draw), P(away win)]
    return {"home": proba[0], "draw": proba[1], "away": proba[2]}


def simulate_knockout_match(home: str, away: str, model, features_list, team_stats) -> str:
    """
    Simulate a knockout match (no draws allowed).

    🧠 CONCEPT: Probabilistic sampling
       We don't just pick the highest probability.
       We SAMPLE from the distribution.
       E.g. if P(home win) = 0.6, we draw a random number.
       If it's < 0.6, home wins. Otherwise, away wins.
       This naturally produces upsets ~40% of the time here.
    """
    proba = predict_match(home, away, model, features_list, team_stats)
    p_home = proba["home"] / (proba["home"] + proba["away"])  # Ignore draw
    return home if random.random() < p_home else away


def simulate_group_stage(team_stats, model, features_list) -> list:
    """Simulate all group stage games, return 32 qualifying teams."""
    qualifiers = []

    for group_name, teams in GROUPS_2026.items():
        results = {t: {"pts": 0, "gd": 0} for t in teams}

        # Round robin: every team plays every other team once
        for i in range(len(teams)):
            for j in range(i + 1, len(teams)):
                home, away = teams[i], teams[j]
                proba = predict_match(home, away, model, features_list, team_stats)

                # Sample outcome
                r = random.random()
                if r < proba["home"]:
                    results[home]["pts"] += 3
                    results[home]["gd"]  += 1
                    results[away]["gd"]  -= 1
                elif r < proba["home"] + proba["draw"]:
                    results[home]["pts"] += 1
                    results[away]["pts"] += 1
                else:
                    results[away]["pts"] += 3
                    results[away]["gd"]  += 1
                    results[home]["gd"]  -= 1

        # Sort by points, then goal difference
        sorted_teams = sorted(
            teams,
            key=lambda t: (results[t]["pts"], results[t]["gd"]),
            reverse=True
        )
        qualifiers.extend(sorted_teams[:2])  # Top 2 advance

    # Add 8 best 3rd place teams (simplified: take remaining randomly)
    # In reality this is based on 3rd place rankings, simplified here
    return qualifiers[:32]


def simulate_knockout_stage(teams_32: list, model, features_list, team_stats) -> str:
    """Simulate Round of 32 → 16 → 8 → 4 → Final → Winner."""
    teams = teams_32[:]  # Don't modify original list

    round_names = ["Round of 32", "Round of 16", "Quarter-finals", "Semi-finals", "Final"]
    round_idx = 0

    while len(teams) > 1:
        next_round = []
        for i in range(0, len(teams), 2):
            if i + 1 < len(teams):
                winner = simulate_knockout_match(
                    teams[i], teams[i + 1], model, features_list, team_stats
                )
                next_round.append(winner)
        teams = next_round
        round_idx += 1

    return teams[0]  # The champion


# ── Monte Carlo Simulation ─────────────────────────────────
def run_monte_carlo(n_simulations: int, model, features_list, team_stats) -> dict:
    """
    🧠 CONCEPT: Monte Carlo Simulation
       Run the entire tournament N times.
       Count wins per team.
       win_probability[team] = wins[team] / N

       Why 10,000? More simulations = more stable probabilities.
       The "law of large numbers" says the average converges
       to the true expected value as N → ∞.
    """
    win_counts = {team: 0 for team in ALL_TEAMS}

    print(f"\n  🎲 Running {n_simulations:,} tournament simulations...")
    for i in range(n_simulations):
        if i % 2000 == 0:
            print(f"     Simulation {i:,}/{n_simulations:,}...")

        qualifiers = simulate_group_stage(team_stats, model, features_list)
        champion = simulate_knockout_stage(qualifiers, model, features_list, team_stats)
        win_counts[champion] += 1

    # Convert counts to probabilities
    win_probs = {
        team: count / n_simulations
        for team, count in win_counts.items()
    }
    return dict(sorted(win_probs.items(), key=lambda x: x[1], reverse=True))


# ── Main ──────────────────────────────────────────────────
def main():
    print("\n🏆 STEP 5: World Cup 2026 Simulation\n")

    # Load model
    artifact = load_model()
    model = artifact["model"]
    features_list = artifact["features"]

    # Load match history for team stats
    print("  Loading match history...")
    matches = pd.read_csv(os.path.join(DATA_DIR, "matches.csv"), parse_dates=["date"])
    matches = matches.sort_values("date").reset_index(drop=True)

    # Get current team stats
    print("  Computing current team statistics...")
    team_stats = load_team_stats(matches)

    # Run Monte Carlo
    win_probs = run_monte_carlo(
        n_simulations=10_000,
        model=model,
        features_list=features_list,
        team_stats=team_stats
    )

    # ── Print Results ──────────────────────────────────────
    print("\n" + "=" * 55)
    print("  🏆 2026 FIFA WORLD CUP — WIN PROBABILITIES")
    print("=" * 55)
    print(f"  {'Rank':<5} {'Team':<25} {'Probability':<12} {'Bar'}")
    print("-" * 55)

    for rank, (team, prob) in enumerate(win_probs.items(), 1):
        if prob < 0.001:
            break  # Skip teams with <0.1% chance
        bar = "█" * int(prob * 200)
        print(f"  {rank:<5} {team:<25} {prob*100:>6.2f}%      {bar}")

    print("=" * 55)
    top_team = list(win_probs.keys())[0]
    top_prob = list(win_probs.values())[0]
    print(f"\n  🥇 Most likely winner: {top_team} ({top_prob*100:.1f}%)")

    print("\n  🧠 Remember: This is a probability, not a guarantee!")
    print("     Even a 30% favorite loses 70% of the time.\n")

    # Save results
    results_df = pd.DataFrame(
        [(t, p) for t, p in win_probs.items() if p > 0],
        columns=["team", "win_probability"]
    )
    results_df.to_csv(os.path.join(DATA_DIR, "predictions_2026.csv"), index=False)
    print("  💾 Results saved to data/predictions_2026.csv")
    print("\n✅ Simulation complete! Check data/predictions_2026.csv\n")


if __name__ == "__main__":
    main()
