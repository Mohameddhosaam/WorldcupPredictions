# 🧠 AI/ML Concepts Cheatsheet
> For software developers learning machine learning through this project

---

## The Big Picture

```
Raw Data → Features → Model Training → Predictions
  (CSV)     (numbers)   (XGBoost)     (probabilities)
```

AI is not magic. It's pattern matching over large datasets using math.

---

## Core Vocabulary

| Term | Plain English | Where you see it |
|------|--------------|-----------------|
| **Model** | A function that maps input → output | `4_train_model.py` |
| **Training** | Showing the model examples so it learns patterns | `model.fit(X, y)` |
| **Features (X)** | The input numbers the model learns from | win_rate, goal_diff... |
| **Labels (y)** | The answer we want the model to predict | 0=home win, 1=draw, 2=away |
| **Hyperparameters** | Settings you choose before training | n_estimators, max_depth |
| **Overfitting** | Model memorizes training data, fails on new data | Bad: train=95%, test=55% |
| **Underfitting** | Model is too simple, misses patterns | Bad: train=50%, test=50% |
| **Accuracy** | % of correct predictions | Not always the right metric |
| **Cross-validation** | Testing across multiple splits for robustness | 5-fold CV |

---

## How XGBoost Works (Simply)

XGBoost builds a **sequence of decision trees**, where each tree tries
to fix the errors of the previous one. This is called **gradient boosting**.

```
Tree 1: Learns basic patterns (win_rate matters)
Tree 2: Focuses on mistakes Tree 1 made
Tree 3: Focuses on mistakes Tree 2 made
...
Tree 300: Final small corrections

Final prediction = weighted vote of all 300 trees
```

Each tree is weak individually ("weak learner"),
but together they form a strong predictor ("ensemble").

---

## Why Feature Engineering Matters Most

Given the same data:
- Bad features + great model  → mediocre results
- Great features + simple model → good results

The model can only learn patterns from numbers you give it.
If you don't include "recent form", it can't learn that momentum matters.

**This is where domain knowledge (knowing football) beats pure math.**

---

## Probabilities vs. Predictions

The model outputs **probabilities**, not certainties:
```
Brazil vs France:
  P(Brazil wins) = 0.52
  P(Draw)        = 0.22
  P(France wins) = 0.26
```

This is more honest than "Brazil will win."
Even the "winner" in Monte Carlo only wins ~25-35% of simulations.

---

## Monte Carlo Simulation

```python
# Instead of solving P(Argentina wins WC) mathematically (hard):
wins = 0
for _ in range(10_000):
    if simulate_tournament() == "Argentina":
        wins += 1
P = wins / 10_000   # Law of large numbers approximates the true P
```

More simulations → more stable probabilities.
This is how financial risk models, physics simulations, and
game AI all handle complex uncertainty.

---

## Evaluation Metrics Explained

**Accuracy** = correct / total
- Simple but misleading if classes are imbalanced

**Precision** (for "home win") = 
  Of all times model said "home win", how often was it right?

**Recall** (for "home win") = 
  Of all actual home wins, how many did the model catch?

**Confusion Matrix** = grid of actual vs predicted outcomes
```
              Predicted
              HW   D   AW
Actual  HW  [500  80  70]   ← 500 correct home win predictions
         D  [120 200  90]
         AW [ 90  60 400]
```
Diagonal = correct. Off-diagonal = errors.

---

## What to Try Next

1. **Add FIFA rankings** as a feature (official team strength)
2. **Add ELO ratings** (chess-style rating for teams)
3. **Try different models**: RandomForest, LightGBM, Neural Network
4. **Tune hyperparameters**: Use `GridSearchCV` or `optuna`
5. **Add more features**: Head-to-head record, squad age, home continent
6. **Calibrate probabilities**: `sklearn.calibration.CalibratedClassifierCV`
7. **Compare to betting odds**: Are your predictions better than bookmakers?
