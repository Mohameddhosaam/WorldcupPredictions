"""
=============================================================
STEP 4: TRAINING THE MODEL
=============================================================

🧠 AI CONCEPT: "How machines learn from data"

   SUPERVISED LEARNING in 4 steps:
   1. Feed the model (X=features, y=labels) for thousands of matches
   2. Model makes a prediction for each match
   3. Compare prediction vs reality → compute "loss" (how wrong it was)
   4. Adjust internal parameters to reduce the loss
   5. Repeat thousands of times → model improves

   WHY XGBOOST?
   XGBoost = eXtreme Gradient Boosting
   - "Boosting": builds many small decision trees one after another,
     each one trying to fix the mistakes of the previous ones
   - Handles missing values, works well with tabular data
   - Gives feature importance: tells you WHICH features matter most
   - Consistent winner on structured data competitions

   WHAT WE'LL COVER:
   - Train/test split (why we can't train and test on the same data)
   - Cross-validation (more robust evaluation)
   - Hyperparameters (the knobs you tune)
   - Evaluation metrics (accuracy, precision, confusion matrix)
   - Feature importance (what the model actually learned)

=============================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix
)
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

DATA_DIR   = "data"
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

LABEL_NAMES = {0: "Home Win", 1: "Draw", 2: "Away Win"}


def load_features() -> tuple[pd.DataFrame, pd.Series]:
    path = os.path.join(DATA_DIR, "features.csv")
    if not os.path.exists(path):
        raise FileNotFoundError("Run 3_feature_engineering.py first!")

    df = pd.read_csv(path)
    X = df.drop(columns=["result"])
    y = df["result"]
    return X, y


def train(X: pd.DataFrame, y: pd.Series):
    print("\n🤖 STEP 4: Training the Model\n")
    print(f"  Training data: {X.shape[0]:,} matches, {X.shape[1]} features")

    # ── Train / Test Split ────────────────────────────────
    # 🧠 CONCEPT: Why split?
    #    If you test on the same data you trained on, you don't know
    #    if the model actually "learned" or just "memorized."
    #    Like studying with the answer key vs a real exam.
    #
    #    We hide 20% of data from training and use it ONLY for testing.
    #    This gives an honest estimate of real-world performance.

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,      # 20% held out for testing
        random_state=42,    # Fixed seed = reproducible results
        stratify=y          # Keep class balance equal in both splits
    )

    print(f"\n  📊 Data split:")
    print(f"     Training:  {len(X_train):,} matches (80%)")
    print(f"     Testing:   {len(X_test):,}  matches (20%)")

    # ── Define the Model ──────────────────────────────────
    # 🧠 CONCEPT: Hyperparameters
    #    These are settings YOU choose before training starts.
    #    The model doesn't learn them — you configure them.
    #    Tuning these is called "hyperparameter optimization."
    #
    #    n_estimators:  How many trees to build (more = slower but often better)
    #    max_depth:     How deep each tree can go (deeper = more complex)
    #    learning_rate: How much each tree corrects the previous one
    #                   (smaller = more careful, needs more trees)
    #    subsample:     % of data used to train each tree (prevents overfitting)

    model = xgb.XGBClassifier(
        n_estimators=300,       # Build 300 trees
        max_depth=4,            # Each tree: max 4 levels deep
        learning_rate=0.05,     # Small steps = more careful learning
        subsample=0.8,          # Use 80% of data per tree
        colsample_bytree=0.8,   # Use 80% of features per tree
        objective="multi:softprob",   # Multi-class classification
        num_class=3,                  # 3 classes: home win / draw / away win
        eval_metric="mlogloss",       # Loss function for multi-class
        random_state=42,
        use_label_encoder=False
    )

    # ── Cross-Validation ──────────────────────────────────
    # 🧠 CONCEPT: Cross-validation
    #    Instead of one train/test split, we split data into 5 "folds."
    #    Train on 4 folds, test on 1. Repeat 5 times with different folds.
    #    Average the 5 scores → more reliable estimate of true performance.
    #
    #    This reduces the "luck" factor of a single split.

    print("\n  🔄 Running 5-fold cross-validation...")
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")
    print(f"     CV Accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    print(f"     (Each fold: {[f'{s:.3f}' for s in cv_scores]})")

    # ── Train Final Model ─────────────────────────────────
    print("\n  🏋️  Training final model on all training data...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )

    # ── Evaluate on Test Set ──────────────────────────────
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n  🎯 Test set accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")

    # 🧠 CONCEPT: Why ~55% accuracy isn't bad for football
    #    Football is inherently unpredictable. Bookmakers (with massive
    #    resources) achieve ~55-58% accuracy on match prediction.
    #    A random guesser on 3 classes would score 33%.
    #    Getting to 50-55% means the model learned real patterns.

    print("\n  📋 Classification Report:")
    print("     (precision = when model says X, how often it's right)")
    print("     (recall    = of all actual X, how many did model catch)")
    print()
    report = classification_report(
        y_test, y_pred,
        target_names=[LABEL_NAMES[i] for i in range(3)]
    )
    for line in report.split("\n"):
        print(f"     {line}")

    # ── Confusion Matrix ──────────────────────────────────
    # 🧠 CONCEPT: Confusion matrix
    #    A grid showing what the model predicted vs reality.
    #    Diagonal = correct predictions.
    #    Off-diagonal = mistakes (and what KIND of mistakes).

    cm = confusion_matrix(y_test, y_pred)
    print("  🔢 Confusion Matrix (rows=actual, cols=predicted):")
    print(f"     {'':15} {'Home Win':>10} {'Draw':>10} {'Away Win':>10}")
    for i, row in enumerate(cm):
        print(f"     {LABEL_NAMES[i]:<15} {row[0]:>10,} {row[1]:>10,} {row[2]:>10,}")

    # ── Feature Importance ────────────────────────────────
    # 🧠 CONCEPT: Feature importance
    #    One of the best things about tree-based models:
    #    they tell you WHICH features they used most.
    #    High importance = model relies on this feature heavily.
    #    Low importance = feature barely helps (consider removing it).

    importances = pd.Series(
        model.feature_importances_,
        index=X.columns
    ).sort_values(ascending=False)

    print("\n  🔍 Feature Importance (what the model learned to use):")
    for feat, importance in importances.items():
        bar = "█" * int(importance * 200)
        print(f"     {feat:<30} {bar} {importance:.4f}")

    # ── Save the Model ────────────────────────────────────
    # 🧠 CONCEPT: Model serialization
    #    After training, we save the model to disk.
    #    This way we don't retrain every time we want predictions.
    #    We use pickle (Python's object serialization format).

    model_path = os.path.join(MODELS_DIR, "xgb_match_predictor.pkl")
    with open(model_path, "wb") as f:
        pickle.dump({"model": model, "features": list(X.columns)}, f)

    print(f"\n  💾 Model saved to {model_path}")

    # ── Save feature importance chart ─────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    importances.plot(kind="barh", ax=ax, color="#2196F3")
    ax.set_title("Feature Importance — What the Model Learned", fontsize=14)
    ax.set_xlabel("Importance Score")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(DATA_DIR, "feature_importance.png"), dpi=120)
    print(f"  📈 Feature importance chart saved to data/feature_importance.png")

    print("\n✅ Model trained! Run: python 5_simulate_tournament.py\n")

    return model


def main():
    X, y = load_features()
    train(X, y)


if __name__ == "__main__":
    main()
