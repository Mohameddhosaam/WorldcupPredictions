# ⚽ World Cup Winner Predictor — AI/ML Learning Project

A hands-on project for software developers learning how AI/ML works.
Built around real historical football data, teaching every concept step by step.

---

## 📁 Project Structure

```
worldcup-ai/
├── README.md               ← You are here
├── requirements.txt        ← Python dependencies
│
├── 1_fetch_data.py         ← STEP 1: Pull data from GitHub
├── 2_explore_data.py       ← STEP 2: Understand your data (EDA)
├── 3_feature_engineering.py← STEP 3: Turn raw data into AI-ready features
├── 4_train_model.py        ← STEP 4: Train the ML model
├── 5_simulate_tournament.py← STEP 5: Simulate the 2026 World Cup
│
├── data/                   ← CSVs saved here after step 1
└── models/                 ← Trained model saved here after step 4
```

---

## 🧠 The AI/ML Concepts You'll Learn

| Step | Concept | What it teaches |
|------|---------|-----------------|
| 1 | Data collection | Where AI gets its "knowledge" from |
| 2 | EDA | Why understanding data beats jumping to models |
| 3 | Feature engineering | The most important (and underrated) skill in ML |
| 4 | Classification model | How ML actually "learns" patterns |
| 5 | Monte Carlo simulation | How to handle uncertainty in predictions |

---

## 🚀 How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run each step in order
python 1_fetch_data.py
python 2_explore_data.py
python 3_feature_engineering.py
python 4_train_model.py
python 5_simulate_tournament.py
```

---

## 🔑 Key AI Concepts (Plain English)

### What is a "model"?
A model is just a mathematical function that maps inputs → outputs.
  Input: [team A stats, team B stats]
  Output: probability of team A winning

### What is "training"?
Showing the model thousands of past matches and letting it find
patterns in the data. It adjusts its internal numbers (weights)
until its predictions match the real historical outcomes.

### What is "feature engineering"?
Raw data (e.g. "Brazil beat Germany 2-1") is not useful to a model.
You need to turn it into numbers: win rate, avg goals, recent form, etc.
This step often matters MORE than which model you pick.

### Why XGBoost?
- Works great on tabular (spreadsheet-like) data
- Fast to train, easy to interpret
- Tells you WHICH features matter most (feature importance)
- Used by most Kaggle competition winners on structured data
