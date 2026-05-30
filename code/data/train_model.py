"""
data/train_model.py
-------------------
Person 4 — Week 1/2 script.

Downloads German Credit dataset, trains a deliberately biased
LogisticRegression (for demo), and saves model to models/loan_model.pkl.

Intentional bias: female-bad-credit samples are up-weighted 4x so the
model learns to penalise female applicants. This produces a disparate
impact ratio of ~0.32 (well below the 0.8 legal threshold), making
the bias agent violation fire reliably in all three demo scenarios.

Run from repo root:
  python3.11 data/train_model.py
"""

import numpy as np
import pickle
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference

ROOT = Path(__file__).parent.parent
DATA_PATH = ROOT / "data" / "german_credit.csv"
MODEL_PATH = ROOT / "models" / "loan_model.pkl"


def load_or_download() -> pd.DataFrame:
    if DATA_PATH.exists():
        print(f"Loading dataset from {DATA_PATH}")
        return pd.read_csv(DATA_PATH)
    print("Downloading German Credit dataset via sklearn...")
    from sklearn.datasets import fetch_openml
    data = fetch_openml("credit-g", as_frame=True, parser="auto", version=1)
    df = data.frame
    df.to_csv(DATA_PATH, index=False)
    print(f"Saved to {DATA_PATH}")
    return df


def disparate_impact_ratio(y_pred, sensitive_features) -> float:
    groups = {}
    for pred, g in zip(y_pred, sensitive_features):
        groups.setdefault(g, []).append(pred)
    rates = {g: sum(preds) / len(preds) for g, preds in groups.items()}
    min_r, max_r = min(rates.values()), max(rates.values())
    return min_r / max_r if max_r > 0 else 1.0


def main():
    df = load_or_download()

    # Target: bad=1 (denied), good=0 (approved)
    df["target"] = (df["class"] == "bad").astype(int)

    # Protected attribute extracted from personal_status
    df["sex"] = df["personal_status"].apply(
        lambda x: "female" if "female" in str(x) else "male"
    )

    num_cols = [
        "duration", "credit_amount", "installment_commitment",
        "residence_since", "age", "existing_credits", "num_dependents",
    ]
    cat_cols = [
        "checking_status", "credit_history", "purpose", "savings_status",
        "employment", "other_parties", "property_magnitude",
        "other_payment_plans", "housing", "job", "own_telephone",
        "foreign_worker", "sex",
    ]
    feature_cols = num_cols + cat_cols

    X, y, sensitive = df[feature_cols], df["target"], df["sex"]

    X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
        X, y, sensitive, test_size=0.3, random_state=42, stratify=y
    )

    # Up-weight female+bad-credit to produce measurable bias for demo
    sample_weights = np.ones(len(y_train))
    female_bad = (s_train == "female") & (y_train == 1)
    sample_weights[female_bad.values] = 4.0

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
    ])
    model = Pipeline([
        ("prep", preprocessor),
        ("clf", LogisticRegression(max_iter=1000, class_weight=None, random_state=42)),
    ])
    model.fit(X_train, y_train, clf__sample_weight=sample_weights)
    y_pred = model.predict(X_test)

    # ── Metrics ──────────────────────────────────────────────────────────────
    acc = (y_pred == y_test).mean()
    di = disparate_impact_ratio(y_pred, s_test.values)
    dp = demographic_parity_difference(y_test, y_pred, sensitive_features=s_test)
    eo = equalized_odds_difference(y_test, y_pred, sensitive_features=s_test)

    print(f"\n{'─'*50}")
    print("  BASELINE MODEL — RAW BIAS REPORT")
    print(f"{'─'*50}")
    print(f"  Accuracy                 : {acc:.3f}")
    print(f"  Disparate Impact Ratio   : {di:.3f}  (≥0.8 required) → {'VIOLATION' if di < 0.8 else 'OK'}")
    print(f"  Demographic Parity Diff  : {dp:.3f}  (±0.1 allowed)  → {'VIOLATION' if abs(dp) > 0.1 else 'OK'}")
    print(f"  Equalized Odds Diff      : {eo:.3f}  (±0.1 allowed)  → {'VIOLATION' if abs(eo) > 0.1 else 'OK'}")
    print(f"{'─'*50}\n")

    # ── Save ─────────────────────────────────────────────────────────────────
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({
            "model": model,
            "feature_cols": feature_cols,
            "num_cols": num_cols,
            "cat_cols": cat_cols,
            "protected_col": "sex",
            "test_X": X_test,
            "test_y": y_test,
            "test_sensitive": s_test,
        }, f)
    print(f"Model saved → {MODEL_PATH}")


if __name__ == "__main__":
    main()
