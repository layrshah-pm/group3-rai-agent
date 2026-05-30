"""
data/train_compas_model.py
--------------------------
Trains a recidivism prediction model on the COMPAS dataset.
Saves models/compas_model.pkl in the same artefact format as loan_model.pkl.

The model intentionally excludes 'race' as a feature, but racial bias
emerges through correlated proxies (priors_count, age) — demonstrating
that removing the sensitive attribute does not guarantee fairness.

Expected output:
  - DI ratio (African-American vs Caucasian) ≈ 0.55–0.65  [below 0.8 threshold]
  - Demographic parity diff ≈ 0.15–0.25                   [above 0.1 threshold]

Run with:
  python data/train_compas_model.py
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).parent.parent
DATA_PATH = ROOT / "data" / "compas_sample.csv"
MODEL_OUT = ROOT / "models" / "compas_model.pkl"


def main():
    print("Loading COMPAS data...")
    df = pd.read_csv(DATA_PATH)

    # Keep only African-American and Caucasian for a clean binary demo
    df = df[df["race"].isin(["African-American", "Caucasian"])].copy()
    df = df.dropna(subset=["two_year_recid", "age", "priors_count"])

    print(f"Records after filtering: {len(df)}")
    print(f"Race distribution:\n{df['race'].value_counts()}")
    print(f"Recidivism rate: {df['two_year_recid'].mean():.3f}")

    # Features — race intentionally excluded to demonstrate proxy bias
    FEATURES = [
        "age",
        "juv_fel_count",
        "juv_misd_count",
        "priors_count",
        "days_b_screening_arrest",
    ]
    # Encode sex if present
    if "sex" in df.columns:
        df["sex_encoded"] = LabelEncoder().fit_transform(df["sex"].fillna("Unknown"))
        FEATURES.append("sex_encoded")

    X = df[FEATURES].fillna(0)
    y = df["two_year_recid"].astype(int)
    sensitive = df["race"]  # "African-American" or "Caucasian"

    X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
        X, y, sensitive, test_size=0.3, random_state=42, stratify=y
    )

    print("\nTraining RandomForestClassifier...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=6,
        random_state=42,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)

    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    print(f"Train accuracy: {train_acc:.3f}  |  Test accuracy: {test_acc:.3f}")

    # Quick fairness check before saving
    y_pred = model.predict(X_test)
    groups = {}
    for pred, g in zip(y_pred, s_test.values):
        groups.setdefault(g, []).append(int(pred))
    rates = {g: sum(v) / len(v) for g, v in groups.items()}
    print(f"\nPrediction rates by race: {rates}")
    min_rate = min(rates.values())
    max_rate = max(rates.values())
    di = min_rate / max_rate if max_rate > 0 else 1.0
    print(f"Disparate Impact ratio (African-American / Caucasian): {di:.3f}")
    if di >= 0.8:
        print("WARNING: DI ratio is above threshold — bias demo may not fire. Check data.")
    else:
        print("✓ DI ratio below 0.8 threshold — bias will be detected in demo.")

    artefact = {
        "model": model,
        "test_X": X_test,
        "test_y": y_test,
        "test_sensitive": s_test.reset_index(drop=True),
    }

    MODEL_OUT.parent.mkdir(exist_ok=True)
    with open(MODEL_OUT, "wb") as f:
        pickle.dump(artefact, f)

    print(f"\n✓ Model saved → {MODEL_OUT}")


if __name__ == "__main__":
    main()
