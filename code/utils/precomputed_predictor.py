"""
utils/precomputed_predictor.py
-------------------------------
Minimal sklearn-compatible predictor that replays pre-computed predictions.
Used by loan_model_fair.pkl to store threshold-adjusted fair predictions
without re-running the original model at inference time.
"""

import numpy as np


class PrecomputedPredictor:
    """Wraps a fixed array of predictions behind a sklearn predict() interface."""

    def __init__(self, y_pred):
        self._y_pred = np.array(y_pred)

    def predict(self, X):
        return self._y_pred
