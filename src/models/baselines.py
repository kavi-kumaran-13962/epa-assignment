"""Naive and prior-based baselines used as sanity floors.

The W-NUT paper reports two such baselines:

* "Every recipient is responsible" — high recall, low precision.
* "Mean probability" — assign every candidate the global positive rate
  as their score; threshold at 0.5 (so this collapses to predicting all-0
  unless the prior >= 0.5).

We replicate both. They serve two purposes: a sanity check that our
trained model beats trivial decisions, and a calibration anchor for
discussions of precision/recall trade-offs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class EveryRecipientBaseline:
    """Predict 1 for every candidate."""

    def fit(self, X: pd.DataFrame, y: Optional[np.ndarray] = None) -> "EveryRecipientBaseline":
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.ones(len(X), dtype=int)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        ones = np.ones(len(X))
        return np.column_stack([1 - ones, ones])


@dataclass
class ClassPriorBaseline:
    """Predict the global positive rate as the score for every candidate.

    Useful for the "mean probability" baseline reported in Table 5 of the
    paper. With the standard 0.5 threshold the predictions collapse to all
    zeros for the typical class prior on Enron (~0.4); we still report
    metrics at 0.5 to mirror the paper, plus an "argmax" view that fires
    whenever prior >= 0.5.
    """

    prior_: float = 0.0

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "ClassPriorBaseline":
        y = np.asarray(y).astype(int)
        self.prior_ = float(y.mean()) if len(y) else 0.0
        return self

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        return (np.full(len(X), self.prior_) >= threshold).astype(int)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        score = np.full(len(X), self.prior_)
        return np.column_stack([1 - score, score])
