"""Evaluation metrics shared by train.py and evaluate.py."""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Standard binary classification metrics with safe defaults."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    out: Dict[str, float] = {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "accuracy": float((y_pred == y_true).mean()),
        "support_pos": int(y_true.sum()),
        "support_neg": int((y_true == 0).sum()),
        "n": int(len(y_true)),
    }
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    out["tn"], out["fp"], out["fn"], out["tp"] = (
        int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
    )
    if y_score is not None and len(np.unique(y_true)) > 1:
        try:
            out["pr_auc"] = float(average_precision_score(y_true, y_score))
            out["roc_auc"] = float(roc_auc_score(y_true, y_score))
        except ValueError:
            pass
    return out


def threshold_sweep(
    y_true: np.ndarray, y_score: np.ndarray, thresholds: List[float]
) -> pd.DataFrame:
    rows = []
    for t in thresholds:
        preds = (y_score >= t).astype(int)
        m = compute_metrics(y_true, preds, y_score)
        m["threshold"] = float(t)
        rows.append(m)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Scenario slicing - mirrors the paper's discussion of single- vs.
# multi-recipient cases plus the take-home's ask for additional cuts.
# ---------------------------------------------------------------------------

def scenario_masks(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Build boolean masks for paper- and product-relevant evaluation slices.

    Slices mirror the W-NUT 2018 paper's discussion of single- vs.
    multi-recipient performance, plus addressee-tagging cuts the take-home
    asked us to look at explicitly (implicit "you", explicit name mention,
    no-one cases, sender vs. recipient candidates).
    """
    body = df["body"].fillna("").str.lower()
    task = df["task"].fillna("").str.lower()
    cand_first = df["candidate_name"].fillna("").str.split().str[0].fillna("").str.lower()
    cand_full = df["candidate_name"].fillna("").str.lower()

    # Per-row "task contains my first name (length>1) as a whole word".
    def _row_explicit(t: str, first: str, full: str) -> bool:
        if not t:
            return False
        if first and len(first) > 1 and re.search(rf"\b{re.escape(first)}\b", t):
            return True
        if full and len(full) > 2 and full in t:
            return True
        return False

    explicit_name = np.array(
        [
            _row_explicit(t, f, fu)
            for t, f, fu in zip(task.tolist(), cand_first.tolist(), cand_full.tolist())
        ],
        dtype=bool,
    )

    # The paper splits on "single recipient" vs "multi recipient" emails.
    # Our num_total_candidates includes the sender; "single recipient" thus
    # corresponds to num_total_candidates <= 2 (sender + one ToCc).
    masks: Dict[str, np.ndarray] = {
        "single_recipient_email": (df["num_total_candidates"] <= 2).to_numpy(),
        "multi_recipient_email": (df["num_total_candidates"] > 2).to_numpy(),
        "task_has_you_or_your": (
            task.str.contains(r"\byou\b", regex=True)
            | task.str.contains(r"\byour\b", regex=True)
        ).to_numpy(),
        "task_has_explicit_name": explicit_name,
        "task_has_no_explicit_person": ~explicit_name,
        "no_one_responsible": (df["no_one_responsible"] == 1).to_numpy(),
        "candidate_is_sender": (df["is_sender"] == 1).to_numpy(),
        "candidate_is_recipient": ((df["is_to"] == 1) | (df["is_cc"] == 1)).to_numpy(),
        "perfect_agreement_only": (
            df.get("perfect_agreement", pd.Series([0] * len(df))).fillna(0).astype(int) == 1
        ).to_numpy(),
    }
    return masks


def evaluate_scenarios(
    df: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray
) -> pd.DataFrame:
    rows = []
    masks = scenario_masks(df)
    for name, mask in masks.items():
        if mask.sum() == 0:
            continue
        m = compute_metrics(y_true[mask], y_pred[mask], y_score[mask])
        m["scenario"] = name
        m["n_examples"] = int(mask.sum())
        rows.append(m)
    return pd.DataFrame(rows)
