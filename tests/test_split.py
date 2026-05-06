"""Group-aware train/val/test splitting tests.

The dominant correctness concern in this project is data leakage: rows
from the same email/task HIT must never straddle train and test. These
tests verify that and the assertion that protects against regressions.
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.models.split import assert_no_group_leakage, group_train_test_split


def _toy_df(n_hits: int = 50, candidates_per_hit: int = 3) -> pd.DataFrame:
    rows = []
    for h in range(n_hits):
        for c in range(candidates_per_hit):
            rows.append(
                {
                    "email_task_id": f"H{h:03d}",
                    "candidate_email": f"c{c}_{h}@x.com",
                    "label": int(c == 0),
                }
            )
    return pd.DataFrame(rows)


def test_group_split_no_leakage():
    df = _toy_df(n_hits=80, candidates_per_hit=4)
    train, val, test = group_train_test_split(
        df, group_key="email_task_id", test_size=0.2, val_size=0.1, random_state=0
    )
    train_groups = set(train["email_task_id"])
    val_groups = set(val["email_task_id"])
    test_groups = set(test["email_task_id"])

    assert train_groups.isdisjoint(test_groups)
    assert train_groups.isdisjoint(val_groups)
    assert val_groups.isdisjoint(test_groups)
    # All HITs accounted for.
    assert len(train_groups | val_groups | test_groups) == 80


def test_assert_no_group_leakage_raises_on_overlap():
    df = _toy_df(n_hits=10, candidates_per_hit=2)
    train = df.iloc[:10].copy()
    test = df.iloc[5:].copy()  # overlaps with train
    with pytest.raises(AssertionError):
        assert_no_group_leakage(train, test, group_key="email_task_id")


def test_split_is_deterministic_for_same_seed():
    df = _toy_df(n_hits=30)
    a = group_train_test_split(df, "email_task_id", 0.2, 0.1, random_state=42)
    b = group_train_test_split(df, "email_task_id", 0.2, 0.1, random_state=42)
    for x, y in zip(a, b):
        pd.testing.assert_frame_equal(x.reset_index(drop=True), y.reset_index(drop=True))
