"""Group-aware train/val/test splitting.

Random row-level splits would leak information across email/task
boundaries: candidates from the same email-task pair share text, sender,
and recipients, so a row-level split tends to overstate test performance.
We use scikit-learn's :class:`GroupShuffleSplit` to ensure that all rows
sharing an ``email_task_id`` go to exactly one split.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


def group_train_test_split(
    df: pd.DataFrame,
    group_key: str,
    test_size: float,
    val_size: float,
    random_state: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (train_df, val_df, test_df) with disjoint group membership."""
    if group_key not in df.columns:
        raise KeyError(f"group_key {group_key!r} not in dataframe columns")
    groups = df[group_key].astype(str).to_numpy()

    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_val_idx, test_idx = next(splitter.split(df, groups=groups))
    train_val_df = df.iloc[train_val_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    if val_size and val_size > 0:
        rel_val = val_size / max(1.0 - test_size, 1e-9)
        rel_val = float(min(max(rel_val, 0.0), 0.99))
        if rel_val > 0:
            inner_groups = train_val_df[group_key].astype(str).to_numpy()
            inner_splitter = GroupShuffleSplit(
                n_splits=1, test_size=rel_val, random_state=random_state
            )
            train_idx, val_idx = next(inner_splitter.split(train_val_df, groups=inner_groups))
            train_df = train_val_df.iloc[train_idx].reset_index(drop=True)
            val_df = train_val_df.iloc[val_idx].reset_index(drop=True)
            return train_df, val_df, test_df

    return train_val_df, train_val_df.iloc[0:0].copy(), test_df


def assert_no_group_leakage(*frames: pd.DataFrame, group_key: str) -> None:
    seen: set = set()
    for f in frames:
        keys = set(f[group_key].astype(str).unique()) if len(f) else set()
        if seen & keys:
            raise AssertionError(
                f"Group leakage detected on column {group_key}: {len(seen & keys)} shared groups."
            )
        seen |= keys
