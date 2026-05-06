"""Train/serve parity tests.

Verify that the inference path (`build_candidate_frame`) produces a
DataFrame with exactly the columns the trained pipeline expects, and
that scoring returns the right shape for a representative payload.
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.models.pipeline import STRUCTURED_COLUMNS
from src.models.predict import build_candidate_frame


def _payload():
    return {
        "sender": {"name": "Caira Wong", "email": "caira@x.com"},
        "to": [
            {"name": "Anna Smith", "email": "anna@x.com"},
            {"name": "Brad Jones", "email": "brad@x.com"},
        ],
        "cc": [{"name": "John Patel", "email": "john@x.com"}],
        "subject": "Draft",
        "body": "Hi Anna, can you and Brad send the draft?",
        "task": "can you and Brad send the draft?",
    }


def test_inference_frame_has_required_columns():
    df = build_candidate_frame(_payload())
    required = set(STRUCTURED_COLUMNS) | {"full_context"}
    missing = required - set(df.columns)
    assert not missing, f"missing required columns at inference: {missing}"


def test_inference_frame_one_row_per_candidate():
    df = build_candidate_frame(_payload())
    # sender + 2 To + 1 Cc = 4 candidates.
    assert len(df) == 4
    emails = set(df["candidate_email"])
    assert emails == {"caira@x.com", "anna@x.com", "brad@x.com", "john@x.com"}


def test_inference_frame_role_flags():
    df = build_candidate_frame(_payload())
    by_email = {r["candidate_email"]: r for _, r in df.iterrows()}
    assert by_email["caira@x.com"]["is_sender"] == 1
    assert by_email["anna@x.com"]["is_to"] == 1
    assert by_email["brad@x.com"]["is_to"] == 1
    assert by_email["john@x.com"]["is_cc"] == 1


def test_inference_dedup_when_email_in_multiple_roles():
    payload = _payload()
    payload["cc"].append({"name": "Anna Smith", "email": "ANNA@x.com"})
    df = build_candidate_frame(payload)
    # Anna (To+Cc) collapses to one row.
    anna_rows = df[df["candidate_email"] == "anna@x.com"]
    assert len(anna_rows) == 1
    assert int(anna_rows.iloc[0]["is_to"]) == 1
    assert int(anna_rows.iloc[0]["is_cc"]) == 1
    assert anna_rows.iloc[0]["candidate_role"] == "multiple"


def test_inference_handles_string_addresses():
    payload = {
        "sender": "Caira <caira@x.com>",
        "to": ["alice@x.com"],
        "cc": [],
        "subject": "S",
        "body": "B",
        "task": "T",
    }
    df = build_candidate_frame(payload)
    assert len(df) == 2
    sender = df[df["is_sender"] == 1].iloc[0]
    assert sender["candidate_email"] == "caira@x.com"


def test_inference_full_context_is_seperator_joined():
    df = build_candidate_frame(_payload())
    fc = df.iloc[0]["full_context"]
    assert "[SEP]" in fc
    assert "Draft" in fc  # subject
    assert "send the draft" in fc  # task / body
