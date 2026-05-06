"""Structured feature tests.

These cover the recently-fixed `is_only_recipient` bug, name-match
behaviour, role flags, and the new first-person commitment features.
"""
from __future__ import annotations

import pandas as pd

from src.features.candidate_features import (
    CandidateFeatureExtractor,
    _compute_row_features,
)


def _row(**overrides) -> pd.Series:
    base = {
        "candidate_email": "anna@x.com",
        "candidate_name": "Anna Smith",
        "sender_email": "caira@x.com",
        "sender_name": "Caira Wong",
        "candidate_role": "to",
        "is_sender": 0,
        "is_to": 1,
        "is_cc": 0,
        "num_to_recipients": 1,
        "num_cc_recipients": 0,
        "num_total_candidates": 2,  # sender + 1 To
        "subject": "",
        "body": "",
        "task": "",
    }
    base.update(overrides)
    return pd.Series(base)


# ---------------------------------------------------------------------------
# is_only_recipient bug regression tests
# ---------------------------------------------------------------------------


def test_only_recipient_sender_plus_one_recipient():
    """Sender + 1 To: the To candidate is the only recipient."""
    feats = _compute_row_features(_row(is_to=1, num_total_candidates=2))
    assert feats["is_only_recipient"] == 1.0


def test_only_recipient_sender_is_not_marked_only_recipient():
    """For the sender row in the same HIT, is_only_recipient must be 0."""
    feats = _compute_row_features(
        _row(is_sender=1, is_to=0, num_total_candidates=2,
             candidate_email="caira@x.com", candidate_name="Caira Wong")
    )
    assert feats["is_only_recipient"] == 0.0


def test_only_recipient_zero_when_multiple_recipients():
    """Sender + 2 recipients → no candidate is the lone recipient."""
    feats = _compute_row_features(_row(num_total_candidates=3, num_to_recipients=2))
    assert feats["is_only_recipient"] == 0.0


def test_only_recipient_no_sender_edge_case():
    """No sender, single candidate: is_only_recipient should fire."""
    feats = _compute_row_features(_row(is_sender=0, num_total_candidates=1))
    assert feats["is_only_recipient"] == 1.0


# ---------------------------------------------------------------------------
# Name match
# ---------------------------------------------------------------------------


def test_first_name_in_task_matches_whole_word():
    feats = _compute_row_features(
        _row(candidate_name="Anna Smith", task="Hi Anna, please send the report.")
    )
    assert feats["first_name_in_task"] == 1.0
    assert feats["full_name_in_task"] == 0.0  # only first name appears


def test_first_name_does_not_match_substring():
    """First name 'Sam' should NOT match within 'samples'."""
    feats = _compute_row_features(
        _row(candidate_name="Sam Diaz", task="Please send the samples.")
    )
    assert feats["first_name_in_task"] == 0.0


def test_local_part_too_short_does_not_match():
    """A 2-character local part shouldn't fire spuriously."""
    feats = _compute_row_features(
        _row(candidate_email="bj@x.com", task="Please bj this approach.")
    )
    assert feats["local_part_in_task"] == 0.0


# ---------------------------------------------------------------------------
# Role flags
# ---------------------------------------------------------------------------


def test_role_flags_basic():
    feats = _compute_row_features(_row(is_sender=0, is_to=1, is_cc=0))
    assert feats["is_sender"] == 0.0
    assert feats["is_to"] == 1.0
    assert feats["is_cc"] == 0.0
    assert feats["appears_in_multiple_roles"] == 0.0


def test_appears_in_multiple_roles():
    feats = _compute_row_features(_row(is_sender=0, is_to=1, is_cc=1))
    assert feats["appears_in_multiple_roles"] == 1.0


# ---------------------------------------------------------------------------
# First-person commitment features
# ---------------------------------------------------------------------------


def test_first_person_future_fires_on_ill():
    feats = _compute_row_features(_row(task="I'll handle this for you."))
    assert feats["task_first_person_future"] == 1.0


def test_let_me_fires():
    feats = _compute_row_features(_row(task="Let me look into this."))
    assert feats["task_let_me"] == 1.0


def test_first_person_and_sender_only_fires_for_sender():
    """The cross-feature should only fire when this candidate is the sender."""
    sender_feats = _compute_row_features(
        _row(is_sender=1, is_to=0, num_total_candidates=2,
             task="I'll handle this for you.",
             candidate_email="caira@x.com", candidate_name="Caira Wong")
    )
    recipient_feats = _compute_row_features(
        _row(is_sender=0, is_to=1, task="I'll handle this for you.")
    )
    assert sender_feats["task_first_person_and_sender"] == 1.0
    assert recipient_feats["task_first_person_and_sender"] == 0.0


# ---------------------------------------------------------------------------
# Sklearn-compatible transformer
# ---------------------------------------------------------------------------


def test_extractor_emits_consistent_feature_count():
    extractor = CandidateFeatureExtractor()
    df = pd.DataFrame([
        _row().to_dict(),
        _row(is_sender=1, is_to=0).to_dict(),
    ])
    matrix = extractor.fit_transform(df)
    assert matrix.shape == (2, len(extractor.feature_names_))
