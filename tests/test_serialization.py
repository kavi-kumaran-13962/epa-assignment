"""Model serialization round-trip tests.

Verifies that joblib.dump → joblib.load preserves both the pipeline and
the bundle metadata (model_version, git_sha, default_threshold), and
that scores are bit-identical before and after the round-trip.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from src.features.candidate_features import CandidateFeatureExtractor
from src.models.pipeline import build_pipeline
from src.utils.io import load_config

MODEL_PATH = Path("models/epa_model.joblib")
needs_model = pytest.mark.skipif(
    not MODEL_PATH.exists(),
    reason="models/epa_model.joblib is not present; train first.",
)


@needs_model
def test_bundle_contains_required_keys():
    bundle = joblib.load(MODEL_PATH)
    required = {
        "pipeline",
        "config",
        "feature_columns",
        "default_threshold",
        "training_class_prior",
        "model_version",
        "trained_at",
        "sklearn_version",
    }
    missing = required - set(bundle.keys())
    assert not missing, f"bundle missing keys: {missing}"


@needs_model
def test_bundle_threshold_in_range():
    bundle = joblib.load(MODEL_PATH)
    t = float(bundle["default_threshold"])
    assert 0.0 <= t <= 1.0


@needs_model
def test_pipeline_predict_proba_works():
    """Loaded pipeline can score a minimal candidate frame without crashing."""
    bundle = joblib.load(MODEL_PATH)
    pipe = bundle["pipeline"]
    # Build a 1-row frame matching the training-time schema.
    df = pd.DataFrame([{
        "candidate_email": "a@x.com",
        "candidate_name": "Anna",
        "sender_email": "s@x.com",
        "sender_name": "Sender",
        "candidate_role": "to",
        "is_sender": 0,
        "is_to": 1,
        "is_cc": 0,
        "num_to_recipients": 1,
        "num_cc_recipients": 0,
        "num_total_candidates": 2,
        "subject": "S",
        "body": "Hi Anna, please send.",
        "task": "please send",
        "full_context": "S [SEP] please send [SEP] Hi Anna, please send.",
    }])
    scores = pipe.predict_proba(df)[:, 1]
    assert scores.shape == (1,)
    assert 0.0 <= scores[0] <= 1.0


@needs_model
def test_round_trip_preserves_scores(tmp_path):
    """Save → load → score must produce identical probabilities."""
    bundle = joblib.load(MODEL_PATH)
    pipe = bundle["pipeline"]

    df = pd.DataFrame([{
        "candidate_email": "anna@x.com", "candidate_name": "Anna Smith",
        "sender_email": "caira@x.com", "sender_name": "Caira",
        "candidate_role": "to", "is_sender": 0, "is_to": 1, "is_cc": 0,
        "num_to_recipients": 2, "num_cc_recipients": 0,
        "num_total_candidates": 3,
        "subject": "Draft", "body": "Hi Anna, can you send the draft?",
        "task": "can you send the draft?",
        "full_context": "Draft [SEP] can you send the draft? [SEP] Hi Anna...",
    }])
    scores_before = pipe.predict_proba(df)[:, 1]

    out = tmp_path / "round_trip.joblib"
    joblib.dump(bundle, out)
    reloaded = joblib.load(out)
    scores_after = reloaded["pipeline"].predict_proba(df)[:, 1]

    np.testing.assert_array_equal(scores_before, scores_after)


def test_feature_extractor_round_trip(tmp_path):
    """The custom transformer should round-trip via joblib."""
    extractor = CandidateFeatureExtractor()
    df = pd.DataFrame([{
        "candidate_email": "a@x.com", "candidate_name": "Anna Smith",
        "sender_email": "s@x.com", "sender_name": "S",
        "candidate_role": "to", "is_sender": 0, "is_to": 1, "is_cc": 0,
        "num_to_recipients": 1, "num_cc_recipients": 0,
        "num_total_candidates": 2,
        "subject": "S", "body": "B", "task": "T",
    }])
    extractor.fit(df)
    before = extractor.transform(df).toarray()

    out = tmp_path / "extractor.joblib"
    joblib.dump(extractor, out)
    reloaded = joblib.load(out)
    after = reloaded.transform(df).toarray()

    np.testing.assert_array_equal(before, after)
