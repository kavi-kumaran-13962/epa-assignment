"""Build the unified train/inference pipeline.

We combine three blocks via :class:`~sklearn.compose.ColumnTransformer`:

1. word-level TF-IDF over ``full_context`` (subject + task + body),
2. optional character-level TF-IDF over the same field,
3. structured ``CandidateFeatureExtractor`` over the rest of the row.

The trained pipeline is serialized in one go via :mod:`joblib`, so the
inference service reuses the *exact* same feature engineering used during
training. This is the single most important property for reproducibility.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.features.candidate_features import CandidateFeatureExtractor
from src.features.text_features import build_char_tfidf, build_word_tfidf

STRUCTURED_COLUMNS = [
    "candidate_email",
    "candidate_name",
    "sender_email",
    "sender_name",
    "candidate_role",
    "is_sender",
    "is_to",
    "is_cc",
    "num_to_recipients",
    "num_cc_recipients",
    "num_total_candidates",
    "subject",
    "body",
    "task",
]


def build_pipeline(cfg: Dict[str, Any]) -> Tuple[Pipeline, ColumnTransformer]:
    """Return the (estimator pipeline, feature_union) pair.

    Returning the column transformer separately is useful for diagnostic
    feature-importance inspection.
    """
    word = build_word_tfidf(cfg)
    transformers = [
        ("word_tfidf", word, "full_context"),
        ("structured", CandidateFeatureExtractor(), STRUCTURED_COLUMNS),
    ]
    if cfg["features"]["char_tfidf"].get("enabled"):
        transformers.insert(1, ("char_tfidf", build_char_tfidf(cfg), "full_context"))

    features = ColumnTransformer(transformers, sparse_threshold=1.0)

    m = cfg["model"]
    clf = LogisticRegression(
        C=m.get("C", 1.0),
        class_weight=(None if m.get("class_weight") in (None, "none", "None") else m.get("class_weight", "balanced")),
        max_iter=m.get("max_iter", 1000),
        solver=m.get("solver", "liblinear"),
        random_state=m.get("random_state", 42),
    )
    pipe = Pipeline([("features", features), ("clf", clf)])
    return pipe, features
