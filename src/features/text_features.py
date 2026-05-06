"""TF-IDF text features over (subject + task + body) per example.

Kept deliberately small and interpretable: word n-grams with optional char
n-grams. We expose builders that play nicely inside a scikit-learn
``ColumnTransformer`` so the same pipeline drives training and inference.
"""
from __future__ import annotations

from typing import Any, Dict

from sklearn.feature_extraction.text import TfidfVectorizer


def build_word_tfidf(cfg: Dict[str, Any]) -> TfidfVectorizer:
    """Word-level TF-IDF vectorizer configured from the YAML config."""
    p = cfg["features"]["tfidf"]
    return TfidfVectorizer(
        ngram_range=tuple(p.get("ngram_range", [1, 2])),
        min_df=p.get("min_df", 2),
        max_df=p.get("max_df", 0.95),
        max_features=p.get("max_features", 50000),
        sublinear_tf=p.get("sublinear_tf", True),
        lowercase=p.get("lowercase", True),
        strip_accents="unicode",
        analyzer="word",
    )


def build_char_tfidf(cfg: Dict[str, Any]) -> TfidfVectorizer:
    """Character n-gram TF-IDF (off by default; useful for Enron's noise)."""
    p = cfg["features"]["char_tfidf"]
    return TfidfVectorizer(
        ngram_range=tuple(p.get("ngram_range", [3, 5])),
        min_df=p.get("min_df", 2),
        max_features=p.get("max_features", 30000),
        sublinear_tf=True,
        lowercase=True,
        analyzer="char_wb",
    )
