"""Structured features for the (email, task, candidate) tuple.

This is where the bulk of the EPA-specific signal lives:

* candidate role flags (sender / to / cc / multi-role / only recipient),
* domain match between candidate and sender,
* presence of the candidate's first/last/full name and email in the
  task / body / subject (the addressee-tagging signal),
* email-pragmatics features over the task text (you / your / please /
  question marks, imperative verbs, etc.),
* simple proximity features (does the candidate appear in the body near
  the task sentence, before/after, character distance).

Everything is implemented as a scikit-learn-compatible transformer
(:class:`CandidateFeatureExtractor`) so it slots into the same Pipeline
as the TF-IDF features and gets serialized in one go via joblib.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin

from src.utils.text_cleaning import (
    email_domain,
    email_local_part,
    normalize_email,
    split_full_name,
    tokenize_basic,
)

# Verbs commonly used to assign tasks in the Enron corpus. The list is
# small on purpose – we want the feature to fire frequently, not perfectly.
IMPERATIVE_VERBS = {
    "send", "complete", "review", "prepare", "handle", "provide", "update",
    "check", "confirm", "call", "schedule", "draft", "forward", "approve",
    "submit", "respond", "follow", "make", "share", "deliver", "finish",
    "give", "set", "let", "verify", "discuss", "fix", "clarify",
}


# ---------------------------------------------------------------------------
# Feature configuration
# ---------------------------------------------------------------------------


@dataclass
class CandidateFeatureNames:
    """Stable list of feature columns this transformer emits."""

    role: List[str]
    nameref: List[str]
    pragmatics: List[str]
    proximity: List[str]
    counts: List[str]

    def all(self) -> List[str]:
        return self.role + self.nameref + self.pragmatics + self.counts + self.proximity


def _feature_names() -> CandidateFeatureNames:
    return CandidateFeatureNames(
        role=[
            "is_sender",
            "is_to",
            "is_cc",
            "appears_in_multiple_roles",
            "is_only_recipient",
            "candidate_email_missing",
            "candidate_name_missing",
            "sender_same_domain_as_candidate",
        ],
        nameref=[
            "first_name_in_task",
            "last_name_in_task",
            "full_name_in_task",
            "email_in_task",
            "local_part_in_task",
            "first_name_in_body",
            "last_name_in_body",
            "full_name_in_body",
            "email_in_body",
            "local_part_in_body",
            "first_name_in_subject",
            "last_name_in_subject",
            "full_name_in_subject",
            "email_in_subject",
        ],
        pragmatics=[
            "task_contains_you",
            "task_contains_your",
            "task_contains_please",
            "task_contains_can_you",
            "task_contains_could_you",
            "task_contains_would_you",
            "task_contains_let_me_know",
            "task_contains_question_mark",
            "task_is_question",
            "task_contains_we",
            "task_contains_us",
            "task_contains_team",
            "task_contains_imperative_verb",
            "task_length_tokens",
            "body_length_tokens",
            "subject_length_tokens",
        ],
        counts=[
            "num_to_recipients",
            "num_cc_recipients",
            "num_total_candidates",
            "log_num_total_candidates",
        ],
        proximity=[
            "candidate_in_same_sentence_as_task",
            "candidate_appears_before_task",
            "candidate_appears_after_task",
            "min_char_distance_to_task_norm",
        ],
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_'\-]+")


def _word_in(text: str, word: str) -> bool:
    if not text or not word:
        return False
    return re.search(rf"\b{re.escape(word)}\b", text, flags=re.IGNORECASE) is not None


def _phrase_in(text: str, phrase: str) -> bool:
    if not text or not phrase:
        return False
    return re.search(re.escape(phrase), text, flags=re.IGNORECASE) is not None


def _split_sentences(text: str) -> List[str]:
    if not text:
        return []
    # Split on . ! ? followed by whitespace, plus newlines.
    return [
        s.strip()
        for s in re.split(r"(?<=[.!?])\s+|\n+", text)
        if s.strip()
    ]


def _safe_index(haystack: str, needle: str) -> int:
    if not haystack or not needle:
        return -1
    m = re.search(re.escape(needle), haystack, flags=re.IGNORECASE)
    return m.start() if m else -1


# ---------------------------------------------------------------------------
# Per-row feature computation
# ---------------------------------------------------------------------------


def _compute_row_features(row: pd.Series) -> Dict[str, float]:
    cand_email = normalize_email(row.get("candidate_email", "") or "")
    cand_name = (row.get("candidate_name", "") or "").strip()
    sender_email = normalize_email(row.get("sender_email", "") or "")
    body = str(row.get("body", "") or "")
    task = str(row.get("task", "") or "")
    subject = str(row.get("subject", "") or "")
    task_lower = task.lower()
    first, last = split_full_name(cand_name)
    local = email_local_part(cand_email)

    is_sender = int(row.get("is_sender", 0) or 0)
    is_to = int(row.get("is_to", 0) or 0)
    is_cc = int(row.get("is_cc", 0) or 0)
    num_to = int(row.get("num_to_recipients", 0) or 0)
    num_cc = int(row.get("num_cc_recipients", 0) or 0)
    num_total = int(row.get("num_total_candidates", 0) or 0)

    multi_role = int(sum([is_sender, is_to, is_cc]) > 1)
    only_recipient = int(num_total - is_sender == 1 and is_sender == 0)
    if num_total == 1:
        only_recipient = 1
    candidate_email_missing = int(not cand_email)
    candidate_name_missing = int(not cand_name)
    sender_same_domain = int(
        bool(email_domain(cand_email))
        and email_domain(cand_email) == email_domain(sender_email)
    )

    # Name reference features. We coerce via ``bool(...)`` first so that
    # empty strings (e.g. a candidate with a missing name) don't end up
    # short-circuiting to a non-numeric ``''`` before the ``int()`` cast.
    def _bi(x: Any) -> int:
        return int(bool(x))

    nameref = {
        "first_name_in_task": _bi(len(first) > 1 and _word_in(task, first)),
        "last_name_in_task": _bi(len(last) > 1 and _word_in(task, last)),
        "full_name_in_task": _bi(cand_name and _phrase_in(task, cand_name)),
        "email_in_task": _bi(cand_email and _phrase_in(task, cand_email)),
        "local_part_in_task": _bi(len(local) > 2 and _word_in(task, local)),
        "first_name_in_body": _bi(len(first) > 1 and _word_in(body, first)),
        "last_name_in_body": _bi(len(last) > 1 and _word_in(body, last)),
        "full_name_in_body": _bi(cand_name and _phrase_in(body, cand_name)),
        "email_in_body": _bi(cand_email and _phrase_in(body, cand_email)),
        "local_part_in_body": _bi(len(local) > 2 and _word_in(body, local)),
        "first_name_in_subject": _bi(len(first) > 1 and _word_in(subject, first)),
        "last_name_in_subject": _bi(len(last) > 1 and _word_in(subject, last)),
        "full_name_in_subject": _bi(cand_name and _phrase_in(subject, cand_name)),
        "email_in_subject": _bi(cand_email and _phrase_in(subject, cand_email)),
    }

    # Pragmatics features.
    pragmatics = {
        "task_contains_you": int(_word_in(task, "you")),
        "task_contains_your": int(_word_in(task, "your")),
        "task_contains_please": int(_word_in(task, "please")),
        "task_contains_can_you": int(_phrase_in(task, "can you")),
        "task_contains_could_you": int(_phrase_in(task, "could you")),
        "task_contains_would_you": int(_phrase_in(task, "would you")),
        "task_contains_let_me_know": int(_phrase_in(task, "let me know")),
        "task_contains_question_mark": int("?" in task),
        "task_is_question": int(task_lower.lstrip().startswith(
            ("can ", "could ", "would ", "will ", "do ", "does ", "is ", "are ", "should ")
        ) or task.endswith("?")),
        "task_contains_we": int(_word_in(task, "we")),
        "task_contains_us": int(_word_in(task, "us")),
        "task_contains_team": int(_word_in(task, "team")),
        "task_contains_imperative_verb": int(
            any(w in IMPERATIVE_VERBS for w in tokenize_basic(task)[:6])
        ),
        "task_length_tokens": float(len(tokenize_basic(task))),
        "body_length_tokens": float(len(tokenize_basic(body))),
        "subject_length_tokens": float(len(tokenize_basic(subject))),
    }

    # Proximity features. We take the most informative of (full name, first
    # name, email/local part) for distance to task in the body.
    sentences = _split_sentences(body)
    same_sentence = 0
    if cand_name or cand_email or first:
        for s in sentences:
            in_s = (
                (cand_name and _phrase_in(s, cand_name))
                or (first and len(first) > 1 and _word_in(s, first))
                or (cand_email and _phrase_in(s, cand_email))
            )
            if in_s and task and _phrase_in(s, task[:80]):
                same_sentence = 1
                break

    name_idx = max(
        _safe_index(body, cand_name) if cand_name else -1,
        _safe_index(body, cand_email) if cand_email else -1,
        _safe_index(body, first) if first and len(first) > 1 else -1,
    )
    task_idx = _safe_index(body, task[:80]) if task else -1

    if name_idx >= 0 and task_idx >= 0:
        before = int(name_idx < task_idx)
        after = int(name_idx > task_idx)
        body_len = max(len(body), 1)
        min_dist = abs(name_idx - task_idx) / body_len
    else:
        before = after = 0
        min_dist = 1.0  # "not found" -> max distance

    proximity = {
        "candidate_in_same_sentence_as_task": same_sentence,
        "candidate_appears_before_task": before,
        "candidate_appears_after_task": after,
        "min_char_distance_to_task_norm": float(min_dist),
    }

    counts = {
        "num_to_recipients": float(num_to),
        "num_cc_recipients": float(num_cc),
        "num_total_candidates": float(num_total),
        "log_num_total_candidates": float(np.log1p(num_total)),
    }

    role = {
        "is_sender": float(is_sender),
        "is_to": float(is_to),
        "is_cc": float(is_cc),
        "appears_in_multiple_roles": float(multi_role),
        "is_only_recipient": float(only_recipient),
        "candidate_email_missing": float(candidate_email_missing),
        "candidate_name_missing": float(candidate_name_missing),
        "sender_same_domain_as_candidate": float(sender_same_domain),
    }

    # Merge in canonical column order.
    out: Dict[str, float] = {}
    for d in (role, nameref, pragmatics, counts, proximity):
        out.update(d)
    return out


# ---------------------------------------------------------------------------
# Sklearn-compatible transformer
# ---------------------------------------------------------------------------


class CandidateFeatureExtractor(BaseEstimator, TransformerMixin):
    """Materialize the structured candidate features as a sparse matrix."""

    def __init__(self) -> None:
        self.feature_names_: List[str] = _feature_names().all()

    def fit(self, X: pd.DataFrame, y: Any = None) -> "CandidateFeatureExtractor":
        return self

    def transform(self, X: pd.DataFrame) -> sparse.csr_matrix:
        if isinstance(X, pd.Series):
            X = X.to_frame().T
        rows = [
            [feat_dict[name] for name in self.feature_names_]
            for feat_dict in (_compute_row_features(r) for _, r in X.iterrows())
        ]
        if not rows:
            return sparse.csr_matrix((0, len(self.feature_names_)))
        return sparse.csr_matrix(np.asarray(rows, dtype=float))

    # Convenience for downstream consumers (e.g. coefficient inspection).
    def get_feature_names_out(
        self, input_features: Sequence[str] | None = None
    ) -> np.ndarray:
        return np.asarray(self.feature_names_, dtype=object)
