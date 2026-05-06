"""Parser tests.

Cover the EPA-shaped record schema, candidate extraction, deduplication,
no-one handling, and the three consensus modes.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.data.parse_epa import (
    _coerce_address,
    _coerce_address_list,
    _judgement_to_email_set,
    _resolve_responsible,
    parse_dataset,
    ParseStats,
)


# ---------------------------------------------------------------------------
# Address coercion
# ---------------------------------------------------------------------------


def test_coerce_address_reference_epa_shape():
    """The reference EPA TSV's nested {'emailAddress': {'Name', 'Address'}}."""
    name, email = _coerce_address(
        {"emailAddress": {"Name": "Anna Smith", "Address": "Anna@Enron.com"}}
    )
    assert name == "Anna Smith"
    assert email == "anna@enron.com"  # lowercased


def test_coerce_address_simple_dict():
    name, email = _coerce_address({"name": "Bob", "email": "BOB@x.com"})
    assert name == "Bob"
    assert email == "bob@x.com"


def test_coerce_address_free_text_with_brackets():
    name, email = _coerce_address("Alice <alice@example.com>")
    assert name == "Alice"
    assert email == "alice@example.com"


def test_coerce_address_bare_email():
    name, email = _coerce_address("solo@example.com")
    assert name == ""
    assert email == "solo@example.com"


def test_coerce_address_list_handles_emailAddressList_wrapper():
    pairs = _coerce_address_list(
        {"emailAddressList": [
            {"emailAddress": {"Name": "A", "Address": "a@x.com"}},
            {"emailAddress": {"Name": "B", "Address": "b@x.com"}},
        ]}
    )
    assert pairs == [("A", "a@x.com"), ("B", "b@x.com")]


def test_coerce_address_list_dedupes_by_email():
    """Same email twice (case-different) should collapse to one entry."""
    pairs = _coerce_address_list([
        {"emailAddress": {"Name": "A", "Address": "x@x.com"}},
        {"emailAddress": {"Name": "A2", "Address": "X@X.COM"}},
    ])
    assert len(pairs) == 1
    assert pairs[0][1] == "x@x.com"


# ---------------------------------------------------------------------------
# Judgement aggregation
# ---------------------------------------------------------------------------


def test_judgement_reference_shape_single_key():
    s = _judgement_to_email_set({"annotator_1": ["A@x.com", "b@x.com"]})
    assert s == frozenset({"a@x.com", "b@x.com"})


def test_judgement_empty_list_is_no_one():
    s = _judgement_to_email_set({"annotator_1": []})
    assert s == frozenset()


def test_judgement_fork_shape_explicit_no_one():
    s = _judgement_to_email_set({"no_one": True})
    assert s == frozenset()


def test_judgement_fork_shape_responsible_emails():
    s = _judgement_to_email_set({"responsible_emails": ["x@y.com"]})
    assert s == frozenset({"x@y.com"})


def test_resolve_responsible_perfect_consensus_drops_disagreement():
    stats = ParseStats()
    annotations = [
        {"j1": ["a@x.com"]},
        {"j2": ["b@x.com"]},  # disagrees
    ]
    resp, no_one = _resolve_responsible(
        annotations, ["a@x.com", "b@x.com"], "s@x.com", "perfect", stats
    )
    assert resp is None  # signal to drop the HIT


def test_resolve_responsible_perfect_consensus_keeps_agreement():
    stats = ParseStats()
    annotations = [{"j1": ["a@x.com"]}, {"j2": ["a@x.com"]}]
    resp, no_one = _resolve_responsible(
        annotations, ["a@x.com", "b@x.com"], "s@x.com", "perfect", stats
    )
    assert resp == ["a@x.com"]
    assert no_one is False


def test_resolve_responsible_majority_per_recipient_vote():
    stats = ParseStats()
    annotations = [
        {"j1": ["a@x.com"]},
        {"j2": ["a@x.com", "b@x.com"]},
        {"j3": ["a@x.com"]},
    ]
    resp, no_one = _resolve_responsible(
        annotations, ["a@x.com", "b@x.com"], "s@x.com", "majority", stats
    )
    # 'a' has 3 votes; 'b' only 1. Majority is at least 2/3.
    assert resp == ["a@x.com"]
    assert no_one is False


def test_resolve_responsible_majority_yields_no_one_when_all_empty():
    stats = ParseStats()
    annotations = [{"j1": []}, {"j2": []}, {"j3": []}]
    resp, no_one = _resolve_responsible(
        annotations, ["a@x.com"], "s@x.com", "majority", stats
    )
    assert resp == []
    assert no_one is True


# ---------------------------------------------------------------------------
# End-to-end parse_dataset on a synthetic TSV
# ---------------------------------------------------------------------------


def _write_tiny_tsv(tmp_path: Path) -> Path:
    """Build a 2-row EPA-shaped TSV in tmp_path and return its path."""
    p = tmp_path / "tiny.tsv"
    rows = [
        # HIT 1: task assigned to Anna; sender Caira; Brad on Cc.
        {
            "EmailID": "E1",
            "Subject": "Draft",
            "From": {"emailAddress": {"Name": "Caira", "Address": "caira@x.com"}},
            "ToRecipients": {"emailAddressList": [
                {"emailAddress": {"Name": "Anna", "Address": "anna@x.com"}},
            ]},
            "CcRecipients": {"emailAddressList": [
                {"emailAddress": {"Name": "Brad", "Address": "brad@x.com"}},
            ]},
            "Message": "<mark>please send the draft</mark>",
            "TaskSentence": "please send the draft",
            "Judgements": [{"j1": ["anna@x.com"]}, {"j2": ["anna@x.com"]}],
        },
        # HIT 2: no-one is responsible (all annotators agree on empty).
        {
            "EmailID": "E2",
            "Subject": "Heads-up",
            "From": {"emailAddress": {"Name": "S", "Address": "s@x.com"}},
            "ToRecipients": {"emailAddressList": [
                {"emailAddress": {"Name": "T", "Address": "t@x.com"}},
            ]},
            "CcRecipients": {"emailAddressList": []},
            "Message": "<mark>Brad will handle the draft.</mark>",
            "TaskSentence": "Brad will handle the draft.",
            "Judgements": [{"j1": []}, {"j2": []}],
        },
    ]
    with open(p, "w", encoding="utf-8") as fp:
        fp.write("Key\tJson\n")
        for i, r in enumerate(rows):
            fp.write(f"EPA-{i:05d}\t{json.dumps(r)}\n")
    return p


def test_parse_dataset_majority_end_to_end(tmp_path: Path):
    path = _write_tiny_tsv(tmp_path)
    records, stats = parse_dataset(path, consensus="majority")
    assert stats.parsed_records == 2
    r1, r2 = records
    # HIT 1: Anna is responsible.
    assert r1["responsible_emails"] == ["anna@x.com"]
    assert r1["no_one_responsible"] is False
    # HIT 2: no-one is responsible.
    assert r2["responsible_emails"] == []
    assert r2["no_one_responsible"] is True


def test_parse_dataset_perfect_drops_disagreements(tmp_path: Path):
    """A HIT where annotators disagree should drop under 'perfect'."""
    p = tmp_path / "tiny.tsv"
    payload = {
        "EmailID": "E1",
        "Subject": "S",
        "From": {"emailAddress": {"Name": "A", "Address": "a@x.com"}},
        "ToRecipients": {"emailAddressList": [
            {"emailAddress": {"Name": "B", "Address": "b@x.com"}},
            {"emailAddress": {"Name": "C", "Address": "c@x.com"}},
        ]},
        "CcRecipients": {"emailAddressList": []},
        "Message": "<mark>send</mark>",
        "TaskSentence": "send",
        "Judgements": [{"j1": ["b@x.com"]}, {"j2": ["c@x.com"]}],
    }
    with open(p, "w", encoding="utf-8") as fp:
        fp.write("Key\tJson\n")
        fp.write(f"EPA-00000\t{json.dumps(payload)}\n")

    records, stats = parse_dataset(p, consensus="perfect")
    assert stats.consensus_dropped == 1
    assert stats.parsed_records == 0
