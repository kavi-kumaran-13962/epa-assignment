"""FastAPI service tests.

Covers the happy path plus the failure modes a production caller will
actually hit: missing fields, malformed payloads, unicode, edge thresholds,
and weird recipient shapes.

These run only when a model artifact is present. They skip cleanly when
the model hasn't been trained yet — convenient for CI environments that
don't pre-train.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

MODEL_PATH = Path("models/epa_model.joblib")
needs_model = pytest.mark.skipif(
    not MODEL_PATH.exists(),
    reason="models/epa_model.joblib is not present; train first.",
)


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    from src.service.app import app, _load_bundle

    # Clear the lru_cache so the test sees a fresh load if the file changed.
    _load_bundle.cache_clear()
    return TestClient(app)


@pytest.fixture(scope="module")
def example_payload():
    return json.loads(Path("example_payload.json").read_text())


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


def test_health_does_not_require_model(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "model_path" in body
    assert "model_present" in body


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


@needs_model
def test_predict_happy_path(client, example_payload):
    r = client.post("/predict", json=example_payload)
    assert r.status_code == 200
    body = r.json()
    assert body["task"] == example_payload["task"]
    assert isinstance(body["assignments"], list)
    assert len(body["assignments"]) == 4

    by_email = {a["email"]: a for a in body["assignments"]}
    # Sender should be far below threshold; named recipients above.
    assert by_email["caira@example.com"]["assigned"] is False
    assert by_email["anna@example.com"]["assigned"] is True


@needs_model
def test_threshold_override_via_query_param(client, example_payload):
    low = client.post("/predict?threshold=0.1", json=example_payload).json()
    high = client.post("/predict?threshold=0.95", json=example_payload).json()
    n_low = sum(1 for a in low["assignments"] if a["assigned"])
    n_high = sum(1 for a in high["assignments"] if a["assigned"])
    assert n_low >= n_high  # lower threshold → at least as many assigned


# ---------------------------------------------------------------------------
# Validation rejection
# ---------------------------------------------------------------------------


def test_missing_task_field_returns_422(client):
    r = client.post(
        "/predict",
        json={"sender": "a@b.com", "to": ["t@x.com"], "cc": []},
    )
    assert r.status_code == 422
    assert "task" in r.text


def test_threshold_out_of_range_returns_422(client, example_payload):
    """Out-of-range thresholds should be rejected at the schema layer."""
    r = client.post("/predict?threshold=1.5", json=example_payload)
    assert r.status_code == 422


def test_negative_threshold_returns_422(client, example_payload):
    r = client.post("/predict?threshold=-0.1", json=example_payload)
    assert r.status_code == 422


def test_malformed_json_returns_422(client):
    r = client.post(
        "/predict",
        content="{ this is not valid json",
        headers={"Content-Type": "application/json"},
    )
    assert r.status_code == 422


# ---------------------------------------------------------------------------
# Edge-case payloads that should NOT crash the service
# ---------------------------------------------------------------------------


@needs_model
def test_empty_body_and_subject_succeeds(client):
    r = client.post(
        "/predict",
        json={
            "sender": "s@x.com", "to": ["a@x.com"], "cc": [],
            "subject": "", "body": "", "task": "send the report",
        },
    )
    assert r.status_code == 200
    assert len(r.json()["assignments"]) == 2


@needs_model
def test_unicode_payload_succeeds(client):
    r = client.post(
        "/predict",
        json={
            "sender": {"name": "Süßer", "email": "s@x.com"},
            "to": [{"name": "Łukasz", "email": "l@x.com"}],
            "cc": [],
            "subject": "Café — résumé",
            "body": "中文 العربية 🎉",
            "task": "请发送报告",
        },
    )
    assert r.status_code == 200
    assert len(r.json()["assignments"]) == 2


@needs_model
def test_long_body_does_not_crash(client):
    r = client.post(
        "/predict",
        json={
            "sender": "s@x.com", "to": ["a@x.com"], "cc": [],
            "subject": "S", "body": "x " * 5000, "task": "please send",
        },
    )
    assert r.status_code == 200


@needs_model
def test_duplicate_recipient_collapses(client):
    """Same email twice (case-different) should collapse to a single assignment."""
    r = client.post(
        "/predict",
        json={
            "sender": "s@x.com",
            "to": ["a@x.com", "A@X.com"],
            "cc": [],
            "subject": "S", "body": "B", "task": "send",
        },
    )
    body = r.json()
    emails = [a["email"] for a in body["assignments"]]
    assert emails.count("a@x.com") == 1


@needs_model
def test_email_in_to_and_cc_gets_multiple_role(client):
    r = client.post(
        "/predict",
        json={
            "sender": "s@x.com",
            "to": ["a@x.com"],
            "cc": ["a@x.com"],
            "subject": "S", "body": "B", "task": "send",
        },
    )
    body = r.json()
    anna = next(a for a in body["assignments"] if a["email"] == "a@x.com")
    assert anna["role"] == "multiple"


@needs_model
def test_no_candidates_returns_empty_assignments(client):
    r = client.post(
        "/predict",
        json={"sender": "", "to": [], "cc": [],
              "subject": "", "body": "", "task": "T"},
    )
    assert r.status_code == 200
    assert r.json()["assignments"] == []


@needs_model
def test_50_recipients_does_not_crash(client):
    r = client.post(
        "/predict",
        json={
            "sender": "s@x.com",
            "to": [f"r{i}@x.com" for i in range(50)],
            "cc": [], "subject": "S", "body": "B", "task": "please review",
        },
    )
    assert r.status_code == 200
    assert len(r.json()["assignments"]) == 51  # sender + 50


@needs_model
def test_string_addresses_with_brackets(client):
    """The 'Name <addr>' string syntax parses into the right name+email."""
    r = client.post(
        "/predict",
        json={
            "sender": "Sender <s@x.com>",
            "to": ["Anna <anna@x.com>", "Brad <brad@x.com>"],
            "cc": [], "subject": "S",
            "body": "Hi Anna, send this please.",
            "task": "Hi Anna, send this please.",
        },
    )
    assert r.status_code == 200
    body = r.json()
    by_email = {a["email"]: a for a in body["assignments"]}
    # All three candidates parsed and scored.
    assert {"s@x.com", "anna@x.com", "brad@x.com"} <= set(by_email.keys())
    # Names were extracted (not absorbed into the email field).
    assert by_email["anna@x.com"]["name"] == "Anna"
    assert by_email["brad@x.com"]["name"] == "Brad"
    # Roles correctly identified.
    assert by_email["s@x.com"]["role"] == "sender"
    assert by_email["anna@x.com"]["role"] == "to"
