#!/usr/bin/env python3
"""End-to-end verification of the EPA pipeline.

Runs eight checks against an installed, trained repo and prints a clear
pass/fail line per stage. Exits non-zero if anything fails.

Usage::

    python3 verify.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
os.chdir(REPO_ROOT)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class CheckFailed(Exception):
    """Raised when a verification step fails."""


def header(idx: int, total: int, label: str) -> None:
    print(f"[{idx}/{total}] {label}")


def ok(message: str) -> None:
    print(f"  OK   {message}")


def info(message: str) -> None:
    print(f"       {message}")


def fail(message: str) -> None:
    raise CheckFailed(message)


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------


def check_python_version() -> None:
    if sys.version_info < (3, 9):
        fail(
            f"Python {sys.version_info.major}.{sys.version_info.minor} detected; "
            f"this project requires Python 3.9+. Run with `python3` (not `python`)."
        )
    ok(f"Python {sys.version.split()[0]}")


def check_dependencies() -> None:
    required = [
        "pandas",
        "numpy",
        "scipy",
        "sklearn",
        "joblib",
        "yaml",
        "fastapi",
        "pydantic",
        "uvicorn",
    ]
    missing = []
    for name in required:
        try:
            module = __import__(name)
            version = getattr(module, "__version__", "?")
            ok(f"{name:10s} {version}")
        except ImportError:
            missing.append(name)
    if missing:
        fail(
            f"Missing packages: {', '.join(missing)}. "
            f"Run: pip install -r requirements.txt"
        )


def check_raw_dataset() -> None:
    path = Path("data/raw/EPADataset.tsv")
    if not path.exists():
        fail(
            f"{path} not found. Run: python3 -m src.data.download "
            f"(or drop the TSV in place manually)."
        )
    size_mb = path.stat().st_size / 1e6
    with open(path, encoding="utf-8") as fp:
        n = sum(1 for _ in fp) - 1  # subtract header
    ok(f"{path}  ({size_mb:.1f} MB, {n} HITs)")
    if n < 6000:
        fail(f"expected ~6,734 HITs, got {n}")


def check_parser() -> None:
    from src.data.parse_epa import parse_dataset

    records, stats = parse_dataset("data/raw/EPADataset.tsv", consensus="majority")
    if stats.parsed_records < 6000:
        fail(f"parser only produced {stats.parsed_records} records")
    ok(f"parsed {stats.parsed_records}/{stats.total_records} HITs")
    info(f"formats: {stats.annotator_format_counts}")
    info(f"no-one HITs: {stats.no_one_records}")

    expected_keys = {
        "email_id",
        "task_id",
        "subject",
        "body",
        "task",
        "sender",
        "to",
        "cc",
        "responsible_emails",
        "no_one_responsible",
        "n_judges",
        "perfect_agreement",
    }
    missing = expected_keys - set(records[0].keys())
    if missing:
        fail(f"sample record missing keys: {missing}")
    ok(f"sample record schema OK ({len(expected_keys)} keys)")


def check_examples_csv() -> None:
    import pandas as pd

    path = Path("data/processed/examples.csv")
    if not path.exists():
        fail(
            f"{path} not found. Run: python3 -m src.data.build_examples"
        )
    df = pd.read_csv(path)
    if len(df) < 18000:
        fail(f"expected >= 18,000 rows, got {len(df)}")
    if not (0.4 < df["label"].mean() < 0.6):
        fail(f"unexpected positive rate {df['label'].mean():.3f}")

    ok(f"{len(df):,} candidate rows from {df['email_task_id'].nunique():,} HITs")
    info(f"positives: {int(df.label.sum()):,} ({df.label.mean() * 100:.1f}%)")

    expected_cols = {
        "example_id",
        "email_id",
        "task_id",
        "email_task_id",
        "candidate_email",
        "candidate_role",
        "is_sender",
        "is_to",
        "is_cc",
        "num_total_candidates",
        "task",
        "body",
        "subject",
        "full_context",
        "label",
        "perfect_agreement",
        "no_one_responsible",
    }
    missing = expected_cols - set(df.columns)
    if missing:
        fail(f"examples.csv missing columns: {missing}")
    ok(f"all {len(expected_cols)} expected columns present")

    pos_by_role = df.groupby("candidate_role")["label"].mean()
    info("positive rate by role:")
    for role, rate in pos_by_role.items():
        info(f"  {role:10s} {rate:.3f}")


def check_model_artifact() -> None:
    import joblib

    path = Path("models/epa_model.joblib")
    if not path.exists():
        fail(f"{path} missing. Run: python3 -m src.models.train")
    bundle = joblib.load(path)
    required = {"pipeline", "config", "feature_columns", "default_threshold"}
    missing = required - set(bundle.keys())
    if missing:
        fail(f"bundle missing keys: {missing}")

    ok(f"{path}  ({path.stat().st_size / 1e6:.2f} MB)")
    info(f"default_threshold = {bundle['default_threshold']}")
    info(f"training_class_prior = {bundle.get('training_class_prior', 'n/a')}")
    info(f"pipeline steps = {[s[0] for s in bundle['pipeline'].steps]}")


def check_held_out_metrics() -> None:
    import joblib
    from sklearn.metrics import (
        average_precision_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    from src.models.split import assert_no_group_leakage, group_train_test_split
    from src.models.train import _read_examples
    from src.utils.io import load_config

    cfg = load_config("config.yaml")
    df = _read_examples(cfg["paths"]["examples_csv"])
    train_df, val_df, test_df = group_train_test_split(
        df,
        group_key=cfg["split"]["group_key"],
        test_size=cfg["split"]["test_size"],
        val_size=cfg["split"]["val_size"],
        random_state=cfg["split"]["random_state"],
    )
    try:
        assert_no_group_leakage(
            train_df, val_df, test_df, group_key=cfg["split"]["group_key"]
        )
    except AssertionError as exc:
        fail(f"leakage check failed: {exc}")
    ok(
        f"no group leakage: train={len(train_df):,} "
        f"val={len(val_df):,} test={len(test_df):,}"
    )

    bundle = joblib.load(cfg["paths"]["model_path"])
    y_test = test_df["label"].to_numpy()
    y_score = bundle["pipeline"].predict_proba(test_df)[:, 1]
    threshold = bundle["default_threshold"]
    y_pred = (y_score >= threshold).astype(int)

    p = precision_score(y_test, y_pred, zero_division=0)
    r = recall_score(y_test, y_pred, zero_division=0)
    f = f1_score(y_test, y_pred, zero_division=0)
    pr_auc = average_precision_score(y_test, y_score)
    roc_auc = roc_auc_score(y_test, y_score)

    if p < 0.78 or r < 0.83 or f < 0.83:
        fail(
            f"metric regression detected: P={p:.4f} R={r:.4f} F1={f:.4f} "
            f"(thresholds: P>=0.78, R>=0.83, F1>=0.83)"
        )

    tp = int(((y_pred == 1) & (y_test == 1)).sum())
    fp = int(((y_pred == 1) & (y_test == 0)).sum())
    fn = int(((y_pred == 0) & (y_test == 1)).sum())
    tn = int(((y_pred == 0) & (y_test == 0)).sum())

    ok(f"precision = {p:.4f}  (README: ~0.79)")
    ok(f"recall    = {r:.4f}  (README: ~0.92)")
    ok(f"F1        = {f:.4f}  (README: ~0.85)")
    ok(f"PR-AUC    = {pr_auc:.4f}  (README: ~0.92)")
    ok(f"ROC-AUC   = {roc_auc:.4f}  (README: ~0.93)")
    info(f"confusion : TP={tp}  FP={fp}  FN={fn}  TN={tn}")


def check_cli_predict() -> None:
    cmd = [
        sys.executable,
        "-m",
        "src.models.predict",
        "--input",
        "example_payload.json",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        fail(f"predict CLI exited {result.returncode}: {result.stderr.strip()}")

    response = json.loads(result.stdout)
    ok(f'task echoed: "{response["task"]}"')
    ok(f"threshold: {response['threshold']}")
    ok(f"{len(response['assignments'])} assignments returned")

    expected = {
        "anna@example.com": True,
        "brad@example.com": True,
        "caira@example.com": False,
    }
    got = {a["email"]: a["assigned"] for a in response["assignments"]}
    mismatches = [
        (k, expected[k], got.get(k)) for k in expected if got.get(k) != expected[k]
    ]
    if mismatches:
        fail(f"assignment mismatches: {mismatches}")
    for assignment in response["assignments"]:
        info(
            f"  {assignment['email']:30s} score={assignment['score']:.3f}  "
            f"role={assignment['role']:8s}  assigned={assignment['assigned']}"
        )


def check_fastapi_service() -> None:
    from fastapi.testclient import TestClient

    from src.service.app import app

    client = TestClient(app)

    health = client.get("/health")
    if health.status_code != 200:
        fail(f"GET /health returned {health.status_code}")
    ok(f"GET /health = 200  {health.json()}")

    payload = json.load(open("example_payload.json"))
    response = client.post("/predict", json=payload)
    if response.status_code != 200:
        fail(f"POST /predict returned {response.status_code}: {response.text}")
    body = response.json()
    if len(body["assignments"]) != 4:
        fail(f"expected 4 assignments, got {len(body['assignments'])}")
    ok(
        f"POST /predict = 200, threshold={body['threshold']}, "
        f"n={len(body['assignments'])}"
    )

    # Threshold override via query param.
    high = client.post("/predict?threshold=0.8", json=payload)
    n_assigned_high = sum(1 for a in high.json()["assignments"] if a["assigned"])
    n_assigned_low = sum(1 for a in body["assignments"] if a["assigned"])
    ok(
        f"POST /predict?threshold=0.8 -> {n_assigned_high} assigned "
        f"(was {n_assigned_low} at threshold=0.5)"
    )

    # Bad payload (missing required `task`) should be rejected by Pydantic.
    bad = client.post(
        "/predict", json={"sender": "a@b", "to": [], "cc": []}
    )
    if bad.status_code != 422:
        fail(f"expected 422 on bad payload, got {bad.status_code}")
    ok(f"POST /predict (bad payload) = 422 (validation rejected)")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


CHECKS = [
    ("Python version", check_python_version),
    ("Dependencies", check_dependencies),
    ("Raw EPA dataset on disk", check_raw_dataset),
    ("Parser loads the TSV", check_parser),
    ("examples.csv shape + schema", check_examples_csv),
    ("Model artifact", check_model_artifact),
    ("Held-out test metrics", check_held_out_metrics),
    ("CLI predict on example payload", check_cli_predict),
    ("FastAPI service", check_fastapi_service),
]


def main() -> int:
    print("=" * 60)
    print("EPA pipeline verification suite")
    print("=" * 60)

    failed = []
    for idx, (label, fn) in enumerate(CHECKS, start=1):
        header(idx, len(CHECKS), label)
        try:
            fn()
        except CheckFailed as exc:
            print(f"  FAIL {exc}")
            failed.append(label)
        except Exception as exc:  # noqa: BLE001 — unexpected failure shouldn't kill the whole run
            print(f"  FAIL unexpected {type(exc).__name__}: {exc}")
            failed.append(label)
        print()

    print("=" * 60)
    if failed:
        print(f"FAILED ({len(failed)}/{len(CHECKS)}): {', '.join(failed)}")
        return 1
    print(f"All {len(CHECKS)} checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
