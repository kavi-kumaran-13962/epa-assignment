"""Lightweight inference helper used by both the CLI and the FastAPI service.

The function builds a one-row-per-candidate dataframe from a single email
payload and delegates feature engineering to the trained pipeline. This
guarantees train-time and inference-time features stay in lockstep.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import joblib
import pandas as pd

from src.utils.io import load_config, setup_logging
from src.utils.text_cleaning import (
    clean_text,
    deduplicate_addresses,
    normalize_email,
    normalize_name,
)

logger = logging.getLogger(__name__)


def _coerce_to_pairs(items: Iterable[Any]) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    for item in items or []:
        if isinstance(item, dict):
            pairs.append(
                (
                    normalize_name(item.get("name") or ""),
                    normalize_email(item.get("email") or item.get("address") or ""),
                )
            )
        else:
            text = str(item)
            if "<" in text and ">" in text:
                name = text.split("<", 1)[0].strip()
                addr = text.split("<", 1)[1].split(">", 1)[0]
                pairs.append((normalize_name(name), normalize_email(addr)))
            else:
                pairs.append(("", normalize_email(text)))
    return deduplicate_addresses(pairs)


def build_candidate_frame(payload: Dict[str, Any]) -> pd.DataFrame:
    """Produce the candidate-level dataframe for a single inference request."""
    sender_raw = payload.get("sender") or ""
    if isinstance(sender_raw, dict):
        sender_name = normalize_name(sender_raw.get("name") or "")
        sender_email = normalize_email(sender_raw.get("email") or "")
    else:
        text = str(sender_raw)
        if "<" in text and ">" in text:
            sender_name = normalize_name(text.split("<", 1)[0].strip())
            sender_email = normalize_email(text.split("<", 1)[1].split(">", 1)[0])
        else:
            sender_name = ""
            sender_email = normalize_email(text)

    to_pairs = _coerce_to_pairs(payload.get("to") or [])
    cc_pairs = _coerce_to_pairs(payload.get("cc") or [])

    subject = clean_text(payload.get("subject") or "")
    body = clean_text(payload.get("body") or "")
    task = clean_text(payload.get("task") or "")

    candidates: Dict[str, Dict[str, Any]] = {}

    def _add(name: str, email: str, role: str) -> None:
        key = email or f"name:{name.lower()}"
        if not key:
            return
        slot = candidates.setdefault(key, {"name": name, "email": email, "roles": set()})
        if name and not slot["name"]:
            slot["name"] = name
        slot["roles"].add(role)

    if sender_name or sender_email:
        _add(sender_name, sender_email, "sender")
    for n, e in to_pairs:
        _add(n, e, "to")
    for n, e in cc_pairs:
        _add(n, e, "cc")

    rows: List[Dict[str, Any]] = []
    full_context = " [SEP] ".join(x for x in [subject, task, body] if x)
    num_to = len([e for _, e in to_pairs if e])
    num_cc = len([e for _, e in cc_pairs if e])
    num_total = len(candidates)

    for key, slot in sorted(candidates.items()):
        roles = slot["roles"]
        if len(roles) > 1:
            role = "multiple"
        else:
            role = next(iter(roles))
        rows.append(
            {
                "candidate_email": slot["email"],
                "candidate_name": slot["name"],
                "sender_email": sender_email,
                "sender_name": sender_name,
                "candidate_role": role,
                "is_sender": int("sender" in roles),
                "is_to": int("to" in roles),
                "is_cc": int("cc" in roles),
                "num_to_recipients": num_to,
                "num_cc_recipients": num_cc,
                "num_total_candidates": num_total,
                "subject": subject,
                "body": body,
                "task": task,
                "full_context": full_context,
                "no_one_responsible": 0,  # unknown at inference time
            }
        )
    return pd.DataFrame(rows)


def predict_payload(
    payload: Dict[str, Any],
    bundle: Dict[str, Any],
    threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """Return the assignment dict for the FastAPI-style payload."""
    pipe = bundle["pipeline"]
    threshold = float(threshold if threshold is not None else bundle.get("default_threshold", 0.5))
    df = build_candidate_frame(payload)
    if len(df) == 0:
        return {"task": payload.get("task", ""), "threshold": threshold, "assignments": []}
    scores = pipe.predict_proba(df)[:, 1]
    assignments = []
    for (_, row), score in zip(df.iterrows(), scores):
        person = row["candidate_email"] or row["candidate_name"]
        assignments.append(
            {
                "person": person,
                "name": row["candidate_name"],
                "email": row["candidate_email"],
                "role": row["candidate_role"],
                "score": float(score),
                "assigned": bool(score >= threshold),
            }
        )
    return {
        "task": payload.get("task", ""),
        "threshold": threshold,
        "assignments": assignments,
    }


def main(argv: Optional[List[str]] = None) -> int:
    setup_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--model", default=None)
    parser.add_argument("--input", required=True, help="Path to a JSON payload file")
    parser.add_argument("--threshold", type=float, default=None)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    bundle = joblib.load(args.model or cfg["paths"]["model_path"])
    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    result = predict_payload(payload, bundle, args.threshold)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
