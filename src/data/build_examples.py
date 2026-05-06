"""Convert parsed EPA records into a candidate-level training dataframe.

Per the W-NUT 2018 paper, the EPA challenge is reduced to a series of
binary decisions over (email, task, candidate person) tuples. This module
materializes that view: one row per candidate, with structured role flags
and a label.

Usage::

    python -m src.data.build_examples \
        --raw_dir data/raw \
        --output data/processed/examples.csv \
        --consensus majority      # one of perfect | majority | any
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

from src.data.parse_epa import parse_directory
from src.utils.io import ensure_dir, load_config, setup_logging

logger = logging.getLogger(__name__)


# Columns we always emit so downstream code can rely on a stable schema.
COLUMNS: List[str] = [
    "example_id",
    "email_id",
    "task_id",
    "email_task_id",
    "sender_email",
    "sender_name",
    "candidate_email",
    "candidate_name",
    "candidate_role",        # sender|to|cc|multiple
    "is_sender",
    "is_to",
    "is_cc",
    "to_emails",
    "cc_emails",
    "all_candidate_emails",
    "num_to_recipients",
    "num_cc_recipients",
    "num_total_candidates",
    "subject",
    "body",
    "task",
    "full_context",
    "no_one_responsible",
    "n_judges",
    "perfect_agreement",
    "label",
]


def _candidate_rows_for_record(record: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    """Yield one candidate-level row per (email, task, candidate person).

    Implementation notes:

    * Candidates are deduplicated by lowercased email; if the same person
      shows up in multiple roles (e.g. To and Cc) we keep one row with
      ``candidate_role = "multiple"`` and set every relevant role flag.
    * The sender is included as a candidate; the paper's annotation spec
      explicitly allows the sender to be marked responsible.
    * ``label = 1`` only if a candidate's email is in the resolved
      ``responsible_emails`` set. ``no_one_responsible`` HITs produce all
      zero rows by construction.
    """
    sender_name, sender_email = record.get("sender", ("", ""))
    to = record.get("to", []) or []
    cc = record.get("cc", []) or []
    responsible = {e.lower() for e in (record.get("responsible_emails") or []) if e}
    no_one = bool(record.get("no_one_responsible"))

    by_email: Dict[str, Dict[str, Any]] = {}

    def _register(name: str, email: str, role: str) -> None:
        key = email or f"name:{name.lower()}"
        if not key:
            return
        slot = by_email.setdefault(
            key, {"name": name, "email": email, "roles": set()}
        )
        if name and not slot["name"]:
            slot["name"] = name
        slot["roles"].add(role)

    _register(sender_name, sender_email, "sender")
    for n, e in to:
        _register(n, e, "to")
    for n, e in cc:
        _register(n, e, "cc")

    if not by_email:
        return

    to_emails = [e for _, e in to if e]
    cc_emails = [e for _, e in cc if e]
    all_candidate_emails = sorted(by_email.keys())
    num_total = len(by_email)

    full_context = " [SEP] ".join(
        x
        for x in [
            record.get("subject", ""),
            record.get("task", ""),
            record.get("body", ""),
        ]
        if x
    )

    email_id = record["email_id"]
    task_id = record["task_id"]
    email_task_id = f"{email_id}::{task_id}"

    for idx, (key, slot) in enumerate(sorted(by_email.items())):
        roles = slot["roles"]
        cand_email = slot["email"]
        if len(roles) > 1:
            role = "multiple"
        else:
            role = next(iter(roles))

        label = 0
        if not no_one and cand_email and cand_email.lower() in responsible:
            label = 1

        yield {
            "example_id": f"{email_task_id}::{idx}",
            "email_id": email_id,
            "task_id": task_id,
            "email_task_id": email_task_id,
            "sender_email": sender_email,
            "sender_name": sender_name,
            "candidate_email": cand_email,
            "candidate_name": slot["name"],
            "candidate_role": role,
            "is_sender": int("sender" in roles),
            "is_to": int("to" in roles),
            "is_cc": int("cc" in roles),
            "to_emails": ";".join(to_emails),
            "cc_emails": ";".join(cc_emails),
            "all_candidate_emails": ";".join(all_candidate_emails),
            "num_to_recipients": len(to_emails),
            "num_cc_recipients": len(cc_emails),
            "num_total_candidates": num_total,
            "subject": record.get("subject", ""),
            "body": record.get("body", ""),
            "task": record.get("task", ""),
            "full_context": full_context,
            "no_one_responsible": int(no_one),
            "n_judges": int(record.get("n_judges", 0) or 0),
            "perfect_agreement": int(bool(record.get("perfect_agreement", False))),
            "label": label,
        }


def build_examples(records: Iterable[Dict[str, Any]]) -> pd.DataFrame:
    """Materialize a candidate-level dataframe from parsed records."""
    rows: List[Dict[str, Any]] = []
    for r in records:
        rows.extend(_candidate_rows_for_record(r))
    if not rows:
        return pd.DataFrame(columns=COLUMNS)
    df = pd.DataFrame(rows)
    for col in COLUMNS:
        if col not in df.columns:
            df[col] = ""
    return df[COLUMNS]


def main(argv: Optional[List[str]] = None) -> int:
    setup_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--raw_dir", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--consensus",
        choices=("perfect", "majority", "any"),
        default=None,
        help="How to aggregate annotator votes (overrides config).",
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    raw_dir = Path(args.raw_dir or cfg["paths"]["raw_dir"])
    output = Path(args.output or cfg["paths"]["examples_csv"])
    consensus = args.consensus or cfg["dataset"].get("consensus", "majority")

    records, stats = parse_directory(raw_dir, consensus=consensus)
    df = build_examples(records)
    ensure_dir(output.parent)
    df.to_csv(output, index=False)
    logger.info(
        "Wrote %d candidate rows (%d positive, %.2f%% positive rate) to %s",
        len(df),
        int(df["label"].sum()) if len(df) else 0,
        100.0 * df["label"].mean() if len(df) else 0.0,
        output,
    )
    logger.info(
        "Parse stats: total=%d, parsed=%d, dropped_for_consensus=%d, "
        "no_one_records=%d, formats=%s",
        stats.total_records,
        stats.parsed_records,
        stats.consensus_dropped,
        stats.no_one_records,
        stats.annotator_format_counts,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
