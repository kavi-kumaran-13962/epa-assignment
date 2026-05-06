"""Acquire the EPA dataset.

The reference distribution is hosted at::

    https://github.com/RevRameshkumar/EPADataset

This module tries the following strategies in order:

1. ``--url`` (or ``dataset.url`` in ``config.yaml``) – an explicit hosted
   copy of ``EPADataset.tsv`` (or any JSON/JSONL/CSV rehost).
2. ``git clone`` of ``dataset.git_repo`` (defaults to the reference repo
   above), copying the data file out of the working tree. Works in the
   common case where ``raw.githubusercontent.com`` egress is blocked but
   ``github.com`` git is allowed.
3. Synthetic fallback – generates a small EPA-shaped dataset under
   ``data/raw`` so the rest of the pipeline remains runnable end-to-end
   for code review even when network access is restricted.

Run::

    python -m src.data.download                 # try real dataset, fall back
    python -m src.data.download --force_synthetic
    python -m src.data.download --url <hosted-tsv-or-json>
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

from src.utils.io import ensure_dir, load_config, setup_logging

logger = logging.getLogger(__name__)

SYNTHETIC_FILENAME = "_epa_synthetic.json"
DEFAULT_GIT_REPO = "https://github.com/RevRameshkumar/EPADataset.git"
DEFAULT_DATA_FILENAME = "EPADataset.tsv"


def _try_download_url(url: str, dest: Path, timeout: int = 60) -> bool:
    """Best-effort HTTP download. Returns True on success."""
    try:
        logger.info("Fetching dataset from %s", url)
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
    except Exception as e:  # noqa: BLE001 – network failures are expected
        logger.warning("URL download failed (%s).", e)
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(resp.content)
    logger.info("Saved %d bytes to %s", len(resp.content), dest)
    return True


def _try_git_clone(
    git_repo: str, data_filename: str, dest_dir: Path
) -> Optional[Path]:
    """Clone ``git_repo`` shallowly and copy ``data_filename`` into ``dest_dir``.

    Returns the destination path on success, ``None`` otherwise. We keep
    a shallow ``--depth 1`` clone in a temp directory so we never leave a
    second working tree on disk.
    """
    if shutil.which("git") is None:
        logger.warning("git is not on PATH; cannot clone %s", git_repo)
        return None
    with tempfile.TemporaryDirectory(prefix="epa_clone_") as tmpdir:
        try:
            logger.info("Cloning %s (shallow) ...", git_repo)
            subprocess.run(
                ["git", "clone", "--depth", "1", git_repo, tmpdir + "/repo"],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=120,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            stderr = (
                e.stderr.decode(errors="replace") if hasattr(e, "stderr") and e.stderr else ""
            )
            logger.warning("git clone failed: %s\n%s", e, stderr)
            return None
        src = Path(tmpdir) / "repo" / data_filename
        if not src.exists():
            logger.warning(
                "Cloned repo does not contain %s; available files: %s",
                data_filename,
                [p.name for p in (Path(tmpdir) / "repo").iterdir()],
            )
            return None
        dest_dir.mkdir(parents=True, exist_ok=True)
        out = dest_dir / data_filename
        shutil.copy2(src, out)
        # Bring the source README along for traceability if present.
        readme = Path(tmpdir) / "repo" / "README.md"
        if readme.exists():
            shutil.copy2(readme, dest_dir / "EPADataset_source_README.md")
        logger.info("Cloned and copied %s -> %s", data_filename, out)
        return out


# ---------------------------------------------------------------------------
# Synthetic fallback
# ---------------------------------------------------------------------------


def _synthetic_records(n: int = 60, seed: int = 13) -> List[Dict[str, Any]]:
    """Generate a small EPA-shaped dataset for smoke tests.

    Each record follows the reference EPA payload schema so the *real*
    parser path is exercised end-to-end. We deliberately include cases
    that exercise the harder branches of the parser:

    * explicit name in task ("Hi Anna")
    * implicit ``you`` with multiple recipients
    * "no-one" responsible
    * sender is responsible
    * group alias as a recipient
    * thread context with original-message header
    """
    rng = random.Random(seed)

    people = [
        ("Anna Smith", "anna@example.com"),
        ("Brad Jones", "brad@example.com"),
        ("Caira Wong", "caira@example.com"),
        ("John Patel", "john@example.com"),
        ("Maya Lopez", "maya@example.com"),
        ("Priya Rao", "priya@example.com"),
        ("Quinn Park", "quinn@example.com"),
        ("Sam Diaz", "sam@example.com"),
    ]
    templates = [
        (
            "Can you and {p1} complete a draft by Friday please.",
            "Hi {p0}, thanks for your work last week. <mark>Can you and {p1} complete a draft by Friday please.</mark> Thanks, {sender_name}",
            lambda recs, sender: [recs[0], recs[1]] if len(recs) >= 2 else [recs[0]],
        ),
        (
            "Please send me the latest numbers.",
            "Hi {p0}, <mark>Please send me the latest numbers.</mark> Best, {sender_name}",
            lambda recs, sender: [recs[0]],
        ),
        (
            "I'll prepare the deck for tomorrow.",
            "Team, <mark>I'll prepare the deck for tomorrow.</mark> Sending by 9am.",
            lambda recs, sender: [sender],
        ),
        (
            "Please complete a draft by Friday.",
            "Hi all, <mark>Please complete a draft by Friday.</mark> Thanks.",
            lambda recs, sender: list(recs),
        ),
        (
            "Brad will complete the draft report.",
            "FYI – <mark>Brad will complete the draft report.</mark> Coordinating offline.",
            lambda recs, sender: [],  # referenced but not on To/Cc
        ),
        (
            "Could you review the contract?",
            "Hi {p0}, hope you're well. <mark>Could you review the contract?</mark> Thanks.",
            lambda recs, sender: [recs[0]],
        ),
    ]
    subjects = [
        "Draft report",
        "Q3 numbers",
        "Tomorrow's deck",
        "Contract review",
        "Quick favor",
        "Status update",
    ]

    records: List[Dict[str, Any]] = []
    for i in range(n):
        sender = rng.choice(people)
        n_to = rng.randint(1, 3)
        n_cc = rng.randint(0, 2)
        pool = [p for p in people if p != sender]
        rng.shuffle(pool)
        to_list = pool[:n_to]
        cc_list = pool[n_to : n_to + n_cc]
        recipients = to_list + cc_list

        if rng.random() < 0.15:
            cc_list.append(("Sales Team", "sales@example.com"))

        task_template, body_template, picker = rng.choice(templates)
        p0 = recipients[0][0].split()[0] if recipients else "team"
        p1 = recipients[1][0].split()[0] if len(recipients) >= 2 else "the team"
        body = body_template.format(
            p0=p0, p1=p1, sender_name=sender[0].split()[0]
        )
        if rng.random() < 0.3:
            body += (
                "\n\n------ Original Message ------\nFrom: "
                + (
                    f"{recipients[0][0]} <{recipients[0][1]}>"
                    if recipients
                    else "someone"
                )
                + f"\nSubject: Re: {rng.choice(subjects)}\n\nThanks for the context."
            )

        responsible_pairs = picker(recipients, sender)

        # Mirror the reference EPA payload format exactly so the real
        # parser path is exercised even on synthetic data.
        record = {
            "EmailID": f"E{i:04d}",
            "Subject": rng.choice(subjects),
            "From": {"emailAddress": {"Name": sender[0], "Address": sender[1]}},
            "ToRecipients": {
                "emailAddressList": [
                    {"emailAddress": {"Name": n, "Address": e}} for n, e in to_list
                ]
            },
            "CcRecipients": {
                "emailAddressList": [
                    {"emailAddress": {"Name": n, "Address": e}} for n, e in cc_list
                ]
            },
            "Message": body,
            "TaskSentence": task_template.format(p1=p1),
            "Judgements": [
                {f"judge_{j}": [e for _, e in responsible_pairs]} for j in range(3)
            ],
        }
        records.append(record)
    return records


def write_synthetic(raw_dir: Path) -> Path:
    """Materialize the synthetic dataset under ``raw_dir``."""
    out = raw_dir / SYNTHETIC_FILENAME
    out.parent.mkdir(parents=True, exist_ok=True)
    records = _synthetic_records()
    with open(out, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    logger.info("Wrote %d synthetic records to %s", len(records), out)
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def acquire_dataset(cfg: Dict[str, Any], raw_dir: Path,
                    url_override: Optional[str] = None,
                    force_synthetic: bool = False) -> Path:
    """End-to-end: try URL → git clone → synthetic. Returns the file path.

    Idempotent: if the data file is already present and non-empty under
    ``raw_dir``, we skip re-downloading and return that path.
    """
    data_filename = cfg.get("dataset", {}).get(
        "data_filename", DEFAULT_DATA_FILENAME
    )
    existing = raw_dir / data_filename
    if existing.exists() and existing.stat().st_size > 0 and not force_synthetic:
        logger.info("Found existing dataset at %s; skipping download.", existing)
        return existing

    if force_synthetic:
        return write_synthetic(raw_dir)

    url = url_override or cfg.get("dataset", {}).get("url")
    if url:
        if _try_download_url(url, existing):
            return existing

    git_repo = cfg.get("dataset", {}).get("git_repo", DEFAULT_GIT_REPO)
    if git_repo:
        cloned = _try_git_clone(git_repo, data_filename, raw_dir)
        if cloned is not None:
            return cloned

    logger.warning(
        "Could not acquire the real EPA dataset; generating synthetic data so "
        "the pipeline remains runnable end-to-end. Configure dataset.url or "
        "dataset.git_repo to use real EPA data."
    )
    return write_synthetic(raw_dir)


def main(argv: Optional[List[str]] = None) -> int:
    setup_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--url", default=None, help="Override dataset URL")
    parser.add_argument("--raw_dir", default=None, help="Override raw dir")
    parser.add_argument(
        "--force_synthetic",
        action="store_true",
        help="Skip URL/git and generate synthetic data (useful for CI).",
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    raw_dir = ensure_dir(args.raw_dir or cfg["paths"]["raw_dir"])
    acquire_dataset(
        cfg,
        Path(raw_dir),
        url_override=args.url,
        force_synthetic=args.force_synthetic,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
