"""Parse the raw EPA dataset into a normalized record schema.

The reference distribution we target is the official rehost at

    https://github.com/RevRameshkumar/EPADataset

which ships a single ``EPADataset.tsv`` with two columns – ``Key`` and
``Json``. Each row's ``Json`` payload encodes one (email, marked-task)
HIT and a list of independent annotator judgements. The salient shape is::

    {
      "EmailID": "<id>",
      "Subject": "...",
      "From":          {"emailAddress": {"Name": "...", "Address": "..."}},
      "ToRecipients":  {"emailAddressList": [{"emailAddress": {...}}, ...]},
      "CcRecipients":  {"emailAddressList": [...]},
      "Message":       "... <mark>task sentence</mark> ... <br/> ...",
      "TaskSentence":  "task sentence",
      "Judgements":    [{"<annotator_id>": ["responsible@x", ...]}, ...]
    }

This module is intentionally **format-flexible** so we can also ingest
JSON / JSONL / CSV rehosts that surface the same fields under slightly
different names. Annotator labels are accepted in three shapes:

* per-judge dict ``{annotator_id: [responsible_emails...]}`` – the EPA
  rehost's native shape; an empty list = "no-one";
* per-annotator object with ``responsible_emails`` / ``no_one`` /
  ``sender_responsible`` keys (used by some forks);
* flat ``{email: [v1, v2, ...]}`` recipient-level binary labels.

Aggregation across judges supports two modes mirroring the paper:

* **perfect agreement** – all judges must agree on the exact responsible
  set (matches the paper's "perfect agreement" / α=0.6123 analysis);
* **majority vote** – per-recipient majority across judges, with a
  tie-break that defaults to the first judge.

Returns one normalized record per HIT plus a :class:`ParseStats` summary.
"""
from __future__ import annotations

import csv
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from src.utils.text_cleaning import (
    EMAIL_RE,
    clean_text,
    deduplicate_addresses,
    normalize_email,
    normalize_name,
    strip_mark_tags,
)

logger = logging.getLogger(__name__)


# Canonical field-name candidates. Lowercase keys.
_FIELD_ALIASES: Dict[str, Tuple[str, ...]] = {
    "email_id": ("email_id", "emailid", "id", "hit_id", "hitid", "thread_id"),
    "task_id": ("task_id", "taskid", "marked_task_id", "key"),
    "subject": ("subject", "subj", "title"),
    "body": ("body", "thread", "thread_body", "email_body", "text", "html_body", "message"),
    "task": ("task", "marked_task", "task_sentence", "task_text", "tasksentence"),
    "sender": ("sender", "from", "from_address", "sender_address"),
    "to": ("to", "to_addresses", "to_recipients", "torecipients"),
    "cc": ("cc", "cc_addresses", "cc_recipients", "ccrecipients"),
    "annotations": (
        "annotations",
        "judgments",
        "judgements",
        "labels",
        "annotator_labels",
    ),
}


@dataclass
class ParseStats:
    """Counters that summarize what the parser saw."""

    total_records: int = 0
    parsed_records: int = 0
    skipped_records: int = 0
    skip_reasons: Dict[str, int] = field(default_factory=dict)
    annotator_format_counts: Dict[str, int] = field(default_factory=dict)
    consensus_kept: int = 0
    consensus_dropped: int = 0
    duplicates_removed: int = 0
    no_one_records: int = 0

    def skip(self, reason: str) -> None:
        self.skipped_records += 1
        self.skip_reasons[reason] = self.skip_reasons.get(reason, 0) + 1


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def _load_tsv_two_column(path: Path) -> List[Dict[str, Any]]:
    """Load the ``Key\\tJson`` TSV shipped by the reference EPA repo.

    Each row's ``Json`` payload is parsed and a ``key`` field is added so
    downstream code can always recover the original ``EPA-#####`` HIT id.
    """
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        header = next(reader, None)
        if not header:
            return []
        # Be tolerant about column ordering or alternative names.
        try:
            key_idx = next(
                i for i, c in enumerate(header) if c.strip().lower() in {"key", "id"}
            )
        except StopIteration:
            key_idx = 0
        try:
            json_idx = next(
                i
                for i, c in enumerate(header)
                if c.strip().lower() in {"json", "payload", "data"}
            )
        except StopIteration:
            json_idx = 1 if len(header) >= 2 else 0
        for row in reader:
            if not row or len(row) <= max(key_idx, json_idx):
                continue
            try:
                payload = json.loads(row[json_idx])
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, dict):
                continue
            payload.setdefault("key", row[key_idx])
            out.append(payload)
    return out


def _load_any(path: Path) -> List[Dict[str, Any]]:
    """Load TSV / JSON / JSONL / CSV into a list of dicts.

    Auto-detects format from the file extension and the leading bytes,
    so users can rename files without breaking ingestion.
    """
    suffix = path.suffix.lower()
    if suffix in {".tsv", ".tab"}:
        return _load_tsv_two_column(path)

    text = path.read_text(encoding="utf-8", errors="replace").strip()
    if not text:
        return []

    if suffix == ".csv" or text[0] not in "[{":
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            return [dict(row) for row in reader]

    # JSON or JSONL.
    if text[0] == "[":
        data = json.loads(text)
        if isinstance(data, list):
            return [d for d in data if isinstance(d, dict)]
    if text[0] == "{":
        # Could be a wrapping {"data": [...]} or JSONL with one object/line.
        try:
            first_brace = json.loads(text.splitlines()[0])
        except json.JSONDecodeError:
            first_brace = None
        if isinstance(first_brace, dict) and isinstance(
            first_brace.get("data"), list
        ):
            return list(first_brace["data"])
        # Fall through to JSONL.
    out: List[Dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            out.append(obj)
    return out


def _pick(record: Dict[str, Any], canonical: str) -> Any:
    """Pull a value from ``record`` using any of the known aliases.

    Match is case-insensitive on the key name, so both ``EmailID`` (as
    shipped by the reference TSV) and ``email_id`` work.
    """
    aliases = _FIELD_ALIASES.get(canonical, (canonical,))
    lower = {k.lower(): v for k, v in record.items()}
    for name in aliases:
        if name in lower:
            return lower[name]
    return None


# ---------------------------------------------------------------------------
# Address parsing
# ---------------------------------------------------------------------------


def _coerce_address(value: Any) -> Tuple[str, str]:
    """Convert an "address-ish" value into ``(name, email)``.

    Accepts the shapes seen across rehosts:
        - reference EPA: ``{"emailAddress": {"Name": "...", "Address": "..."}}``
        - simple dict:   ``{"name": "...", "email": "..."}`` /
                          ``{"displayName": "...", "address": "..."}``
        - free text:     ``"Name <addr@x>"`` or a bare email
    """
    if value is None:
        return "", ""
    if isinstance(value, dict):
        # Reference-EPA shape: nested under ``emailAddress``.
        inner = value.get("emailAddress") or value.get("emailaddress")
        if isinstance(inner, dict):
            return _coerce_address(inner)
        name = (
            value.get("Name")
            or value.get("name")
            or value.get("displayName")
            or value.get("display_name")
            or ""
        )
        email = (
            value.get("Address")
            or value.get("address")
            or value.get("email")
            or ""
        )
        return normalize_name(name), normalize_email(email)
    text = str(value).strip()
    if not text:
        return "", ""
    if "<" in text and ">" in text:
        name = text.split("<", 1)[0].strip().strip('"')
        email_match = EMAIL_RE.search(text)
        email = email_match.group(0) if email_match else ""
        return normalize_name(name), normalize_email(email)
    if EMAIL_RE.fullmatch(text):
        return "", normalize_email(text)
    m = EMAIL_RE.search(text)
    if m:
        name = text.replace(m.group(0), "").strip().strip("<>").strip()
        return normalize_name(name), normalize_email(m.group(0))
    return normalize_name(text), ""


def _coerce_address_list(value: Any) -> List[Tuple[str, str]]:
    """Coerce a recipients field to a list of ``(name, email)`` tuples."""
    if value is None or value == "":
        return []
    # Reference-EPA shape: ``{"emailAddressList": [...]}``.
    if isinstance(value, dict) and any(
        k in value for k in ("emailAddressList", "emailaddresslist")
    ):
        inner = value.get("emailAddressList") or value.get("emailaddresslist") or []
        return _coerce_address_list(inner)
    if isinstance(value, list):
        items = [_coerce_address(v) for v in value]
    elif isinstance(value, str):
        # "a@x.com, b@x.com" or "Name <a@x.com>; Name2 <b@x.com>".
        parts = [p.strip() for p in value.replace(";", ",").split(",")]
        items = [_coerce_address(p) for p in parts if p]
    elif isinstance(value, dict):
        items = [_coerce_address(value)]
    else:
        items = []
    items = [(n, e) for n, e in items if n or e]
    return deduplicate_addresses(items)


# ---------------------------------------------------------------------------
# Annotation parsing
# ---------------------------------------------------------------------------


def _judgement_to_email_set(jd: Any) -> Optional[frozenset]:
    """Project one annotator judgement to a frozenset of responsible emails.

    The reference EPA shape for one judge is ``{annotator_id: [emails]}``.
    Some forks instead expose ``{"responsible_emails": [...], "no_one":
    bool, "sender_responsible": bool}``. Both are accepted here. Returns
    ``None`` if the judgement is unintelligible.
    """
    if jd is None:
        return None
    if isinstance(jd, dict):
        # Ref-EPA: single key whose value is the list of responsible emails.
        if len(jd) == 1 and not any(
            k in jd
            for k in (
                "responsible_emails",
                "responsible",
                "no_one",
                "nobody",
                "none",
                "sender_responsible",
            )
        ):
            (only_value,) = jd.values()
            if only_value is None:
                return frozenset()
            if isinstance(only_value, list):
                return frozenset(normalize_email(str(e)) for e in only_value if e)
            if isinstance(only_value, str):
                return frozenset({normalize_email(only_value)}) if only_value else frozenset()
            return None
        # Fork shape with explicit fields.
        if jd.get("no_one") or jd.get("nobody") or jd.get("none"):
            return frozenset()
        emails = (
            jd.get("responsible_emails")
            or jd.get("responsible")
            or jd.get("recipients")
            or []
        )
        out = {normalize_email(str(e)) for e in emails if e}
        if jd.get("sender_responsible") and jd.get("sender_email"):
            out.add(normalize_email(jd["sender_email"]))
        return frozenset(out)
    if isinstance(jd, list):
        return frozenset(normalize_email(str(e)) for e in jd if e)
    return None


def _resolve_responsible(
    annotations: Any,
    candidate_emails: List[str],
    sender_email: str,
    consensus: str,
    stats: ParseStats,
) -> Tuple[Optional[List[str]], bool]:
    """Resolve annotator votes into a final set of responsible emails.

    ``consensus`` is one of:
        - ``"perfect"`` – keep only HITs where every judge agrees on the
          exact responsible set; otherwise return ``(None, False)`` so
          the caller can drop the row.
        - ``"majority"`` – per-recipient majority vote across judges.
        - ``"any"``      – any judge that marked the recipient counts.

    Returns ``(responsible_emails_or_None, no_one)``. ``responsible_emails``
    is sorted; ``no_one`` is true when the resolved set is empty.
    """
    if annotations is None:
        stats.annotator_format_counts["missing"] = (
            stats.annotator_format_counts.get("missing", 0) + 1
        )
        return [], True

    # Recipient-level binary labels: ``{email: [v1, v2, v3]}``.
    if isinstance(annotations, dict) and not (
        len(annotations) == 1 and isinstance(next(iter(annotations.values())), list)
    ) and all(isinstance(v, (list, tuple)) for v in annotations.values()):
        stats.annotator_format_counts["per_recipient_dict"] = (
            stats.annotator_format_counts.get("per_recipient_dict", 0) + 1
        )
        responsible: List[str] = []
        for email_key, votes in annotations.items():
            ones = sum(1 for v in votes if int(v) == 1)
            n = len(votes)
            need = n if consensus == "perfect" else (1 if consensus == "any" else (n // 2 + 1))
            if ones >= need:
                responsible.append(normalize_email(str(email_key)))
        return sorted(set(responsible)), len(responsible) == 0

    # Otherwise we expect a list of judgements.
    if isinstance(annotations, list):
        stats.annotator_format_counts["per_judge_list"] = (
            stats.annotator_format_counts.get("per_judge_list", 0) + 1
        )
        judge_sets: List[frozenset] = []
        for jd in annotations:
            s = _judgement_to_email_set(jd)
            if s is not None:
                judge_sets.append(s)
        if not judge_sets:
            return [], True

        if consensus == "perfect":
            if all(s == judge_sets[0] for s in judge_sets):
                resp = sorted(judge_sets[0])
                return resp, len(resp) == 0
            return None, False  # drop signal

        # Vote per email mentioned by any judge plus all candidates.
        universe = set(candidate_emails) | set().union(*judge_sets)
        n = len(judge_sets)
        need = 1 if consensus == "any" else (n // 2 + 1)
        responsible = sorted(
            e for e in universe if sum(1 for s in judge_sets if e in s) >= need
        )
        return responsible, len(responsible) == 0

    stats.annotator_format_counts["unknown"] = (
        stats.annotator_format_counts.get("unknown", 0) + 1
    )
    return [], True


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_MARK_RE = re.compile(r"<mark[^>]*>(.*?)</mark>", re.IGNORECASE | re.DOTALL)


def _extract_marked_task(body_raw: str) -> str:
    """Pull the ``<mark>...</mark>`` task sentence from a raw email body."""
    if not body_raw:
        return ""
    m = _MARK_RE.search(str(body_raw))
    return clean_text(m.group(1)) if m else ""


def parse_dataset(
    path: str | Path,
    consensus: str = "majority",
) -> Tuple[List[Dict[str, Any]], ParseStats]:
    """Parse a raw EPA dataset file into normalized records.

    ``consensus`` is forwarded to :func:`_resolve_responsible`. Pass
    ``"perfect"`` to drop HITs where annotators disagreed (matches the
    paper's strictest evaluation slice), ``"majority"`` (default) to keep
    everything via per-recipient majority vote, or ``"any"`` to maximise
    recall during exploration.

    Returns a list of dicts with this schema::

        {
            "email_id": str,
            "task_id": str,
            "subject": str,
            "body": str,                 # cleaned, with <mark> kept inline
            "task": str,                 # cleaned, no <mark> tags
            "sender": (name, email),
            "to":  [(name, email), ...],
            "cc":  [(name, email), ...],
            "responsible_emails": [email, ...],
            "no_one_responsible": bool,
            "n_judges": int,
            "perfect_agreement": bool,
        }
    """
    path = Path(path)
    if consensus not in {"perfect", "majority", "any"}:
        raise ValueError(f"Unknown consensus mode: {consensus!r}")

    raw_records = _load_any(path)
    stats = ParseStats(total_records=len(raw_records))
    out: List[Dict[str, Any]] = []

    for idx, record in enumerate(raw_records):
        try:
            sender_pair = _coerce_address(_pick(record, "sender"))
            to_list = _coerce_address_list(_pick(record, "to"))
            cc_list = _coerce_address_list(_pick(record, "cc"))

            body_raw = _pick(record, "body") or ""
            task_raw = _pick(record, "task") or ""
            body_clean = clean_text(body_raw)
            task_clean = strip_mark_tags(clean_text(task_raw)) if task_raw else ""

            # Recover task from the in-line <mark> if a separate field is missing.
            if not task_clean:
                task_clean = _extract_marked_task(body_raw)

            email_id = str(_pick(record, "email_id") or f"E{idx:06d}")
            task_id = str(record.get("key") or _pick(record, "task_id") or f"{email_id}_T0")
            subject = clean_text(_pick(record, "subject") or "")

            candidates: List[Tuple[str, str]] = []
            if sender_pair[1] or sender_pair[0]:
                candidates.append(sender_pair)
            candidates.extend(to_list)
            candidates.extend(cc_list)
            candidates = deduplicate_addresses(candidates)
            if not candidates:
                stats.skip("no_candidates")
                continue
            candidate_emails = [e for _, e in candidates if e]

            annotations = _pick(record, "annotations")
            responsible, no_one = _resolve_responsible(
                annotations,
                candidate_emails,
                sender_pair[1],
                consensus,
                stats,
            )
            if responsible is None:
                # Perfect-agreement requested and judges disagreed.
                stats.consensus_dropped += 1
                continue
            stats.consensus_kept += 1

            # Compute a perfect-agreement flag regardless of mode (handy at
            # eval time when slicing to the paper's strict subset).
            perfect = False
            if isinstance(annotations, list):
                judge_sets = [
                    _judgement_to_email_set(j)
                    for j in annotations
                    if _judgement_to_email_set(j) is not None
                ]
                perfect = bool(judge_sets) and all(
                    s == judge_sets[0] for s in judge_sets
                )
            n_judges = len(annotations) if isinstance(annotations, list) else 0
            if no_one:
                stats.no_one_records += 1

            out.append(
                {
                    "email_id": email_id,
                    "task_id": task_id,
                    "subject": subject,
                    "body": body_clean,
                    "task": task_clean,
                    "sender": sender_pair,
                    "to": to_list,
                    "cc": cc_list,
                    "responsible_emails": responsible,
                    "no_one_responsible": no_one,
                    "n_judges": n_judges,
                    "perfect_agreement": perfect,
                }
            )
            stats.parsed_records += 1
        except Exception as e:  # noqa: BLE001 – tolerant per-record failure
            logger.warning("Failed to parse record %d: %s", idx, e)
            stats.skip("parse_error")
            continue

    logger.info(
        "Parsed %d/%d records (%d skipped, %d dropped for non-consensus). "
        "Annotator formats: %s",
        stats.parsed_records,
        stats.total_records,
        stats.skipped_records,
        stats.consensus_dropped,
        stats.annotator_format_counts,
    )
    return out, stats


def parse_directory(
    raw_dir: str | Path,
    consensus: str = "majority",
) -> Tuple[List[Dict[str, Any]], ParseStats]:
    """Parse every TSV/JSON/JSONL/CSV under ``raw_dir`` and concatenate.

    Files starting with ``_`` (e.g. ``_synthetic.json``) and any file named
    like a source-README are ignored so the raw dir can host non-data
    artefacts safely.
    """
    raw_dir = Path(raw_dir)
    candidates = (
        list(raw_dir.glob("*.tsv"))
        + list(raw_dir.glob("*.tab"))
        + list(raw_dir.glob("*.json"))
        + list(raw_dir.glob("*.jsonl"))
        + list(raw_dir.glob("*.csv"))
    )
    def _is_data_file(p: Path) -> bool:
        nm = p.name.lower()
        if p.name.startswith("_"):
            return False
        if "source_readme" in nm:
            return False
        # Skip the legacy synthetic dataset emitted by older download.py runs
        # if the user has the real EPADataset.tsv alongside it; the new
        # synthetic file uses a leading underscore so it's already excluded.
        if nm.startswith("epa_synthetic") and any(
            c.name.lower().startswith("epadataset") for c in candidates
        ):
            return False
        return True

    files = sorted(f for f in candidates if _is_data_file(f))
    if not files:
        raise FileNotFoundError(f"No EPA-shaped files under {raw_dir}")
    all_records: List[Dict[str, Any]] = []
    aggregate = ParseStats()
    for f in files:
        logger.info("Parsing %s", f)
        records, stats = parse_dataset(f, consensus=consensus)
        all_records.extend(records)
        aggregate.total_records += stats.total_records
        aggregate.parsed_records += stats.parsed_records
        aggregate.skipped_records += stats.skipped_records
        aggregate.consensus_kept += stats.consensus_kept
        aggregate.consensus_dropped += stats.consensus_dropped
        aggregate.no_one_records += stats.no_one_records
        for k, v in stats.skip_reasons.items():
            aggregate.skip_reasons[k] = aggregate.skip_reasons.get(k, 0) + v
        for k, v in stats.annotator_format_counts.items():
            aggregate.annotator_format_counts[k] = (
                aggregate.annotator_format_counts.get(k, 0) + v
            )
    return all_records, aggregate
