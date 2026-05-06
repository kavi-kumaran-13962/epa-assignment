"""Lightweight text/email normalization helpers.

We deliberately keep cleaning *minimal*:
- the EPA paper notes that aggressive normalization removes useful signal
  (sentence boundaries, addressee cues, original casing of names, etc.),
- so we only fix things that are clearly noise: HTML break tags, tabs,
  collapsing extreme whitespace.
"""
from __future__ import annotations

import re
from typing import Iterable, List, Optional, Tuple

# Match an email address. Permissive on purpose - email addresses in the
# Enron corpus include unusual local parts, group aliases, etc.
EMAIL_RE = re.compile(r"[\w.+\-]+@[\w\-.]+\.[A-Za-z]{2,}", re.IGNORECASE)
# Crude but effective HTML break/paragraph stripping.
BR_RE = re.compile(r"<\s*br\s*/?\s*>", re.IGNORECASE)
TAG_RE = re.compile(r"<[^>]+>")
WS_RE = re.compile(r"[ \t]+")
NL_RE = re.compile(r"\n{3,}")


def clean_text(text: Optional[str]) -> str:
    """Apply minimal cleaning to email text.

    - Replace ``<br/>`` with newlines (per paper's preprocessing).
    - Convert tabs to spaces.
    - Collapse runs of spaces and excessive blank lines.
    - Strip leading/trailing whitespace.
    """
    if text is None:
        return ""
    s = str(text)
    s = BR_RE.sub("\n", s)
    s = s.replace("\t", " ")
    s = WS_RE.sub(" ", s)
    s = NL_RE.sub("\n\n", s)
    return s.strip()


def strip_mark_tags(text: Optional[str]) -> str:
    """Remove ``<mark>`` task-marker tags but keep the inner text."""
    if text is None:
        return ""
    return re.sub(r"</?mark[^>]*>", "", str(text), flags=re.IGNORECASE)


def normalize_email(addr: Optional[str]) -> str:
    """Lowercase and strip an email address. Return ``''`` on missing."""
    if not addr:
        return ""
    addr = str(addr).strip().lower()
    # Pull the address out of "Name <addr@x>" if needed.
    m = EMAIL_RE.search(addr)
    return m.group(0) if m else addr


def normalize_name(name: Optional[str]) -> str:
    """Light name cleanup – strip quotes and extra whitespace."""
    if not name:
        return ""
    name = str(name).strip().strip('"').strip("'")
    return WS_RE.sub(" ", name).strip()


def split_full_name(name: str) -> Tuple[str, str]:
    """Best-effort first/last split. Returns ``('', '')`` for empty input."""
    if not name:
        return "", ""
    parts = [p for p in re.split(r"\s+", name.strip()) if p]
    if not parts:
        return "", ""
    if len(parts) == 1:
        return parts[0], ""
    return parts[0], parts[-1]


def email_local_part(addr: str) -> str:
    """Return the part of an email before the ``@`` sign."""
    addr = normalize_email(addr)
    return addr.split("@", 1)[0] if "@" in addr else addr


def email_domain(addr: str) -> str:
    addr = normalize_email(addr)
    return addr.split("@", 1)[1] if "@" in addr else ""


def deduplicate_addresses(items: Iterable[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """Deduplicate (name, email) pairs by lowercased email.

    The first occurrence's name is preferred so we keep more informative names
    when later occurrences are anonymous.
    """
    seen: set = set()
    out: List[Tuple[str, str]] = []
    for name, addr in items:
        key = normalize_email(addr) or normalize_name(name).lower()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append((normalize_name(name), normalize_email(addr)))
    return out


def tokenize_basic(text: str) -> List[str]:
    """Lightweight whitespace+punct tokenizer used for length features."""
    if not text:
        return []
    return re.findall(r"[A-Za-z0-9_']+", text.lower())
