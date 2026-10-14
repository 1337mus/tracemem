"""Heuristic privacy indicators for local transcript inspection.

Matches are signals for human review, not proof that a value is secret or safe.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Finding:
    """A privacy indicator location without retaining matched source text."""

    category: str
    char_start: int
    char_end: int


_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "github_token",
        re.compile(
            r"(?<![A-Za-z0-9_])(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})(?![A-Za-z0-9_])"
        ),
    ),
    (
        "openai_token",
        re.compile(
            r"(?<![A-Za-z0-9_])sk-(?:proj|svcacct|admin)-[A-Za-z0-9_-]{20,}(?![A-Za-z0-9_-])"
        ),
    ),
    (
        "aws_access_key_id",
        re.compile(r"(?<![A-Za-z0-9])(?:AKIA|ASIA)[0-9A-Z]{16}(?![A-Za-z0-9])"),
    ),
    (
        "bearer_token",
        re.compile(r"\bBearer\s+[A-Za-z0-9._~+/-]{15,}[A-Za-z0-9+/_-](?:={1,2})?", re.IGNORECASE),
    ),
    (
        "pem_private_key",
        re.compile(
            r"-----BEGIN (?P<label>(?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY)-----"
            r"[\s\S]*?-----END (?P=label)-----"
        ),
    ),
    (
        "email",
        re.compile(
            r"(?<![A-Za-z0-9.!#$%&'*+/=?^_`{|}~-])"
            r"[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@"
            r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?"
            r"(?:\.[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?)+"
        ),
    ),
)


def scan(text: str) -> list[Finding]:
    """Return sorted indicator categories and character ranges for ``text``."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    findings = {
        Finding(category, match.start(), match.end())
        for category, pattern in _PATTERNS
        for match in pattern.finditer(text)
    }
    return sorted(
        findings, key=lambda finding: (finding.char_start, finding.char_end, finding.category)
    )


def redaction_preview(text: str) -> str:
    """Return a preview with detected ranges replaced, preserving other text."""
    findings = scan(text)
    if not findings:
        return text

    ranges: list[tuple[int, int]] = []
    for finding in findings:
        if ranges and finding.char_start <= ranges[-1][1]:
            previous_start, previous_end = ranges[-1]
            ranges[-1] = (previous_start, max(previous_end, finding.char_end))
        else:
            ranges.append((finding.char_start, finding.char_end))

    preview = text
    for char_start, char_end in reversed(ranges):
        preview = preview[:char_start] + "[REDACTED]" + preview[char_end:]
    return preview
