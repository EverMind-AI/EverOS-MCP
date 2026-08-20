"""Pre-write credential guard.

Long-term cloud memory is a terrible place for secrets: they outlive the
conversation, replicate into search indexes, and surface in future prompts.
Every write path scans for high-confidence credential formats and refuses the
write — there is deliberately no bypass flag. Store a reference instead.
"""

from __future__ import annotations

import re

_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("API key", re.compile(r"\bsk-[A-Za-z0-9_-]{16,}")),
    ("AWS access key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("GitHub token", re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})")),
    ("Slack token", re.compile(r"\bxox[bpoas]-[A-Za-z0-9-]{10,}")),
    ("Google API key", re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b")),
    ("private key", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("JWT", re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b")),
    ("URL with embedded password", re.compile(r"\b[a-z][a-z0-9+.-]*://[^/\s:@]{1,64}:[^@\s]{1,256}@")),
]


def find_secret(text: str) -> str | None:
    """Return a human-readable finding like 'API key (sk-proj…Q4x)', or None."""
    for kind, pattern in _PATTERNS:
        m = pattern.search(text)
        if m:
            token = m.group(0)
            masked = token[:7] + "…" + token[-3:] if len(token) > 13 else token[:4] + "…"
            return f"{kind} ({masked})"
    return None


def refusal(finding: str) -> str:
    return (
        f"Blocked: the content appears to contain a credential — {finding}. "
        "Long-term memory is not a safe place for secrets; they would persist "
        "and resurface in future sessions. Store a reference instead (e.g. "
        "'the deploy key lives in 1Password under X') or redact the value and retry."
    )
