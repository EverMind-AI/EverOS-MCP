"""Pre-write credential guard.

Long-term cloud memory is a terrible place for secrets: they outlive the
conversation, replicate into search indexes, and surface in future prompts.
Every write path scans for high-confidence credential formats and refuses the
write — there is deliberately no bypass flag. Store a reference instead.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from typing import Any

_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("API key", re.compile(r"\bsk-[A-Za-z0-9_-]{16,}")),
    ("Stripe key", re.compile(r"\b[sr]k_(?:live|test)_[A-Za-z0-9]{16,}")),
    ("AWS access key", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("GitHub token", re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})")),
    ("Slack token", re.compile(r"\bxox[bpoas]-[A-Za-z0-9-]{10,}")),
    ("Google API key", re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b")),
    ("private key", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("JWT", re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b")),
    ("bearer token", re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]{20,}")),
    (
        "URL with embedded password",
        re.compile(r"\b[a-z][a-z0-9+.-]*://[^/\s:@]{1,64}:[^@\s]{1,256}@"),
    ),
    # `password=hunter2x`, `AWS_SECRET_ACCESS_KEY: "..."`, `"api_key": "..."`.
    # The value must mix letters and digits, so prose ("password: the one in
    # 1Password") and env references (`$DB_PASSWORD`, `${TOKEN}`) pass.
    (
        "secret assignment",
        re.compile(
            r"(?i)\b[a-z0-9_]*(?:password|passwd|secret|api_?key|access_?key"
            r"|(?:access|auth|refresh|session)_?token)[a-z0-9_]*[\"']?\s*[:=]\s*[\"']?"
            r"(?=[^\s\"'$]*\d)(?=[^\s\"'$]*[A-Za-z])[^\s\"'${}<>]{8,}"
        ),
    ),
]


def find_secret(text: str, patterns: list[tuple[str, re.Pattern[str]]] = _PATTERNS) -> str | None:
    """Return a human-readable finding like 'API key (sk-proj…Q4x)', or None."""
    for kind, pattern in patterns:
        m = pattern.search(text)
        if m:
            token = m.group(0)
            masked = token[:7] + "…" + token[-3:] if len(token) > 13 else token[:4] + "…"
            return f"{kind} ({masked})"
    return None


_ASSIGNMENT = [p for p in _PATTERNS if p[0] == "secret assignment"]


def _strings(value: Any) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for k, v in value.items():
            yield str(k)
            yield from _strings(v)
    elif isinstance(value, (list, tuple)):
        for v in value:
            yield from _strings(v)


def find_secret_in(value: Any) -> str | None:
    """Scan every string nested anywhere in `value` — message content, tool-call
    arguments, tool results — not just the top-level text.

    Two passes: every pattern over the strings alone (catches values inside
    JSON-encoded strings), then the assignment pattern alone over the
    structure serialized as JSON, which keeps a key next to its value so
    `{"password": "..."}` reads as an assignment. Only that pattern: JSON
    escapes newlines into literal `\\n`, which would let the others match
    across what were separate lines."""
    return find_secret("\n".join(_strings(value))) or find_secret(
        json.dumps(value, ensure_ascii=False, default=str), _ASSIGNMENT
    )


def refusal(finding: str) -> str:
    return (
        f"Blocked: the content appears to contain a credential — {finding}. "
        "Long-term memory is not a safe place for secrets; they would persist "
        "and resurface in future sessions. Store a reference instead (e.g. "
        "'the deploy key lives in 1Password under X') or redact the value and retry."
    )
