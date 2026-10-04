"""Environment-based configuration.

Identity and scope are server-level configuration, not tool parameters: the
LLM should never choose whose memory it is reading or writing.
"""

from __future__ import annotations

import os
import re
import uuid
from dataclasses import dataclass
from urllib.parse import urlparse

PROD_BASE_URL = "https://api.evermind.ai"

# The id charset both the cloud and the self-hosted server accept (ids become
# path segments on self-hosted EverOS). EverOS caps session and sender ids at
# 128 characters, and the ids derived from a user id add up to 17
# ("mcp-" + "-" + 12 hex), so a user id stops at 100.
MAX_USER_ID = 100
_ID_RE = re.compile(rf"[A-Za-z0-9_.@+-]{{1,{MAX_USER_ID}}}")

# Owner of the memories when no user id is configured: one memory per API key,
# the same on every machine (as in EverOS's Claude Code plugin).
DEFAULT_USER_ID = "default-user"


class ConfigError(Exception):
    pass


@dataclass(frozen=True)
class Settings:
    api_key: str
    base_url: str
    user_id: str
    app_id: str
    project_id: str
    session_id: str
    assistant_sender_id: str
    # True when EVEROS_SESSION_ID fixed the session, so it may hold what
    # earlier runs stored under the same id.
    session_pinned: bool = False

    @classmethod
    def from_env(cls) -> Settings:
        api_key = os.environ.get("EVEROS_API_KEY", "").strip()
        base_url = base_url_from_env()
        # Self-hosted / OSS deployments have no gateway auth; only require a
        # key when pointing at a hosted environment (anything on evermind.ai).
        host = urlparse(base_url).hostname or ""
        if not api_key and (host == "evermind.ai" or host.endswith(".evermind.ai")):
            raise ConfigError(
                "EVEROS_API_KEY is not set. Get one at "
                "https://everos.evermind.ai/api-keys and put it in your MCP "
                "server config. (Self-hosted EverOS needs no key — set "
                "EVEROS_BASE_URL to your own deployment instead.)"
            )
        user_id = os.environ.get("EVEROS_USER_ID", "").strip()
        if user_id and not valid_id(user_id):
            raise ConfigError(
                f"EVEROS_USER_ID {user_id!r} must be at most {MAX_USER_ID} letters, "
                "digits and the characters _ . @ + -"
            )
        # Optional: a cloud user only has to set the API key. Set it to keep
        # several people's memories apart under one key.
        user_id = user_id or DEFAULT_USER_ID
        pinned_session = os.environ.get("EVEROS_SESSION_ID", "").strip()
        return cls(
            api_key=api_key,
            base_url=base_url,
            user_id=user_id,
            app_id=_env_scope("EVEROS_APP_ID"),
            project_id=_env_scope("EVEROS_PROJECT_ID"),
            # One session per server process (≈ one client connection): two
            # clients never share a buffer, and forget_session has a bounded
            # blast radius.
            session_id=pinned_session or f"mcp-{user_id}-{uuid.uuid4().hex[:12]}",
            session_pinned=bool(pinned_session),
            # Per-user by default so one user's trajectories (and the data in
            # them) never surface in another user's recall. Set it to a shared
            # value explicitly to pool agent experience across a team.
            assistant_sender_id=os.environ.get("EVEROS_ASSISTANT_SENDER_ID", "").strip()
            or f"assistant-{user_id}",
        )

    @classmethod
    def remote(cls, *, api_key: str, user_id: str, base_url: str, session_id: str) -> Settings:
        """Settings for one conversation on the HTTP server: credentials and
        identity come from the request, the endpoint and scope from the
        operator. The agent identity is always per user here — a shared one
        would pool trajectories across everyone who uses the server."""
        return cls(
            api_key=api_key,
            base_url=base_url,
            user_id=user_id,
            app_id=_env_scope("EVEROS_APP_ID"),
            project_id=_env_scope("EVEROS_PROJECT_ID"),
            session_id=session_id,
            assistant_sender_id=f"assistant-{user_id}",
        )


def base_url_from_env() -> str:
    return (
        os.environ.get("EVEROS_BASE_URL", "").strip()
        # Fallback: same variable the everos-cloud SDK uses.
        or os.environ.get("EVER_OS_BASE_URL", "").strip()
        or PROD_BASE_URL
    ).rstrip("/")


def valid_id(value: str) -> bool:
    return bool(_ID_RE.fullmatch(value))


def _env_scope(var: str) -> str:
    return os.environ.get(var, "default").strip() or "default"
