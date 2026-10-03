"""Environment-based configuration.

Identity and scope are server-level configuration, not tool parameters: the
LLM should never choose whose memory it is reading or writing.
"""

from __future__ import annotations

import getpass
import os
import re
import uuid
from dataclasses import dataclass
from urllib.parse import urlparse

PROD_BASE_URL = "https://api.evermind.ai"

# The id charset both the cloud and the self-hosted server accept (ids become
# path segments on self-hosted EverOS).
_ID_RE = re.compile(r"[A-Za-z0-9_.@+-]{1,128}")


def default_user_id() -> str:
    try:
        name = getpass.getuser()
    except (KeyError, OSError):  # no passwd entry / no login name (containers)
        name = ""
    name = re.sub(r"[^A-Za-z0-9_.@+-]", "-", name).strip("-")[:128]
    return name or "default-user"


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

    @classmethod
    def from_env(cls) -> Settings:
        api_key = os.environ.get("EVEROS_API_KEY", "").strip()
        # Self-hosted / OSS deployments have no gateway auth; only require a
        # key when pointing at a hosted environment (anything on evermind.ai).
        base_url = (
            os.environ.get("EVEROS_BASE_URL", "").strip()
            # Fallback: same variable the everos-cloud SDK uses.
            or os.environ.get("EVER_OS_BASE_URL", "").strip()
            or PROD_BASE_URL
        ).rstrip("/")
        host = urlparse(base_url).hostname or ""
        if not api_key and (host == "evermind.ai" or host.endswith(".evermind.ai")):
            raise ConfigError(
                "EVEROS_API_KEY is not set. Get one at "
                "https://everos.evermind.ai/api-keys and put it in your MCP "
                "server config. (Self-hosted EverOS needs no key — set "
                "EVEROS_BASE_URL to your own deployment instead.)"
            )
        user_id = os.environ.get("EVEROS_USER_ID", "").strip()
        if user_id and not _ID_RE.fullmatch(user_id):
            raise ConfigError(
                f"EVEROS_USER_ID {user_id!r} may only contain letters, digits "
                "and the characters _ . @ + -"
            )
        # Optional: default to the OS account so a cloud user only has to set
        # the API key. Set it explicitly to share one memory across machines.
        user_id = user_id or default_user_id()
        return cls(
            api_key=api_key,
            base_url=base_url,
            user_id=user_id,
            app_id=os.environ.get("EVEROS_APP_ID", "default").strip() or "default",
            project_id=os.environ.get("EVEROS_PROJECT_ID", "default").strip() or "default",
            # One session per server process (≈ one client connection): two
            # clients never share a buffer, and forget_session has a bounded
            # blast radius.
            session_id=os.environ.get("EVEROS_SESSION_ID", "").strip()
            or f"mcp-{user_id}-{uuid.uuid4().hex[:12]}",
            # Per-user by default so one user's trajectories (and the data in
            # them) never surface in another user's recall. Set it to a shared
            # value explicitly to pool agent experience across a team.
            assistant_sender_id=os.environ.get("EVEROS_ASSISTANT_SENDER_ID", "").strip()
            or f"assistant-{user_id}",
        )
