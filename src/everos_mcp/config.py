"""Environment-based configuration.

Identity and scope are server-level configuration, not tool parameters: the
LLM should never choose whose memory it is reading or writing.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

PROD_BASE_URL = "https://api.evermind.ai"


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
        if not api_key and "evermind.ai" in base_url:
            raise ConfigError(
                "EVEROS_API_KEY is not set. Get one at "
                "https://everos.evermind.ai/api-keys and put it in your MCP "
                "server config. (Self-hosted EverOS needs no key — set "
                "EVEROS_BASE_URL to your own deployment instead.)"
            )
        user_id = os.environ.get("EVEROS_USER_ID", "").strip()
        if not user_id:
            raise ConfigError(
                "EVEROS_USER_ID is not set. Memories are stored per user; set "
                "a stable identifier for whose memory this server manages, "
                'e.g. EVEROS_USER_ID="dani".'
            )
        return cls(
            api_key=api_key,
            base_url=base_url,
            user_id=user_id,
            app_id=os.environ.get("EVEROS_APP_ID", "default").strip() or "default",
            project_id=os.environ.get("EVEROS_PROJECT_ID", "default").strip()
            or "default",
            session_id=os.environ.get("EVEROS_SESSION_ID", "").strip()
            or f"mcp-{user_id}",
            assistant_sender_id=os.environ.get(
                "EVEROS_ASSISTANT_SENDER_ID", "assistant"
            ).strip()
            or "assistant",
        )
