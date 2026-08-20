"""Thin HTTP client for the EverOS Cloud Memory API v2.

Deliberately dependency-free (plain httpx, no everos-cloud SDK) so this
package tracks only the wire contract (docs.evermind.ai / openapi.json).

Envelope: success `{"request_id": ..., "data": {...}}`, error
`{"request_id": ..., "error": {"code": ..., "message": ...}}`.
"""

from __future__ import annotations

import time
from typing import Any

import httpx

from .config import Settings


class EverOSError(Exception):
    """API-level error, with a message safe to show to the model."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


def now_ms() -> int:
    """Unix time in milliseconds — the only timestamp unit v2 accepts."""
    return int(time.time() * 1000)


class EverOSClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        headers = {}
        if settings.api_key:
            headers["Authorization"] = f"Bearer {settings.api_key}"
        self._http = httpx.Client(
            base_url=settings.base_url,
            headers=headers,
            timeout=httpx.Timeout(60.0, connect=10.0),
        )

    def close(self) -> None:
        self._http.close()

    # -- transport -----------------------------------------------------------

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            resp = self._http.post(path, json=payload)
        except httpx.HTTPError as exc:
            raise EverOSError(
                "unavailable", f"could not reach EverOS at {self.settings.base_url}: {exc}"
            ) from exc

        try:
            body = resp.json()
        except ValueError:
            body = {}

        error = body.get("error") if isinstance(body, dict) else None
        if resp.status_code < 400 and not error:
            return body.get("data", {}) if isinstance(body, dict) else {}

        code = (error or {}).get("code") or str(resp.status_code)
        message = (error or {}).get("message") or resp.text[:300]
        if resp.status_code == 429 and code in ("resource_exhausted", "quota_exceeded"):
            message = (
                "the EverOS account's usage quota (MCU) is exhausted; writing/reading "
                "memory will keep failing until the plan is upgraded — do not retry"
            )
        elif resp.status_code == 401:
            message = (
                "authentication failed: the API key is invalid or belongs to a "
                "different environment than the configured base URL"
            )
        raise EverOSError(code, message)

    # -- memory ops (all scoped to the configured app/project) ----------------

    def _scope(self) -> dict[str, str]:
        return {"app_id": self.settings.app_id, "project_id": self.settings.project_id}

    def add(
        self, messages: list[dict[str, Any]], session_id: str, *, sync: bool = False
    ) -> dict[str, Any]:
        payload = {**self._scope(), "session_id": session_id, "messages": messages}
        if sync:
            # Gateway-level flag: without it the gateway queues the add and
            # returns "queued", so an immediate flush races the buffer write.
            payload["async_mode"] = False
        return self._post("/api/v2/memory/add", payload)

    def flush(self, session_id: str) -> dict[str, Any]:
        return self._post(
            "/api/v2/memory/flush", {**self._scope(), "session_id": session_id}
        )

    def _owner(self, agent: bool) -> dict[str, str]:
        # v2 requires exactly one of user_id / agent_id. The agent identity is
        # the assistant sender id — the engine attributes agent-track memories
        # to the assistant participant of the trajectory.
        if agent:
            return {"agent_id": self.settings.assistant_sender_id}
        return {"user_id": self.settings.user_id}

    def search(
        self,
        query: str,
        *,
        top_k: int = 10,
        method: str = "hybrid",
        include_profile: bool = False,
        agent: bool = False,
    ) -> dict[str, Any]:
        return self._post(
            "/api/v2/memory/search",
            {
                **self._scope(),
                **self._owner(agent),
                "query": query,
                "method": method,
                "top_k": top_k,
                "include_profile": include_profile,
            },
        )

    def get(
        self,
        memory_type: str,
        *,
        page: int = 1,
        page_size: int = 20,
        agent: bool = False,
    ) -> dict[str, Any]:
        return self._post(
            "/api/v2/memory/get",
            {
                **self._scope(),
                **self._owner(agent),
                "memory_type": memory_type,
                "page": page,
                "page_size": page_size,
            },
        )
