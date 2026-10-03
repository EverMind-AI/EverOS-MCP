"""Per-conversation runtime state.

Over stdio a server process serves one user, so there is exactly one
conversation, configured from the environment. Over HTTP one process serves
many users: every request carries its own API key, and everything that belongs
to a conversation — its session id, the failure notes of its background saves,
the trajectories it recorded — must never leak into another one.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from collections.abc import Awaitable, Mapping
from typing import Any

import httpx

from .client import EverOSClient, EverOSError, make_http
from .config import Settings, valid_id

log = logging.getLogger("everos_mcp")

USER_HEADER = "x-everos-user-id"
DEFAULT_REMOTE_USER = "default-user"


class Conversation:
    def __init__(self, client: EverOSClient) -> None:
        self.client = client
        self.session_id = client.settings.session_id
        # Outcome notes of background saves that finished badly; they ride
        # along on the next tool result so a failure is never silent.
        self.notices: list[str] = []
        # Sessions created by record_trajectory, for forget_session.
        self.trajectory_sessions: list[str] = []
        self._pending: set[asyncio.Task[None]] = set()
        # Serializes add -> flush on the conversation session so two
        # concurrent saves cannot steal each other's flush. Trajectories get a
        # session of their own and need no lock.
        self._lock = asyncio.Lock()
        self.last_used = time.monotonic()

    @property
    def settings(self) -> Settings:
        return self.client.settings

    @property
    def busy(self) -> bool:
        return bool(self._pending)

    def spawn(self, coro: Awaitable[None], label: str) -> None:
        async def run() -> None:
            try:
                await coro
            except EverOSError as exc:
                self.notices.append(f"background {label} failed: {exc}")
            except Exception as exc:  # never let a background task die silently
                log.exception("background %s crashed", label)
                self.notices.append(f"background {label} failed unexpectedly: {exc!r}")

        task = asyncio.ensure_future(run())
        self._pending.add(task)
        task.add_done_callback(self._pending.discard)

    async def drain(self, timeout: float | None = None) -> None:
        """Wait for in-flight background writes to finish."""
        if self._pending:
            await asyncio.wait(set(self._pending), timeout=timeout)

    def reply(self, text: str) -> str:
        if not self.notices:
            return text
        notes = "\n".join(f"- {n}" for n in self.notices)
        self.notices.clear()
        return f"{text}\n\nEarlier background saves reported problems:\n{notes}"

    async def store(self, messages: list[dict[str, Any]], session_id: str) -> str | None:
        """Write messages, then make sure extraction ran.
        Returns the final status; "extracted" means searchable."""
        if session_id != self.session_id:
            return await self._add_and_flush(messages, session_id)
        async with self._lock:
            return await self._add_and_flush(messages, session_id)

    async def _add_and_flush(self, messages: list[dict[str, Any]], session_id: str) -> str | None:
        # Synchronous add: the buffer write has landed before any flush runs.
        status = (await self.client.add(messages, session_id=session_id, sync=True)).get("status")
        if status != "extracted":
            status = (await self.client.flush(session_id)).get("status")
        return status

    async def store_in_background(
        self, messages: list[dict[str, Any]], session_id: str, *, label: str
    ) -> None:
        status = await self.store(messages, session_id)
        if status != "extracted":
            self.notices.append(
                f"{label}: stored, but extraction returned {status!r}; it is not searchable yet"
            )


class ConversationRegistry:
    """Conversations of the HTTP server, keyed by (API key, user, MCP session).

    The API key is only ever held in memory and forwarded to EverOS; the key
    of this map is its hash. Idle conversations are evicted so a long-running
    server does not grow without bound.
    """

    def __init__(
        self,
        base_url: str,
        *,
        idle_seconds: float = 3600,
        max_size: int = 10_000,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.base_url = base_url
        self._http = make_http(base_url, transport)
        self._items: dict[tuple[str, str, str], Conversation] = {}
        self._idle_seconds = idle_seconds
        self._max_size = max_size

    def resolve(self, headers: Mapping[str, str]) -> Conversation:
        auth = headers.get("authorization", "")
        scheme, _, api_key = auth.partition(" ")
        api_key = api_key.strip()
        if scheme.lower() != "bearer" or not api_key:
            raise EverOSError(
                "unauthorized",
                "missing API key: send 'Authorization: Bearer <EverOS API key>'",
            )
        user_id = headers.get(USER_HEADER, "").strip() or DEFAULT_REMOTE_USER
        if not valid_id(user_id):
            raise EverOSError(
                "invalid_argument",
                f"{USER_HEADER} may only contain letters, digits and _ . @ + -",
            )
        mcp_session = headers.get("mcp-session-id", "").strip()
        if not mcp_session:
            # Stateless client: nothing ties its calls together, so each call
            # is its own conversation.
            return self._new(api_key, user_id)
        key = (hashlib.sha256(api_key.encode()).hexdigest(), user_id, mcp_session)
        conv = self._items.get(key)
        if conv is None:
            self._evict()
            conv = self._items[key] = self._new(api_key, user_id)
        conv.last_used = time.monotonic()
        return conv

    def _new(self, api_key: str, user_id: str) -> Conversation:
        settings = Settings.remote(api_key=api_key, user_id=user_id, base_url=self.base_url)
        return Conversation(EverOSClient(settings, http=self._http))

    def _evict(self) -> None:
        now = time.monotonic()
        idle = [
            k
            for k, c in self._items.items()
            if not c.busy and now - c.last_used > self._idle_seconds
        ]
        for k in idle:
            del self._items[k]
        if len(self._items) >= self._max_size:
            # Still full of live conversations: drop the least recently used.
            oldest = sorted(self._items, key=lambda k: self._items[k].last_used)
            for k in oldest[: len(self._items) - self._max_size + 1]:
                del self._items[k]

    async def aclose(self, drain_timeout: float = 30) -> None:
        pending = [c.drain(drain_timeout) for c in self._items.values() if c.busy]
        if pending:
            await asyncio.gather(*pending)
        await self._http.aclose()

    def __len__(self) -> int:
        return len(self._items)
