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
import uuid
from collections.abc import Awaitable, Mapping
from typing import Any

import httpx

from .client import EverOSClient, EverOSError, make_http
from .config import DEFAULT_USER_ID, Settings, valid_id
from .oauth import Identity

log = logging.getLogger("everos_mcp")

USER_HEADER = "x-everos-user-id"
# Bounds per conversation, so one caller cannot pile up unbounded work: past
# MAX_PENDING background saves, further saves run in the foreground; only the
# latest MAX_NOTICES failure notes are kept.
MAX_PENDING = 16
MAX_NOTICES = 20


class Conversation:
    """One conversation: one EverOS session holding everything it stored —
    memories and recorded trajectories alike — so forget_session can delete
    it all with one session-scoped delete.

    `ephemeral` marks an HTTP caller that sends no MCP session id: nothing
    ties its calls together, so it gets no background work (a later call
    could never report the outcome) and nothing to forget.
    """

    def __init__(self, client: EverOSClient, *, ephemeral: bool = False) -> None:
        self.client = client
        self.session_id = client.settings.session_id
        self.ephemeral = ephemeral
        # Outcome notes of background saves that finished badly; they ride
        # along on the next tool result so a failure is never silent.
        self.notices: list[str] = []
        self._pending: set[asyncio.Task[None]] = set()
        # Serializes add -> flush on the session. Every store ends with the
        # buffer extracted, so a trajectory never mixes with earlier messages
        # and each store is extracted as its own unit.
        self._lock = asyncio.Lock()
        self.last_used = time.monotonic()
        # When record_trajectory last ran: cases are distilled from it after
        # extraction, so a delete soon after can miss one still in progress.
        self.last_trajectory_at: float | None = None

    @property
    def settings(self) -> Settings:
        return self.client.settings

    @property
    def busy(self) -> bool:
        # A foreground store or a delete holds the lock without a task.
        return bool(self._pending) or self._lock.locked()

    @property
    def saturated(self) -> bool:
        return len(self._pending) >= MAX_PENDING

    def note(self, text: str) -> None:
        self.notices.append(text)
        del self.notices[:-MAX_NOTICES]

    def spawn(self, coro: Awaitable[None], label: str) -> None:
        async def run() -> None:
            try:
                await coro
            except EverOSError as exc:
                self.note(f"background {label} failed: {exc}")
            except Exception as exc:  # never let a background task die silently
                log.exception("background %s crashed", label)
                self.note(f"background {label} failed unexpectedly: {exc!r}")

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
        async with self._lock:
            return await self._add_and_flush(messages, session_id)

    async def _add_and_flush(self, messages: list[dict[str, Any]], session_id: str) -> str | None:
        # Synchronous add: the buffer write has landed before the flush runs.
        added = (await self.client.add(messages, session_id=session_id, sync=True)).get("status")
        # Always flush: an add answering "extracted" only means at least one
        # cell closed — boundary detection leaves the open tail buffered. The
        # flush closes it, so every store ends with an empty buffer.
        flushed = (await self.client.flush(session_id)).get("status")
        return "extracted" if "extracted" in (added, flushed) else flushed or added

    async def store_in_background(
        self, messages: list[dict[str, Any]], session_id: str, *, label: str
    ) -> None:
        status = await self.store(messages, session_id)
        if status != "extracted":
            self.note(
                f"{label}: stored, but extraction returned {status!r}; it is not searchable yet"
            )

    async def delete_all(self) -> dict[str, Any]:
        """Delete everything this conversation's session holds. Background
        saves are drained first, and the delete holds the save lock, so a
        save running in the foreground cannot land after it."""
        await self.drain()
        async with self._lock:
            # Session-scoped, without an owner: covers the user's memories and
            # the agent cases distilled from this session's trajectories.
            return await self.client.delete_session(self.session_id, owner=False)


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

    def resolve(self, headers: Mapping[str, str], identity: Identity | None = None) -> Conversation:
        """`identity` comes from a verified OAuth token (OAuth mode); without
        it the bearer token is the caller's own EverOS API key."""
        if identity is not None:
            api_key, user_id = identity.api_key, identity.user_id
        else:
            api_key, user_id = self._api_key_identity(headers)
        mcp_session = headers.get("mcp-session-id", "").strip()
        if not mcp_session:
            # Stateless client: nothing ties its calls together, so each call
            # is its own conversation.
            return self._new(api_key, user_id, f"mcp-{user_id}-{uuid.uuid4().hex[:12]}", True)
        key = (hashlib.sha256(api_key.encode()).hexdigest(), user_id, mcp_session)
        conv = self._items.get(key)
        if conv is None:
            self._evict()
            # Derived, not random: a conversation evicted while idle comes
            # back on the same EverOS session, so forget_session still
            # reaches everything the MCP session stored.
            digest = hashlib.sha256("|".join(key).encode()).hexdigest()[:12]
            conv = self._items[key] = self._new(api_key, user_id, f"mcp-{user_id}-{digest}")
        conv.last_used = time.monotonic()
        return conv

    @staticmethod
    def _api_key_identity(headers: Mapping[str, str]) -> tuple[str, str]:
        auth = headers.get("authorization", "")
        scheme, _, api_key = auth.partition(" ")
        api_key = api_key.strip()
        if scheme.lower() != "bearer" or not api_key:
            raise EverOSError(
                "unauthorized",
                "missing API key: send 'Authorization: Bearer <EverOS API key>'",
            )
        user_id = headers.get(USER_HEADER, "").strip() or DEFAULT_USER_ID
        if not valid_id(user_id):
            raise EverOSError(
                "invalid_argument",
                f"{USER_HEADER} must be at most 100 letters, digits and _ . @ + -",
            )
        return api_key, user_id

    def _new(
        self, api_key: str, user_id: str, session_id: str, ephemeral: bool = False
    ) -> Conversation:
        settings = Settings.remote(
            api_key=api_key, user_id=user_id, base_url=self.base_url, session_id=session_id
        )
        return Conversation(EverOSClient(settings, http=self._http), ephemeral=ephemeral)

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
            # Still full: drop the least recently used, but never one with a
            # save in flight — its outcome note, its shutdown drain and its
            # add/flush lock all live on it. If every one is busy, the map
            # runs over its cap until saves finish.
            idle_lru = sorted(
                (k for k, c in self._items.items() if not c.busy),
                key=lambda k: self._items[k].last_used,
            )
            for k in idle_lru[: len(self._items) - self._max_size + 1]:
                del self._items[k]

    async def aclose(self, drain_timeout: float = 30) -> None:
        pending = [c.drain(drain_timeout) for c in self._items.values() if c.busy]
        if pending:
            await asyncio.gather(*pending)
        await self._http.aclose()

    def __len__(self) -> int:
        return len(self._items)
