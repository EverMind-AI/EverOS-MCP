"""OAuth resource-server side of the HTTP transport.

The MCP authorization spec forbids passing a client's access token through to
upstream APIs, and requires checking that the token was issued for this
server. So in OAuth mode the bearer token never leaves this process: it is
introspected (RFC 7662) at the authorization server, which answers with the
EverOS API key the user granted on its consent screen, and only that key
reaches the EverOS API. The gateway keeps authenticating API keys, unchanged.

Introspection contract (the authorization server's side):

    POST <EVEROS_MCP_INTROSPECTION_URL>
    Authorization: Bearer <EVEROS_MCP_INTROSPECTION_SECRET>
    Content-Type: application/x-www-form-urlencoded

    token=<access token>&token_type_hint=access_token

    200 {"active": true, "aud": "<this server's resource URL>",
         "exp": <unix seconds>, "sub": "<user id>",
         "everos_api_key": "<key for the space the user picked>"}
    200 {"active": false}                  (unknown, expired or revoked)
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass

import httpx

from .config import valid_id

# How long an introspection answer is trusted before asking again; bounds
# how late a revocation takes effect.
CACHE_SECONDS = 300
NEGATIVE_CACHE_SECONDS = 60
_MAX_CACHE = 10_000


@dataclass(frozen=True)
class Identity:
    api_key: str
    user_id: str


def user_id_for(sub: str) -> str:
    """The EverOS user id for an OAuth subject. Subjects that are already a
    valid id are used as they are; others (`auth0|abc`, `google-oauth2|1…`)
    are hashed, never collapsed into a shared default."""
    if valid_id(sub):
        return sub
    return "oauth-" + hashlib.sha256(sub.encode()).hexdigest()[:32]


class InvalidToken(Exception):
    """The token is unknown, expired, revoked, or not issued for this server."""


class IntrospectionUnavailable(Exception):
    """The authorization server could not be asked; the token may be fine."""


class Introspector:
    def __init__(
        self,
        url: str,
        *,
        secret: str,
        resource: str,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.resource = resource
        self._url = url
        self._http = httpx.AsyncClient(
            timeout=httpx.Timeout(10.0),
            headers={"Authorization": f"Bearer {secret}"} if secret else {},
            transport=transport,
        )
        # digest -> (valid until, identity); None marks a known-bad token.
        self._cache: dict[str, tuple[float, Identity | None]] = {}

    async def identify(self, token: str) -> Identity:
        digest = hashlib.sha256(token.encode()).hexdigest()
        now = time.time()
        hit = self._cache.get(digest)
        if hit is not None and hit[0] > now:
            if hit[1] is None:
                raise InvalidToken("token is not active")
            return hit[1]

        try:
            resp = await self._http.post(
                self._url, data={"token": token, "token_type_hint": "access_token"}
            )
        except httpx.HTTPError as exc:
            raise IntrospectionUnavailable(str(exc)) from exc
        if resp.status_code != 200:
            raise IntrospectionUnavailable(f"introspection answered HTTP {resp.status_code}")
        try:
            body = resp.json()
        except ValueError as exc:
            raise IntrospectionUnavailable("introspection answered non-JSON") from exc

        if not isinstance(body, dict) or body.get("active") is not True:
            # Remembered briefly, so a flood of bogus tokens does not turn
            # into a flood of introspection calls.
            self._remember(digest, now + NEGATIVE_CACHE_SECONDS, None, now)
            raise InvalidToken("token is not active")
        aud = body.get("aud")
        audiences = aud if isinstance(aud, list) else [aud]
        if self.resource not in audiences:
            # A token minted for some other resource must not be honoured here.
            self._remember(digest, now + NEGATIVE_CACHE_SECONDS, None, now)
            raise InvalidToken("token was not issued for this server")
        api_key = body.get("everos_api_key")
        if not isinstance(api_key, str) or not api_key:
            raise IntrospectionUnavailable("introspection answer lacks everos_api_key")
        sub = str(body.get("sub") or "")
        if not sub:
            # Without a subject every user would share one memory owner.
            raise IntrospectionUnavailable("introspection answer lacks sub")
        identity = Identity(api_key=api_key, user_id=user_id_for(sub))

        exp = body.get("exp")
        until = now + CACHE_SECONDS
        if isinstance(exp, (int, float)):
            until = min(until, float(exp))
        self._remember(digest, until, identity, now)
        return identity

    def _remember(self, digest: str, until: float, identity: Identity | None, now: float) -> None:
        if len(self._cache) >= _MAX_CACHE:
            self._cache = {k: v for k, v in self._cache.items() if v[0] > now}
            if len(self._cache) >= _MAX_CACHE:
                self._cache.clear()
        self._cache[digest] = (until, identity)

    async def aclose(self) -> None:
        await self._http.aclose()
