"""Offline tests: wire contract, tool behavior, tool metadata, credential guard.

No network, no API key. The live end-to-end check is scripts/smoke_test.py.
"""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from everos_mcp import server
from everos_mcp.client import EverOSClient, EverOSError
from everos_mcp.config import ConfigError, Settings
from everos_mcp.conversation import Conversation, ConversationRegistry
from everos_mcp.guard import find_secret, find_secret_in
from everos_mcp.oauth import IntrospectionUnavailable, Introspector, InvalidToken
from everos_mcp.server import mcp

SETTINGS = Settings(
    api_key="test-key",
    base_url="https://gateway.test",
    user_id="u1",
    app_id="default",
    project_id="default",
    session_id="s1",
    assistant_sender_id="assistant-u1",
)

OK = {"request_id": "r1", "data": {}}


def make_client(respond=None, status=200, routes=None, raise_exc=None):
    """Client over a MockTransport; returns (client, captured requests).

    `routes` maps an endpoint suffix ("add", "flush", ...) to a response body.
    """
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if raise_exc is not None:
            raise raise_exc
        body = respond if respond is not None else OK
        if routes:
            body = routes.get(request.url.path.rsplit("/", 1)[-1], OK)
        return httpx.Response(status, json=body)

    return EverOSClient(SETTINGS, transport=httpx.MockTransport(handler)), seen


def sent(req: httpx.Request) -> dict:
    return json.loads(req.content)


def paths(seen: list[httpx.Request]) -> list[str]:
    return [r.url.path.rsplit("/", 1)[-1] for r in seen]


def run(coro):
    return asyncio.run(coro)


@pytest.fixture
def tool_client(monkeypatch):
    """Install a mock client into the server and reset its runtime state."""

    def install(**kwargs):
        client, seen = make_client(**kwargs)
        monkeypatch.setattr(server, "_local", Conversation(client))
        return seen

    yield install


async def call_and_drain(coro):
    result = await coro
    await server._local.drain()
    return result


# -- wire contract -------------------------------------------------------------


def test_add_path_payload_and_auth():
    client, seen = make_client()
    ts = 1_787_000_000_000
    run(
        client.add(
            [{"sender_id": "u1", "role": "user", "timestamp": ts, "content": "hi"}],
            session_id="s1",
        )
    )
    req = seen[-1]
    assert req.url.path == "/api/v2/memory/add"
    assert req.headers["authorization"] == "Bearer test-key"
    payload = sent(req)
    assert payload["session_id"] == "s1"
    assert payload["app_id"] == "default" and payload["project_id"] == "default"
    assert payload["messages"][0]["timestamp"] == ts
    assert "async_mode" not in payload  # async unless sync=True


def test_add_sync_sends_gateway_flag():
    client, seen = make_client()
    run(
        client.add(
            [{"sender_id": "u1", "role": "user", "timestamp": 2_000_000_000_000, "content": "x"}],
            session_id="s1",
            sync=True,
        )
    )
    assert sent(seen[-1])["async_mode"] is False


def test_search_owner_and_defaults():
    client, seen = make_client()
    run(client.search("topic", top_k=5))
    assert seen[-1].url.path == "/api/v2/memory/search"
    payload = sent(seen[-1])
    assert payload["user_id"] == "u1" and "agent_id" not in payload
    assert payload["method"] == "hybrid" and payload["top_k"] == 5


def test_get_agent_owner_switches_to_agent_id():
    client, seen = make_client()
    run(client.get("agent_case", agent=True))
    payload = sent(seen[-1])
    assert payload["agent_id"] == "assistant-u1" and "user_id" not in payload
    assert payload["memory_type"] == "agent_case"


def test_flush_path():
    client, seen = make_client()
    run(client.flush("s1"))
    assert seen[-1].url.path == "/api/v2/memory/flush"
    assert sent(seen[-1])["session_id"] == "s1"


def test_delete_session_scopes():
    client, seen = make_client()
    run(client.delete_session("s1"))
    run(client.delete_session("traj-x", owner=False))
    first, second = sent(seen[0]), sent(seen[1])
    assert seen[0].url.path == "/api/v2/memory/delete"
    assert first["user_id"] == "u1" and first["session_id"] == "s1"
    assert "user_id" not in second and second["session_id"] == "traj-x"


# -- error mapping ---------------------------------------------------------------


def test_quota_429_maps_to_do_not_retry():
    client, _ = make_client(
        respond={"request_id": "r", "error": {"code": "quota_exceeded", "message": "x"}},
        status=429,
    )
    with pytest.raises(EverOSError, match="do not retry"):
        run(client.flush("s1"))


def test_401_maps_to_auth_message():
    client, _ = make_client(respond={"error": {"code": "401", "message": ""}}, status=401)
    with pytest.raises(EverOSError, match="authentication failed"):
        run(client.flush("s1"))


def test_write_read_timeout_warns_against_retry():
    client, _ = make_client(raise_exc=httpx.ReadTimeout("slow"))
    with pytest.raises(EverOSError, match="do NOT retry"):
        run(client.add([], session_id="s1", sync=True))


def test_connect_failure_on_write_is_plain_unavailable():
    client, _ = make_client(raise_exc=httpx.ConnectTimeout("down"))
    with pytest.raises(EverOSError) as info:
        run(client.add([], session_id="s1"))
    assert info.value.code == "unavailable"


# -- config ------------------------------------------------------------------------


def test_config_defaults_isolate_sessions_and_agents(monkeypatch):
    monkeypatch.setenv("EVEROS_API_KEY", "k")
    monkeypatch.setenv("EVEROS_USER_ID", "alice")
    for var in ("EVEROS_SESSION_ID", "EVEROS_ASSISTANT_SENDER_ID", "EVEROS_BASE_URL"):
        monkeypatch.delenv(var, raising=False)
    a, b = Settings.from_env(), Settings.from_env()
    assert a.session_id.startswith("mcp-alice-") and a.session_id != b.session_id
    assert a.assistant_sender_id == "assistant-alice"


def test_config_user_id_defaults_to_a_constant(monkeypatch):
    # One memory per key, the same on every machine and in every container.
    monkeypatch.setenv("EVEROS_API_KEY", "k")
    monkeypatch.delenv("EVEROS_USER_ID", raising=False)
    monkeypatch.delenv("EVEROS_BASE_URL", raising=False)
    s = Settings.from_env()
    assert s.user_id == "default-user" and s.assistant_sender_id == "assistant-default-user"


def test_config_rejects_unsafe_user_id(monkeypatch):
    monkeypatch.setenv("EVEROS_API_KEY", "k")
    monkeypatch.setenv("EVEROS_USER_ID", "../etc")
    with pytest.raises(ConfigError):
        Settings.from_env()


# -- tool behavior -----------------------------------------------------------------


def test_add_memory_background_returns_then_extracts(tool_client):
    seen = tool_client(
        routes={
            "add": {"data": {"status": "accumulated"}},
            "flush": {"data": {"status": "extracted"}},
        }
    )

    async def scenario():
        reply = await server.add_memory(user_message="I prefer dark mode")
        assert "background" in reply
        await server._local.drain()

    run(scenario())
    assert paths(seen) == ["add", "flush"]
    assert sent(seen[0])["async_mode"] is False  # buffer write lands before flush
    assert sent(seen[0])["session_id"] == "s1"
    assert not server._local.notices


def test_add_memory_always_flushes_the_open_tail(tool_client):
    # "extracted" from add means at least one cell closed; the tail can still
    # be buffered, so the flush always runs and "extracted" from either counts.
    seen = tool_client(
        routes={
            "add": {"data": {"status": "extracted"}},
            "flush": {"data": {"status": "no_extraction"}},
        }
    )
    reply = run(server.add_memory(user_message="x", wait=True))
    assert "now searchable" in reply
    assert paths(seen) == ["add", "flush"]


def test_background_failure_surfaces_on_next_reply(tool_client):
    tool_client(respond={"error": {"code": "internal", "message": "boom"}}, status=500)

    async def scenario():
        await call_and_drain(server.add_memory(user_message="x"))
        return await server.search_memory("anything")

    with pytest.raises(EverOSError):  # the search itself also hits the 500
        run(scenario())
    assert server._local.notices and "boom" in server._local.notices[0]


def test_notices_are_attached_once(tool_client):
    tool_client()
    server._local.notices.append("background add_memory failed: boom")
    first = run(server.search_memory("x"))
    second = run(server.search_memory("x"))
    assert "boom" in first and "boom" not in second


def test_add_memory_refuses_secret_without_network(tool_client):
    seen = tool_client()
    reply = run(call_and_drain(server.add_memory(user_message="key sk-abcdefghijklmnop123")))
    assert reply.startswith("Blocked")
    assert seen == []


def test_record_trajectory_scans_tool_call_arguments(tool_client):
    seen = tool_client()
    msgs = [
        {"role": "user", "content": "deploy it"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "c1",
                    "name": "bash",
                    "arguments": {
                        "cmd": "curl -H 'Authorization: Bearer abcdEFGH1234ijklMNOP5678'"
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "ok"},
    ]
    reply = run(call_and_drain(server.record_trajectory(msgs)))
    assert reply.startswith("Blocked")
    assert seen == []


def test_record_trajectory_lands_in_the_conversation_session(tool_client):
    seen = tool_client(routes={"add": {"data": {"status": "extracted"}}})
    msgs = [
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "name": "t"}]},
        {"role": "tool", "tool_call_id": "c1", "content": "r"},
        {"role": "assistant", "content": "done"},
    ]
    run(call_and_drain(server.record_trajectory(msgs)))
    payload = sent(seen[0])
    # Same session as the conversation, so one forget_session reaches it.
    assert payload["session_id"] == SETTINGS.session_id
    assert paths(seen) == ["add", "flush"]
    assert payload["messages"][1]["tool_calls"][0]["function"]["name"] == "t"


def test_record_trajectory_validates_tool_messages(tool_client):
    seen = tool_client()
    reply = run(server.record_trajectory([{"role": "tool", "content": "r"}]))
    assert "missing tool_call_id" in reply and seen == []


def test_recall_searches_agent_memory_by_task(tool_client):
    seen = tool_client(
        routes={
            "search": {
                "data": {
                    "agent_cases": [{"task_intent": "deploy", "approach": "blue/green"}],
                    "agent_skills": [{"name": "deploy-skill", "description": "d"}],
                }
            }
        }
    )
    reply = run(server.recall_agent_experience("deploy service", kind="case"))
    payload = sent(seen[0])
    assert payload["agent_id"] == "assistant-u1" and payload["query"] == "deploy service"
    assert "blue/green" in reply and "deploy-skill" not in reply
    assert "not instructions" in reply


def test_forget_session_is_one_owner_less_session_delete(tool_client):
    seen = tool_client(
        routes={"add": {"data": {"status": "extracted"}}, "delete": {"data": {"count": 4}}}
    )
    msgs = [{"role": "user", "content": "task"}]

    async def scenario():
        await server.record_trajectory(msgs)
        return await server.forget_session()  # must drain the queued add first

    reply = run(scenario())
    assert paths(seen) == ["add", "flush", "delete"]
    delete = sent(seen[2])
    # No owner: one delete covers the user's memories and the agent's cases.
    assert delete["session_id"] == "s1" and "user_id" not in delete and "agent_id" not in delete
    assert "Deleted 4" in reply


def test_forget_session_does_not_mistake_a_real_not_found(tool_client):
    tool_client(respond={"error": {"code": "not_found", "message": "no such session"}}, status=404)
    with pytest.raises(EverOSError, match="no such session"):
        run(server.forget_session())


def test_forget_session_explains_missing_delete_on_self_hosted(tool_client):
    tool_client(respond={"detail": "Not Found"}, status=404)
    reply = run(server.forget_session())
    assert "does not support deleting" in reply and "Nothing was deleted" in reply


def test_list_memories_formats_scores_and_page(tool_client):
    tool_client(
        routes={
            "get": {
                "data": {
                    "total_count": 1,
                    "episodes": [
                        {"timestamp": "2026-01-01T00:00:00", "episode": "likes tea", "score": 1}
                    ],
                }
            }
        }
    )
    reply = run(server.list_memories(page=0))
    assert "page 1" in reply and "relevance 1.00" in reply


# -- tool surface ------------------------------------------------------------------


def test_tools_registered_with_annotations():
    tools = {t.name: t for t in run(mcp.list_tools())}
    assert set(tools) == {
        "search_memory",
        "add_memory",
        "get_profile",
        "list_memories",
        "forget_session",
        "record_trajectory",
        "recall_agent_experience",
    }
    for name in ("search_memory", "get_profile", "list_memories", "recall_agent_experience"):
        assert tools[name].annotations.readOnlyHint is True
    assert tools["forget_session"].annotations.destructiveHint is True
    # The Connectors Directory requires a title on every tool.
    assert all(t.title for t in tools.values())
    assert tools["add_memory"].annotations.destructiveHint is False


def test_tool_descriptions_keep_model_guidance():
    """Tool descriptions are our API to the model — losing a guidance phrase is
    a regression nothing else would catch."""
    desc = {t.name: t.description or "" for t in run(mcp.list_tools())}
    assert "KEYWORDS ONLY" in desc["search_memory"]
    assert "MORE THAN THREE tool-call rounds" in desc["record_trajectory"]
    assert "wait=true" in desc["add_memory"]
    assert "ONLY when the user explicitly asks" in desc["forget_session"]


# -- credential guard ---------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "my key is sk-proj-abc123DEF456ghi789jkl",
        "aws AKIAIOSFODNN7EXAMPLE please",
        "-----BEGIN RSA PRIVATE KEY-----",
        "push token ghp_abcdefghij0123456789abcdefghij",
        "db is postgres://admin:hunter2secret@db.internal/prod",
        "Authorization: Bearer abcdEFGH1234ijklMNOP5678",
        "export AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
        'config = {"api_key": "a1b2c3d4e5f6g7h8"}',
        "DB_PASSWORD=Tr0ub4dor&3",
    ],
)
def test_guard_blocks_credentials(text):
    assert find_secret(text) is not None


@pytest.mark.parametrize(
    "text",
    [
        "User prefers reports in Chinese; favorite city is Hangzhou.",
        "the deploy key lives in 1Password under 'prod-deploy'",
        "password: the one in 1Password",
        "export DB_PASSWORD=$DB_PASSWORD",
        "set max_tokens = 4096 and token_count = 12345678",
    ],
)
def test_guard_passes_normal_content(text):
    assert find_secret(text) is None


def test_guard_scans_nested_structures():
    assert find_secret_in({"a": [{"b": "ghp_abcdefghij0123456789abcdefghij"}]}) is not None
    assert find_secret_in({"a": [{"b": "nothing here"}]}) is None


# -- remote (HTTP) mode -------------------------------------------------------------


def make_registry():
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"data": {"status": "extracted", "count": 1}})

    reg = ConversationRegistry("https://gateway.test", transport=httpx.MockTransport(handler))
    return reg, seen


def hdrs(key="key-a", user=None, session="sess-1"):
    h = {"authorization": f"Bearer {key}"}
    if user:
        h["x-everos-user-id"] = user
    if session:
        h["mcp-session-id"] = session
    return h


def test_registry_requires_bearer_and_valid_user():
    reg, _ = make_registry()
    with pytest.raises(EverOSError, match="missing API key"):
        reg.resolve({})
    with pytest.raises(EverOSError, match="missing API key"):
        reg.resolve({"authorization": "Basic abc"})
    with pytest.raises(EverOSError, match="x-everos-user-id"):
        reg.resolve(hdrs(user="../etc"))


def test_registry_reuses_conversation_per_session_and_isolates_keys(monkeypatch):
    monkeypatch.setenv("EVEROS_ASSISTANT_SENDER_ID", "shared")  # ignored remotely
    reg, _ = make_registry()
    a1 = reg.resolve(hdrs("key-a", "alice"))
    assert reg.resolve(hdrs("key-a", "alice")) is a1
    b = reg.resolve(hdrs("key-b", "alice"))
    other_session = reg.resolve(hdrs("key-a", "alice", session="sess-2"))
    assert len({id(a1), id(b), id(other_session)}) == 3
    assert a1.session_id != other_session.session_id
    assert a1.settings.user_id == "alice"
    assert a1.settings.assistant_sender_id == "assistant-alice"
    assert reg.resolve(hdrs(user=None)).settings.user_id == "default-user"


def test_registry_without_session_header_never_shares():
    reg, _ = make_registry()
    assert reg.resolve(hdrs(session=None)) is not reg.resolve(hdrs(session=None))
    assert len(reg) == 0


def test_registry_forwards_each_callers_key():
    reg, seen = make_registry()

    async def scenario():
        await reg.resolve(hdrs("key-a")).client.search("x")
        await reg.resolve(hdrs("key-b")).client.search("x")
        await reg.aclose()

    run(scenario())
    assert [r.headers["authorization"] for r in seen] == ["Bearer key-a", "Bearer key-b"]


def test_registry_evicts_idle_conversations():
    reg, _ = make_registry()
    reg._idle_seconds = -1  # everything counts as idle
    reg.resolve(hdrs(session="s1"))
    reg.resolve(hdrs(session="s2"))
    assert len(reg) == 1


def test_notices_do_not_cross_conversations():
    reg, _ = make_registry()
    a = reg.resolve(hdrs("key-a"))
    b = reg.resolve(hdrs("key-b"))
    a.notices.append("background add_memory failed: boom")
    assert "boom" not in b.reply("ok")
    assert "boom" in a.reply("ok")


def test_http_app_rejects_missing_bearer_and_serves_health(monkeypatch):
    reg, _ = make_registry()
    monkeypatch.setattr(server, "_registry", None)
    app = server.http_app(reg)

    async def scenario():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1") as c:
            denied = await c.post("/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "ping"})
            health = await c.get("/healthz")
        return denied, health

    denied, health = run(scenario())
    assert denied.status_code == 401
    assert denied.headers["www-authenticate"].startswith("Bearer")
    assert health.status_code == 200 and health.text == "ok"


def test_http_app_publishes_protected_resource_metadata(monkeypatch):
    reg, _ = make_registry()
    monkeypatch.setattr(server, "_registry", None)
    intro, _ = make_introspector({"active": False})
    app = server.http_app(
        reg,
        public_url="https://mcp.example.com",
        authorization_server="https://auth.example.com",
        introspector=intro,
    )

    async def scenario():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1") as c:
            meta = await c.get("/.well-known/oauth-protected-resource/mcp")
            denied = await c.post("/mcp", json={})
        return meta, denied

    meta, denied = run(scenario())
    assert meta.json() == {
        "resource": "https://mcp.example.com/mcp",
        "authorization_servers": ["https://auth.example.com"],
        "bearer_methods_supported": ["header"],
    }
    assert (
        'resource_metadata="https://mcp.example.com/.well-known/oauth-protected-resource/mcp"'
        in denied.headers["www-authenticate"]
    )


def test_http_app_without_authorization_server_has_no_metadata(monkeypatch):
    reg, _ = make_registry()
    monkeypatch.setattr(server, "_registry", None)
    app = server.http_app(reg)

    async def scenario():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1") as c:
            return await c.get("/.well-known/oauth-protected-resource")

    assert run(scenario()).status_code == 404


# -- OAuth mode: token introspection -----------------------------------------------

RESOURCE = "https://mcp.example.com/mcp"


def make_introspector(answer, status=200):
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(status, json=answer)

    intro = Introspector(
        "https://auth.example.com/introspect",
        secret="s3cret",
        resource=RESOURCE,
        transport=httpx.MockTransport(handler),
    )
    return intro, seen


ACTIVE = {"active": True, "aud": RESOURCE, "sub": "user-1", "everos_api_key": "sk-granted"}


def test_introspection_exchanges_token_for_granted_key_and_caches():
    intro, seen = make_introspector(ACTIVE)

    async def scenario():
        first = await intro.identify("tok")
        second = await intro.identify("tok")
        return first, second

    first, second = run(scenario())
    assert first.api_key == "sk-granted" and first.user_id == "user-1"
    assert second is first and len(seen) == 1
    assert seen[0].headers["authorization"] == "Bearer s3cret"
    assert b"token=tok" in seen[0].content


@pytest.mark.parametrize(
    "answer",
    [{"active": False}, {**ACTIVE, "aud": "https://other.example.com/mcp"}],
)
def test_introspection_rejects_inactive_or_foreign_tokens(answer):
    intro, _ = make_introspector(answer)
    with pytest.raises(InvalidToken):
        run(intro.identify("tok"))


@pytest.mark.parametrize(
    "answer,status", [({}, 500), ({k: v for k, v in ACTIVE.items() if k != "everos_api_key"}, 200)]
)
def test_introspection_failures_are_not_auth_failures(answer, status):
    intro, _ = make_introspector(answer, status)
    with pytest.raises(IntrospectionUnavailable):
        run(intro.identify("tok"))


def test_registry_uses_verified_identity_over_headers():
    from everos_mcp.oauth import Identity

    reg, _ = make_registry()
    conv = reg.resolve(
        {"authorization": "Bearer oauth-token", "x-everos-user-id": "mallory"},
        Identity(api_key="sk-granted", user_id="user-1"),
    )
    assert conv.settings.api_key == "sk-granted" and conv.settings.user_id == "user-1"


def test_http_app_oauth_mode_rejects_invalid_token(monkeypatch):
    reg, _ = make_registry()
    intro, _ = make_introspector({"active": False})
    monkeypatch.setattr(server, "_registry", None)
    app = server.http_app(
        reg,
        public_url="https://mcp.example.com",
        authorization_server="https://auth.example.com",
        introspector=intro,
    )

    async def scenario():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1") as c:
            return await c.post("/mcp", json={}, headers={"Authorization": "Bearer bad"})

    denied = run(scenario())
    assert denied.status_code == 401
    assert 'error="invalid_token"' in denied.headers["www-authenticate"]


# -- review fixes ------------------------------------------------------------------


def test_guard_catches_secrets_in_tool_call_argument_objects():
    msgs = [
        {
            "role": "assistant",
            "tool_calls": [
                {"id": "c1", "name": "db", "arguments": {"password": "hunter2xyz9"}},
            ],
        }
    ]
    assert find_secret_in(msgs) is not None
    assert find_secret_in([{"arguments": {"api_key": "abc123def456"}}]) is not None
    assert find_secret_in([{"arguments": {"query": "tea", "limit": 10}}]) is None


def test_oauth_subjects_are_hashed_not_collapsed():
    from everos_mcp.oauth import user_id_for

    a, b = user_id_for("auth0|abc123"), user_id_for("auth0|def456")
    assert a != b and a.startswith("oauth-") and user_id_for("auth0|abc123") == a
    assert user_id_for("plain-user_1") == "plain-user_1"


def test_introspection_without_sub_is_refused():
    intro, _ = make_introspector({k: v for k, v in ACTIVE.items() if k != "sub"})
    with pytest.raises(IntrospectionUnavailable, match="sub"):
        run(intro.identify("tok"))


def test_session_id_survives_eviction():
    reg, _ = make_registry()
    first = reg.resolve(hdrs("key-a", "alice", "sess-1")).session_id
    reg._items.clear()  # idle eviction
    assert reg.resolve(hdrs("key-a", "alice", "sess-1")).session_id == first
    assert reg.resolve(hdrs("key-a", "alice", "sess-2")).session_id != first
    assert reg.resolve(hdrs("key-b", "alice", "sess-1")).session_id != first


def test_ephemeral_conversation_waits_and_has_nothing_to_forget(monkeypatch):
    reg, seen = make_registry()
    conv = reg.resolve(hdrs(session=None))
    assert conv.ephemeral
    monkeypatch.setattr(server, "_conversation", lambda ctx: conv)

    async def scenario():
        saved = await server.add_memory(user_message="likes tea")
        forgot = await server.forget_session()
        return saved, forgot

    saved, forgot = run(scenario())
    assert "now searchable" in saved  # waited instead of backgrounding
    assert "Nothing to forget" in forgot
    assert [r.url.path.rsplit("/", 1)[-1] for r in seen] == ["add", "flush"]


def test_http_app_refuses_half_configured_oauth(monkeypatch):
    reg, _ = make_registry()
    monkeypatch.setattr(server, "_registry", None)
    with pytest.raises(ConfigError, match="both"):
        server.http_app(
            reg, public_url="https://mcp.example.com", authorization_server="https://a.example"
        )


def test_user_ids_leave_room_for_derived_ids(monkeypatch):
    monkeypatch.setenv("EVEROS_API_KEY", "k")
    monkeypatch.setenv("EVEROS_USER_ID", "u" * 100)
    s = Settings.from_env()
    assert len(s.session_id) <= 128 and len(s.assistant_sender_id) <= 128
    monkeypatch.setenv("EVEROS_USER_ID", "u" * 101)
    with pytest.raises(ConfigError, match="at most 100"):
        Settings.from_env()
    reg, _ = make_registry()
    with pytest.raises(EverOSError):
        reg.resolve(hdrs(user="u" * 101))
    assert len(reg.resolve(hdrs(user="u" * 100)).session_id) <= 128


def test_guard_json_pass_does_not_match_across_lines():
    text = "ready at http://localhost:3000\ngit@github.com:org/repo.git"
    assert find_secret_in([{"content": text}]) is None


def test_eviction_never_drops_a_conversation_with_saves_in_flight():
    reg, _ = make_registry()
    reg._max_size = 1
    busy = reg.resolve(hdrs(session="s1"))
    busy._pending.add(object())  # stands in for a running save
    reg.resolve(hdrs(session="s2"))
    assert reg.resolve(hdrs(session="s1")) is busy


# -- third review ------------------------------------------------------------------


def test_forget_waits_for_a_foreground_save(monkeypatch):
    order: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        ep = request.url.path.rsplit("/", 1)[-1]
        if ep == "add":
            await asyncio.sleep(0.05)
        order.append(ep)
        return httpx.Response(200, json={"data": {"status": "accumulated", "count": 1}})

    client = EverOSClient(SETTINGS, transport=httpx.MockTransport(handler))
    monkeypatch.setattr(server, "_local", Conversation(client))

    async def scenario():
        save = asyncio.ensure_future(server.add_memory(user_message="x", wait=True))
        await asyncio.sleep(0.01)  # the save holds the lock, mid-add
        await server.forget_session()
        await save

    run(scenario())
    assert order == ["add", "flush", "delete"]


def test_forget_with_pinned_session_says_so(tool_client, monkeypatch):
    tool_client(routes={"delete": {"data": {"count": 7}}})
    import dataclasses

    conv = server._local
    conv.client.settings = dataclasses.replace(conv.settings, session_pinned=True)
    reply = run(server.forget_session())
    assert "fixed session" in reply and "earlier runs" in reply


def test_record_trajectory_respects_the_api_message_limit(tool_client):
    seen = tool_client()
    msgs = [{"role": "user", "content": "x"}] * 501
    reply = run(server.record_trajectory(msgs))
    assert "at most 500" in reply and seen == []


def test_saturated_conversation_saves_in_the_foreground(tool_client):
    tool_client(routes={"add": {"data": {"status": "extracted"}}})
    conv = server._local
    for _ in range(16):
        conv._pending.add(object())  # stand-ins for running saves
    reply = run(server.add_memory(user_message="x"))
    assert "now searchable" in reply


def test_notices_are_capped():
    reg, _ = make_registry()
    conv = reg.resolve(hdrs())
    for i in range(50):
        conv.note(f"n{i}")
    assert len(conv.notices) == 20 and conv.notices[-1] == "n49"


def test_recalled_data_is_fenced_and_notes_stay_outside(tool_client):
    tool_client(routes={"search": {"data": {"episodes": [{"episode": "likes tea"}]}}})
    server._local.note("background add_memory failed: boom")
    reply = run(server.search_memory("tea"))
    end = reply.index("--- END STORED MEMORY ---")
    assert reply.index("--- BEGIN STORED MEMORY") < reply.index("likes tea") < end
    assert reply.index("boom") > end


def test_profile_is_compact_json(tool_client):
    tool_client(routes={"get": {"data": {"profiles": [{"profile_data": {"a": 1, "b": [2]}}]}}})
    assert '{"a":1,"b":[2]}' in run(server.get_profile())


def test_http_app_rejects_oversized_bodies(monkeypatch):
    reg, _ = make_registry()
    monkeypatch.setattr(server, "_registry", None)
    app = server.http_app(reg)

    async def scenario():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1") as c:
            return await c.post(
                "/mcp",
                content=b"x" * (server.MAX_BODY_BYTES + 1),
                headers={"Authorization": "Bearer k"},
            )

    assert run(scenario()).status_code == 413


def test_invalid_tokens_are_negatively_cached():
    intro, seen = make_introspector({"active": False})

    async def scenario():
        for _ in range(3):
            with pytest.raises(InvalidToken):
                await intro.identify("bogus")

    run(scenario())
    assert len(seen) == 1


def test_guard_lets_secret_references_through():
    for text in ("SecretId: prod/db-creds-2", "secret_name=deploy-key-v2", "sk-loading-spinner"):
        assert find_secret(text) is None, text
    assert find_secret_in([{"SecretId": "prod/db-creds-2"}]) is None
    assert find_secret("SECRET_KEY=abc123def456ghi") is not None


def test_guard_refusal_echoes_no_part_of_the_secret():
    finding = find_secret("my key is sk-proj-abc123DEF456ghi789jkl")
    assert finding == "API key"


# -- API alignment -----------------------------------------------------------------


def test_search_top_k_follows_the_api_range(tool_client):
    seen = tool_client()
    run(server.search_memory("tea"))
    run(server.search_memory("tea", top_k=500))
    run(server.recall_agent_experience("deploy", top_k=80))
    assert [sent(r)["top_k"] for r in seen] == [-1, 100, 80]


def test_dot_ids_are_refused():
    from everos_mcp.config import valid_id

    assert not valid_id(".") and not valid_id("..") and valid_id("a.b")


# -- fourth review ------------------------------------------------------------------


def test_guard_passes_dev_urls_with_at_in_the_path():
    for text in ("http://localhost:5173/@vite/client", "https://host:8443/users/@me"):
        assert find_secret(text) is None, text
    assert find_secret("postgres://admin:hunter2secret@db.internal/prod") is not None


def test_conversation_holding_the_lock_counts_as_busy():
    reg, _ = make_registry()
    conv = reg.resolve(hdrs())

    async def scenario():
        async with conv._lock:
            return conv.busy

    assert run(scenario()) is True and conv.busy is False


def test_forget_soon_after_a_trajectory_warns_about_late_cases(tool_client):
    tool_client(routes={"add": {"data": {"status": "extracted"}}, "delete": {"data": {"count": 1}}})

    async def scenario():
        await server.record_trajectory([{"role": "user", "content": "task"}])
        return await server.forget_session()

    assert "call forget_session again" in run(scenario())


def test_forget_without_recent_trajectory_has_no_warning(tool_client):
    tool_client(routes={"delete": {"data": {"count": 1}}})
    assert "again" not in run(server.forget_session())


def test_chunked_bodies_are_capped_and_replayed(monkeypatch):
    reg, _ = make_registry()
    monkeypatch.setattr(server, "_registry", None)
    app = server.http_app(reg)

    async def chunks(total):  # an async generator: sent chunked, no Content-Length
        for _ in range(total // (512 * 1024)):
            yield b"x" * (512 * 1024)

    async def scenario():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1") as c:
            big = await c.post(
                "/mcp", content=chunks(6 * 1024 * 1024), headers={"Authorization": "Bearer k"}
            )
        return big

    assert run(scenario()).status_code == 413


def test_replay_hands_the_body_over_once():
    async def receive():
        return {"type": "http.disconnect"}

    async def scenario():
        r = server._replay(b"abc", receive)
        return await r(), await r()

    first, second = run(scenario())
    assert first == {"type": "http.request", "body": b"abc", "more_body": False}
    assert second["type"] == "http.disconnect"
