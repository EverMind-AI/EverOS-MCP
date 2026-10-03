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
from everos_mcp.guard import find_secret, find_secret_in
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
        monkeypatch.setattr(server, "_client", client)
        return seen

    server._pending.clear()
    server._notices.clear()
    server._locks.clear()
    server._trajectory_sessions.clear()
    yield install
    server._notices.clear()


async def call_and_drain(coro):
    result = await coro
    await server._drain()
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


def test_config_user_id_defaults_to_os_account(monkeypatch):
    monkeypatch.setenv("EVEROS_API_KEY", "k")
    monkeypatch.delenv("EVEROS_USER_ID", raising=False)
    monkeypatch.delenv("EVEROS_BASE_URL", raising=False)
    monkeypatch.setattr("getpass.getuser", lambda: "Dani Zhu")
    s = Settings.from_env()
    assert s.user_id == "Dani-Zhu" and s.assistant_sender_id == "assistant-Dani-Zhu"


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
        await server._drain()

    run(scenario())
    assert paths(seen) == ["add", "flush"]
    assert sent(seen[0])["async_mode"] is False  # buffer write lands before flush
    assert sent(seen[0])["session_id"] == "s1"
    assert not server._notices


def test_add_memory_wait_skips_flush_when_already_extracted(tool_client):
    seen = tool_client(routes={"add": {"data": {"status": "extracted"}}})
    reply = run(server.add_memory(user_message="x", wait=True))
    assert "now searchable" in reply
    assert paths(seen) == ["add"]


def test_background_failure_surfaces_on_next_reply(tool_client):
    tool_client(respond={"error": {"code": "internal", "message": "boom"}}, status=500)

    async def scenario():
        await call_and_drain(server.add_memory(user_message="x"))
        return await server.search_memory("anything")

    with pytest.raises(EverOSError):  # the search itself also hits the 500
        run(scenario())
    assert server._notices and "boom" in server._notices[0]


def test_notices_are_attached_once(tool_client):
    tool_client()
    server._notices.append("background add_memory failed: boom")
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


def test_record_trajectory_uses_its_own_session(tool_client):
    seen = tool_client(routes={"add": {"data": {"status": "extracted"}}})
    msgs = [
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "name": "t"}]},
        {"role": "tool", "tool_call_id": "c1", "content": "r"},
        {"role": "assistant", "content": "done"},
    ]
    run(call_and_drain(server.record_trajectory(msgs)))
    payload = sent(seen[0])
    assert payload["session_id"].startswith("traj-")
    assert payload["session_id"] != SETTINGS.session_id
    assert paths(seen) == ["add"]  # already extracted: no redundant flush
    assert payload["messages"][1]["tool_calls"][0]["function"]["name"] == "t"
    assert server._trajectory_sessions == [payload["session_id"]]


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


def test_forget_session_deletes_conversation_and_trajectories(tool_client):
    seen = tool_client(
        routes={"add": {"data": {"status": "extracted"}}, "delete": {"data": {"count": 2}}}
    )
    msgs = [{"role": "user", "content": "task"}]

    async def scenario():
        await server.record_trajectory(msgs)
        return await server.forget_session()  # must drain the queued add first

    reply = run(scenario())
    assert paths(seen) == ["add", "delete", "delete"]
    assert sent(seen[1])["session_id"] == "s1"
    assert sent(seen[2])["session_id"].startswith("traj-")
    assert "Deleted 4" in reply and server._trajectory_sessions == []


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
