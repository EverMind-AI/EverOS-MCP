"""Offline tests: wire contract, tool metadata, credential guard.

No network, no API key. The live end-to-end check is scripts/smoke_test.py.
"""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from everos_mcp.client import EverOSClient, EverOSError
from everos_mcp.config import Settings
from everos_mcp.guard import find_secret
from everos_mcp.server import mcp

SETTINGS = Settings(
    api_key="test-key",
    base_url="https://gateway.test",
    user_id="u1",
    app_id="default",
    project_id="default",
    session_id="s1",
    assistant_sender_id="assistant",
)


def make_client(respond=None, status=200):
    """Client over a MockTransport; returns (client, captured requests)."""
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        body = respond if respond is not None else {"request_id": "r1", "data": {}}
        return httpx.Response(status, json=body)

    return EverOSClient(SETTINGS, transport=httpx.MockTransport(handler)), seen


def sent(seen: list[httpx.Request]) -> dict:
    return json.loads(seen[-1].content)


# -- wire contract -------------------------------------------------------------


def test_add_path_payload_and_auth():
    client, seen = make_client()
    ts = 1_787_000_000_000
    client.add(
        [{"sender_id": "u1", "role": "user", "timestamp": ts, "content": "hi"}],
        session_id="s1",
    )
    req = seen[-1]
    assert req.url.path == "/api/v2/memory/add"
    assert req.headers["authorization"] == "Bearer test-key"
    payload = sent(seen)
    assert payload["session_id"] == "s1"
    assert payload["app_id"] == "default" and payload["project_id"] == "default"
    assert payload["messages"][0]["timestamp"] == ts
    assert "async_mode" not in payload  # async unless sync=True


def test_add_sync_sends_gateway_flag():
    client, seen = make_client()
    client.add(
        [{"sender_id": "u1", "role": "user", "timestamp": 2_000_000_000_000, "content": "x"}],
        session_id="s1",
        sync=True,
    )
    assert sent(seen)["async_mode"] is False


def test_search_owner_and_defaults():
    client, seen = make_client()
    client.search("topic", top_k=5)
    assert seen[-1].url.path == "/api/v2/memory/search"
    payload = sent(seen)
    assert payload["user_id"] == "u1" and "agent_id" not in payload
    assert payload["method"] == "hybrid" and payload["top_k"] == 5


def test_get_agent_owner_switches_to_agent_id():
    client, seen = make_client()
    client.get("agent_case", agent=True)
    payload = sent(seen)
    assert payload["agent_id"] == "assistant" and "user_id" not in payload
    assert payload["memory_type"] == "agent_case"


def test_flush_path():
    client, seen = make_client()
    client.flush("s1")
    assert seen[-1].url.path == "/api/v2/memory/flush"
    assert sent(seen)["session_id"] == "s1"


# -- error mapping ---------------------------------------------------------------


def test_quota_429_maps_to_do_not_retry():
    client, _ = make_client(
        respond={"request_id": "r", "error": {"code": "quota_exceeded", "message": "x"}},
        status=429,
    )
    with pytest.raises(EverOSError, match="do not retry"):
        client.flush("s1")


def test_401_maps_to_auth_message():
    client, _ = make_client(respond={"error": {"code": "401", "message": ""}}, status=401)
    with pytest.raises(EverOSError, match="authentication failed"):
        client.flush("s1")


# -- tool surface ------------------------------------------------------------------


def test_seven_tools_registered():
    tools = {t.name for t in asyncio.run(mcp.list_tools())}
    assert tools == {
        "search_memory",
        "add_memory",
        "flush_memory",
        "get_profile",
        "list_memories",
        "record_trajectory",
        "recall_agent_experience",
    }


def test_tool_descriptions_keep_model_guidance():
    """Tool descriptions are our API to the model — losing a guidance phrase is
    a regression nothing else would catch."""
    desc = {t.name: t.description or "" for t in asyncio.run(mcp.list_tools())}
    assert "KEYWORDS ONLY" in desc["search_memory"]
    assert "MORE THAN THREE tool-call rounds" in desc["record_trajectory"]
    assert "flush_now" in desc["add_memory"]


# -- credential guard ---------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "my key is sk-proj-abc123DEF456ghi789jkl",
        "aws AKIAIOSFODNN7EXAMPLE please",
        "-----BEGIN RSA PRIVATE KEY-----",
        "push token ghp_abcdefghij0123456789abcdefghij",
        "db is postgres://admin:hunter2secret@db.internal/prod",
    ],
)
def test_guard_blocks_credentials(text):
    assert find_secret(text) is not None


def test_guard_passes_normal_content():
    assert find_secret("User prefers reports in Chinese; favorite city is Hangzhou.") is None
    assert find_secret("the deploy key lives in 1Password under 'prod-deploy'") is None
