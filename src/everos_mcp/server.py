"""EverOS MCP server: long-term memory tools over the EverOS Cloud API v2."""

from __future__ import annotations

import json
import sys
from typing import Any, Literal

from mcp.server.fastmcp import FastMCP

from .client import EverOSClient, now_ms
from .config import ConfigError, Settings
from .guard import find_secret, refusal

# Spliced into the host's system prompt by clients that honour
# `initialize.instructions`. This is the autonomy protocol: memory only
# works as a product if the model uses these tools at the right moments
# without the user having to say "remember" or "recall".
INSTRUCTIONS = """\
EverOS long-term memory is connected. Follow this protocol AUTONOMOUSLY — act
the moment a trigger fires, never wait for the user to ask you to "remember"
or "recall":

1. At the START of a session, call `get_profile` once to learn who the user
   is. Do not re-fetch it later in the session unless the user asks.
2. The moment the user states a durable fact about themselves — a preference,
   habit, decision, or long-term goal — call `add_memory` immediately. Only a
   result saying the memory is extracted and searchable confirms it is stored;
   otherwise do NOT tell the user it was remembered.
3. When the user references earlier conversations, decisions, or previously
   solved problems ("what did we decide about X", "like last time", "did we
   fix this before"), call `search_memory`. Keep the query SHORT — two to
   eight keywords naming the topic, never a pasted message or transcript.
4. BEFORE starting a non-trivial task that involved tools in the past, call
   `recall_agent_experience` to reuse proven approaches. AFTER solving a
   substantial task (more than three tool-call rounds) in a way worth
   reusing, call `record_trajectory` with the COMPLETE sequence: user
   request, assistant tool calls, tool results, final answer.
5. Do not repeat an identical search in the same turn, and use
   `list_memories` only when the user wants to browse what is stored.
"""

mcp = FastMCP(name="everos", instructions=INSTRUCTIONS)

_client: EverOSClient | None = None


def _get_client() -> EverOSClient:
    global _client
    if _client is None:
        _client = EverOSClient(Settings.from_env())
    return _client


# -- formatting helpers -------------------------------------------------------


def _fmt_episode(ep: dict) -> str:
    text = ep.get("readable_episode") or ep.get("episode") or ep.get("summary") or ""
    ts = str(ep.get("timestamp", ""))[:19]
    score = ep.get("score")
    head = f"[{ts}]" + (f" (relevance {score:.2f})" if isinstance(score, float) else "")
    return f"- {head} {text}"


def _fmt_profile(prof: dict) -> str:
    data = prof.get("profile_data") or {}
    return json.dumps(data, ensure_ascii=False, indent=2)


def _fmt_case(case: dict) -> str:
    parts = [
        f"- [{str(case.get('timestamp', ''))[:19]}] task: {case.get('task_intent', '')}",
        f"  approach: {case.get('approach', '')}",
    ]
    if case.get("key_insight"):
        parts.append(f"  insight: {case['key_insight']}")
    parts.append(f"  quality: {case.get('quality_score', 0)}")
    return "\n".join(parts)


def _fmt_skill(skill: dict) -> str:
    return (
        f"- {skill.get('name', '')} (confidence {skill.get('confidence', 0)}): "
        f"{skill.get('description', '')}\n  {skill.get('content', '')}"
    )


# -- personal-memory tools ------------------------------------------------------


@mcp.tool()
def search_memory(query: str, top_k: int = 10, include_profile: bool = False) -> str:
    """Search the user's long-term memory for past conversations, decisions,
    preferences, and facts.

    Call this proactively — without being asked — whenever the user references
    something from before: "what did we say about X", "like last time", "did
    we fix this before", "continue where we left off".

    `query` is KEYWORDS ONLY: two to eight words naming the topic. Never paste
    the user's whole message, a file, or a transcript — long queries search
    worse. Memories stored moments ago may lag the search index briefly;
    list_memories shows them immediately.
    """
    client = _get_client()
    data = client.search(
        query, top_k=max(1, min(top_k, 100)), include_profile=include_profile
    )
    parts: list[str] = []
    episodes = data.get("episodes") or []
    if episodes:
        parts.append("Relevant memories:\n" + "\n".join(_fmt_episode(e) for e in episodes))
    profiles = data.get("profiles") or []
    if profiles:
        parts.append("User profile:\n" + "\n".join(_fmt_profile(p) for p in profiles))
    if not parts:
        return (
            "No relevant memories found. Note: memories stored in the last few "
            "seconds may not be indexed yet — list_memories shows them immediately."
        )
    return "\n\n".join(parts)


@mcp.tool()
def add_memory(
    user_message: str = "",
    assistant_message: str = "",
    flush_now: bool = True,
) -> str:
    """Store a durable fact or noteworthy exchange in the user's long-term
    memory.

    Call this immediately — without being asked — when the user states
    something that should outlive this conversation: a preference ("I want
    reports in Chinese"), a habit, a decision, or a long-term goal. Pass what
    the user said as user_message and/or your reply as assistant_message.

    With flush_now=true (default) the memory is extracted synchronously and
    the result tells you whether it is really stored — only "extracted" means
    yes. Set flush_now=false only for bulk imports of many related exchanges,
    where deferred extraction groups them into more coherent memories.
    """
    if not user_message and not assistant_message:
        return "Error: provide user_message and/or assistant_message."
    finding = find_secret(user_message + "\n" + assistant_message)
    if finding:
        return refusal(finding)
    client = _get_client()
    s = client.settings
    ts = now_ms()
    messages = []
    if user_message:
        messages.append(
            {"sender_id": s.user_id, "role": "user", "timestamp": ts, "content": user_message}
        )
    if assistant_message:
        messages.append(
            {
                "sender_id": s.assistant_sender_id,
                "role": "assistant",
                "timestamp": ts + 1,
                "content": assistant_message,
            }
        )
    data = client.add(messages, session_id=s.session_id, sync=flush_now)
    status = data.get("status")
    if flush_now and status != "extracted":
        flush_status = client.flush(s.session_id).get("status")
        if flush_status == "extracted":
            return "Stored and extracted; the memory is now searchable."
        return (
            f"Stored, but extraction did not run (flush returned {flush_status!r}). "
            "The memory is NOT searchable yet — do not tell the user it was "
            "remembered."
        )
    if status == "extracted":
        return "Stored and extracted; the memory is now searchable."
    return (
        f"Stored in the session buffer (status: {status}). It is NOT searchable "
        "until extraction runs — call flush_memory when done adding, and do not "
        "claim it was remembered before that."
    )


@mcp.tool()
def flush_memory() -> str:
    """Force extraction of buffered messages so memories stored with
    flush_now=false become searchable. Call it after finishing a bulk import;
    single add_memory calls with the default flush_now=true never need it."""
    client = _get_client()
    status = client.flush(client.settings.session_id).get("status")
    if status == "extracted":
        return "Flushed: buffered messages were extracted and are now searchable."
    return "Nothing to flush: the session buffer was empty or already extracted."


@mcp.tool()
def get_profile() -> str:
    """Get the synthesized profile of the user: stable facts, traits, and
    preferences distilled from all past sessions.

    Call this ONCE at the start of a session to personalize your behavior. Do
    not re-fetch it during the session, and do not use it to recall specific
    past events — that is search_memory's job.
    """
    data = _get_client().get("profile", page_size=10)
    profiles = data.get("profiles") or []
    if not profiles:
        return "No profile yet — it is synthesized after enough memories accumulate."
    return "\n\n".join(_fmt_profile(p) for p in profiles)


@mcp.tool()
def list_memories(
    memory_type: Literal["episode", "profile"] = "episode",
    page: int = 1,
    page_size: int = 20,
) -> str:
    """Browse stored memories chronologically (newest first), with pagination.
    Use this when the user asks what is remembered about them; use
    search_memory when looking for something specific."""
    data = _get_client().get(
        memory_type, page=max(1, page), page_size=max(1, min(page_size, 100))
    )
    total = data.get("total_count", 0)
    if memory_type == "profile":
        items = [_fmt_profile(p) for p in data.get("profiles") or []]
    else:
        items = [_fmt_episode(e) for e in data.get("episodes") or []]
    if not items:
        return "No memories stored yet."
    return f"{total} total, page {page}:\n" + "\n".join(items)


# -- agent-experience tools ------------------------------------------------------


def _normalize_tool_calls(tool_calls: list[dict]) -> list[dict]:
    """Accept both the wire shape {id, type, function:{name, arguments}} and
    the flat shorthand {id, name, arguments} a model is likely to produce."""
    out = []
    for tc in tool_calls:
        if "function" in tc:
            out.append(tc)
        else:
            args = tc.get("arguments", "{}")
            if not isinstance(args, str):
                args = json.dumps(args, ensure_ascii=False)
            out.append(
                {
                    "id": tc.get("id", f"call_{len(out)}"),
                    "type": "function",
                    "function": {"name": tc.get("name", ""), "arguments": args},
                }
            )
    return out


@mcp.tool()
def record_trajectory(messages: list[dict[str, Any]]) -> str:
    """Record how a task was solved so the approach can be reused in future
    sessions. EverOS distills trajectories into cases (concrete solutions) and
    skills (generalized procedures).

    Call this after solving a SUBSTANTIAL task worth repeating. The quality
    gate only distills trajectories with MORE THAN THREE tool-call rounds
    (assistant tool_calls + tool result pairs) — do not record trivial tasks
    or trajectories missing the tool round-trip; they are silently gated out
    and produce no reusable case. Pass the COMPLETE sequence as messages.
    Each message is an object:
      {"role": "user",      "content": "the task request"}
      {"role": "assistant", "content": "...", "tool_calls": [{"id": "c1",
          "name": "tool_name", "arguments": "{\\"json\\": \\"string\\"}"}]}
      {"role": "tool",      "tool_call_id": "c1", "content": "tool result"}
      {"role": "assistant", "content": "the final answer"}
    Timestamps are added automatically.
    """
    if not messages:
        return "Error: messages must be a non-empty trajectory."
    finding = find_secret(
        "\n".join(str(m.get("content") or "") for m in messages)
    )
    if finding:
        return refusal(finding)
    client = _get_client()
    s = client.settings
    ts = now_ms()
    wire: list[dict[str, Any]] = []
    for i, m in enumerate(messages):
        role = m.get("role")
        if role not in ("user", "assistant", "tool"):
            return f"Error: message {i} has invalid role {role!r} (user|assistant|tool)."
        item: dict[str, Any] = {
            "sender_id": s.user_id if role == "user" else s.assistant_sender_id,
            "role": role,
            "timestamp": ts + i,
            "content": m.get("content") or "",
        }
        if m.get("tool_calls"):
            item["tool_calls"] = _normalize_tool_calls(m["tool_calls"])
        if role == "tool":
            if not m.get("tool_call_id"):
                return f"Error: tool message {i} is missing tool_call_id."
            item["tool_call_id"] = m["tool_call_id"]
        wire.append(item)
    client.add(wire, session_id=s.session_id, sync=True)
    status = client.flush(s.session_id).get("status")
    if status == "extracted":
        return (
            "Trajectory recorded and extracted. Distilled cases are available "
            "shortly via recall_agent_experience; skills evolve over time."
        )
    return f"Trajectory stored but extraction returned {status!r} — it may not distill into a case."


@mcp.tool()
def recall_agent_experience(
    kind: Literal["case", "skill", "both"] = "both", page_size: int = 10
) -> str:
    """Recall past problem-solving experience before starting a similar task:
    cases are concrete past solutions (what was tried, how well it worked),
    skills are generalized procedures distilled from many cases.

    Call this at the start of a non-trivial task to reuse proven approaches
    instead of solving from scratch.
    """
    client = _get_client()
    page_size = max(1, min(page_size, 100))
    parts: list[str] = []
    if kind in ("case", "both"):
        cases = client.get("agent_case", page_size=page_size, agent=True).get(
            "agent_cases"
        ) or []
        if cases:
            parts.append("Past cases:\n" + "\n".join(_fmt_case(c) for c in cases))
    if kind in ("skill", "both"):
        skills = client.get("agent_skill", page_size=page_size, agent=True).get(
            "agent_skills"
        ) or []
        if skills:
            parts.append("Learned skills:\n" + "\n".join(_fmt_skill(sk) for sk in skills))
    if not parts:
        return (
            "No recorded experience yet. Record solved tasks with "
            "record_trajectory to build it up."
        )
    return "\n\n".join(parts)


# -- entry point ---------------------------------------------------------------


def main() -> None:
    import logging

    logging.getLogger("httpx").setLevel(logging.WARNING)
    try:
        _get_client()  # fail fast on missing config, before the MCP handshake
    except ConfigError as exc:
        print(f"everos-mcp: {exc}", file=sys.stderr)
        sys.exit(1)
    mcp.run()


if __name__ == "__main__":
    main()
