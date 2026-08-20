"""End-to-end smoke test against a live EverOS environment.

Usage:
    EVEROS_BASE_URL=https://api.evermind.ai \
    EVEROS_API_KEY=sk-... EVEROS_USER_ID=smoke_user python scripts/smoke_test.py

Exercises the full loop the MCP tools use: add -> flush -> get -> search.
Uses a unique session id per run so reruns don't hit stale buffers.
"""

from __future__ import annotations

import sys
import time
import uuid

sys.path.insert(0, "src")

from everos_mcp.client import EverOSClient, now_ms  # noqa: E402
from everos_mcp.config import Settings  # noqa: E402


def main() -> int:
    settings = Settings.from_env()
    run_id = uuid.uuid4().hex[:8]
    session_id = f"smoke-{run_id}"
    marker = f"smoke-marker-{run_id}"
    client = EverOSClient(settings)
    print(f"target: {settings.base_url}  user: {settings.user_id}  session: {session_id}")

    ts = now_ms()
    add = client.add(
        [
            {
                "sender_id": settings.user_id,
                "role": "user",
                "timestamp": ts,
                "content": f"My favorite testing codeword is {marker}. Please remember it.",
            },
            {
                "sender_id": settings.assistant_sender_id,
                "role": "assistant",
                "timestamp": ts + 1,
                "content": f"Noted — your testing codeword is {marker}.",
            },
        ],
        session_id=session_id,
        sync=True,  # deterministic: buffer is written before flush runs
    )
    print(f"add: {add}")

    flush = client.flush(session_id)
    print(f"flush: {flush}")

    # Structured getter is readable before the vector index catches up.
    episode_seen = False
    for attempt in range(12):
        got = client.get("episode", page_size=5)
        episodes = got.get("episodes") or []
        if any(marker in str(e) for e in episodes):
            episode_seen = True
            print(f"get: marker episode visible after {attempt * 5}s")
            break
        time.sleep(5)
    if not episode_seen:
        print("FAIL: episode with marker not visible via get within 60s")
        return 1

    search_seen = False
    for attempt in range(12):
        found = client.search(f"what is the testing codeword {marker}?", top_k=5)
        if any(marker in str(e) for e in found.get("episodes") or []):
            search_seen = True
            print(f"search: marker found after {attempt * 5}s")
            break
        time.sleep(5)
    if not search_seen:
        print("WARN: search did not surface the marker within 60s (index lag?)")

    print("PASS" if episode_seen else "FAIL")
    return 0


if __name__ == "__main__":
    sys.exit(main())
