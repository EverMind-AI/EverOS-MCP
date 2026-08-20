# everos-mcp

MCP server for [EverOS](https://evermind.ai) — long-term memory for AI agents.

Gives any MCP client (Claude Code, Claude Desktop, Cursor, …) persistent memory
backed by the EverOS Cloud Memory API: store conversation exchanges, search past
context, and recall a synthesized user profile across sessions.

## Quick start

1. Get an API key at <https://everos.evermind.ai/api-keys>.
2. Add the server to your MCP client config:

```json
{
  "mcpServers": {
    "everos": {
      "command": "uvx",
      "args": ["everos-mcp"],
      "env": {
        "EVEROS_API_KEY": "sk-...",
        "EVEROS_USER_ID": "your-stable-user-id"
      }
    }
  }
}
```

Claude Code one-liner:

```bash
claude mcp add everos -e EVEROS_API_KEY=sk-... -e EVEROS_USER_ID=me -- uvx everos-mcp
```

## Tools

| Tool | Purpose |
|---|---|
| `search_memory` | Relevance search over stored memories (optionally with the user profile) |
| `add_memory` | Store a durable fact or exchange; extracted immediately by default |
| `flush_memory` | Force extraction of messages buffered with `flush_now=false` |
| `get_profile` | The synthesized user profile (facts, traits, preferences) |
| `list_memories` | Chronological, paginated browsing |
| `record_trajectory` | Record how a task was solved (incl. tool calls) for future reuse |
| `recall_agent_experience` | Recall distilled cases/skills before a similar task |

Trajectories need more than three tool-call rounds to pass the distillation
quality gate; shorter ones are stored as episodes but produce no case.

The server ships an autonomy protocol via MCP `instructions`, so hosts that
honour the field make the model load the profile at session start, store
stated facts, and recall past context without being asked.

Note: profile updates and agent case/skill distillation run in an offline
pipeline and land seconds to minutes after the write.

## Configuration

| Env var | Required | Default | Meaning |
|---|---|---|---|
| `EVEROS_API_KEY` | yes (cloud) | — | API key; issued per environment |
| `EVEROS_USER_ID` | yes | — | Stable id owning the memories |
| `EVEROS_BASE_URL` | no | `https://api.evermind.ai` | API endpoint; point at your own deployment for self-hosted EverOS |
| `EVEROS_APP_ID` / `EVEROS_PROJECT_ID` | no | `default` | Business scope |
| `EVEROS_SESSION_ID` | no | `mcp-<user_id>` | Conversation buffer key |
| `EVEROS_ASSISTANT_SENDER_ID` | no | `assistant` | `sender_id` used for assistant messages |

## Self-hosted / open-source EverOS

Set `EVEROS_BASE_URL` to your own deployment. No API key is required when the
URL is not an evermind.ai host.

## Security

Every write path runs a credential guard before content leaves the process:
high-confidence secret formats (API keys, AWS/GitHub/Slack tokens, private
keys, JWTs, URLs with embedded passwords) are refused with no bypass flag —
long-term memory is not a safe place for secrets. Store a reference instead.

## Development

```bash
uv sync --dev
uv run ruff check . && uv run pytest      # offline: wire contract, tools, guard
EVEROS_API_KEY=... EVEROS_USER_ID=... python scripts/smoke_test.py   # live e2e
```

## License

MIT
