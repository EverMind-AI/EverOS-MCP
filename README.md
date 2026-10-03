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
        "EVEROS_API_KEY": "sk-..."
      }
    }
  }
}
```

Claude Code one-liner:

```bash
claude mcp add everos -e EVEROS_API_KEY=sk-... -- uvx everos-mcp
```

## Tools

| Tool | Purpose |
|---|---|
| `search_memory` | Relevance search over stored memories (optionally with the user profile) |
| `add_memory` | Store a durable fact or exchange; saves in the background by default (`wait=true` to block until it is searchable) |
| `get_profile` | The synthesized user profile (facts, traits, preferences) |
| `list_memories` | Chronological, paginated browsing |
| `forget_session` | Delete what this connection stored (memories + trajectories); the profile is kept |
| `record_trajectory` | Record how a task was solved (incl. tool calls) for future reuse |
| `recall_agent_experience` | Search distilled cases/skills relevant to the task at hand |

Trajectories need more than three tool-call rounds to pass the distillation
quality gate; shorter ones are stored as episodes but produce no case.

The server ships an autonomy protocol via MCP `instructions`, so hosts that
honour the field make the model load the profile at session start, store
stated facts, and recall past context without being asked.

Note: profile updates and agent case/skill distillation run in an offline
pipeline and land seconds to minutes after the write. A background save that
fails is reported on the next tool result, so the model never silently
believes something was remembered.

Read-only tools carry MCP `readOnlyHint` annotations, so hosts can run them
without a permission prompt; `forget_session` is marked destructive.

## Configuration

| Env var | Required | Default | Meaning |
|---|---|---|---|
| `EVEROS_API_KEY` | yes (cloud) | — | API key; issued per environment |
| `EVEROS_USER_ID` | no | your OS user name | Id owning the memories. Set the same value on every machine to share one memory; letters, digits and `_ . @ + -` only |
| `EVEROS_BASE_URL` | no | `https://api.evermind.ai` | API endpoint; point at your own deployment for self-hosted EverOS |
| `EVEROS_APP_ID` / `EVEROS_PROJECT_ID` | no | `default` | Business scope |
| `EVEROS_SESSION_ID` | no | `mcp-<user_id>-<random>` | Conversation buffer key; a fresh one per server process, so two clients never share a buffer |
| `EVEROS_ASSISTANT_SENDER_ID` | no | `assistant-<user_id>` | Agent identity for trajectories and recalled experience. Per user by default; set the same value for everyone to pool agent experience across a team (their trajectories then become visible to each other) |

## Self-hosted / open-source EverOS

Set `EVEROS_BASE_URL` to your own deployment. No API key is required when the
URL is not an evermind.ai host.

## Security

Every write path runs a credential guard before content leaves the process.
It scans every string in the payload — including trajectory tool-call
arguments and tool results — for high-confidence secret formats (API keys,
AWS/GitHub/Slack/Stripe tokens, private keys, JWTs, bearer tokens, URLs with
embedded passwords, `password=`/`api_key:`-style assignments) and refuses the
write with no bypass flag — long-term memory is not a safe place for secrets.
Store a reference instead.

Recalled memories are returned marked as stored data, and the server
instructions tell the model never to follow directions found inside them.

## Development

```bash
uv sync --dev
uv run ruff check . && uv run pytest      # offline: wire contract, tools, guard
EVEROS_API_KEY=... EVEROS_USER_ID=... python scripts/smoke_test.py   # live e2e
```

## License

MIT
