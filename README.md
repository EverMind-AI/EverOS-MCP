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
| `forget_session` | Delete what this connection stored (memories + cases distilled from its trajectories); the profile and learned skills are kept |
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
| `EVEROS_USER_ID` | no | `default-user` | Id owning the memories: one memory per API key by default, the same on every machine. Set it to keep several people apart under one key; up to 100 letters, digits and `_ . @ + -` |
| `EVEROS_BASE_URL` | no | `https://api.evermind.ai` | API endpoint; point at your own deployment for self-hosted EverOS |
| `EVEROS_APP_ID` / `EVEROS_PROJECT_ID` | no | `default` | Business scope |
| `EVEROS_SESSION_ID` | no | `mcp-<user_id>-<random>` | Session everything is stored under; a fresh one per server process. Setting a fixed value makes `forget_session` delete everything ever stored under it, by any run |
| `EVEROS_ASSISTANT_SENDER_ID` | no | `assistant-<user_id>` | Agent identity for trajectories and recalled experience. Per user by default; set the same value for everyone to pool agent experience across a team (their trajectories then become visible to each other) |

## Self-hosted / open-source EverOS

Set `EVEROS_BASE_URL` to your own deployment. No API key is required when the
URL is not an evermind.ai host.

## Remote server (streamable HTTP)

The same package runs as a shared, hosted MCP server. It holds no credentials
of its own: every request brings the caller's EverOS API key, which is
forwarded to the EverOS API and never stored.

```bash
everos-mcp --transport http --host 0.0.0.0 --port 8765
```

Clients connect with their own key:

```bash
claude mcp add --transport http everos https://mcp.example.com/mcp \
  --header "Authorization: Bearer sk-..."
```

| Request header | Required | Meaning |
|---|---|---|
| `Authorization: Bearer <key>` | yes | The caller's EverOS API key. Missing → HTTP 401 with `WWW-Authenticate: Bearer` |
| `X-EverOS-User-Id` | no | Whose memory within the key's space (default `default-user`) |

| Server env var | Default | Meaning |
|---|---|---|
| `EVEROS_BASE_URL` | `https://api.evermind.ai` | EverOS API the server talks to |
| `EVEROS_APP_ID` / `EVEROS_PROJECT_ID` | `default` | Business scope for every caller |
| `EVEROS_MCP_HOST` / `EVEROS_MCP_PORT` | `127.0.0.1` / `8765` | Bind address (same as `--host` / `--port`) |
| `EVEROS_MCP_ALLOWED_HOSTS` | — | Comma-separated public host names to accept (DNS-rebinding protection) |
| `EVEROS_MCP_PUBLIC_URL` | — | Public base URL of this server, e.g. `https://mcp.example.com` |
| `EVEROS_MCP_AUTHORIZATION_SERVER` | — | OAuth issuer that signs users in. When set, the server publishes RFC 9728 metadata at `/.well-known/oauth-protected-resource/mcp` and points to it from the 401 challenge |
| `EVEROS_MCP_INTROSPECTION_URL` / `EVEROS_MCP_INTROSPECTION_SECRET` | — | OAuth mode (both required, together with the authorization server): bearer tokens are verified at this RFC 7662 endpoint (audience must be this server) and exchanged for the EverOS API key the user granted. The token itself is never forwarded upstream, as the MCP authorization spec requires. Contract: `src/everos_mcp/oauth.py` |

Deployment notes:

- Terminate TLS at the ingress; `GET /healthz` is the liveness probe.
- Each MCP session lives in the memory of the replica that created it. With
  more than one replica, route by the `Mcp-Session-Id` header (sticky
  sessions).
- An API key is the trust boundary: anyone holding a key can read and write
  every user id within that key's space (`X-EverOS-User-Id` is chosen by the
  caller). Give separate people separate keys when that matters.
- Conversations are isolated by (API key, user, MCP session): one caller never
  sees another's session, background-save notes, or trajectories.
- Hosts that only connect through OAuth (claude.ai connectors, ChatGPT) need an
  authorization server; set `EVEROS_MCP_AUTHORIZATION_SERVER` once one exists.

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

## Releasing

Bump `version` in `pyproject.toml` and `__version__` in
`src/everos_mcp/__init__.py`, merge, then push a matching tag:

```bash
git tag v0.1.0 && git push origin v0.1.0
```

`.github/workflows/release.yml` tests, builds and smoke-tests the wheel, waits
for approval on the `release` environment, publishes to PyPI through Trusted
Publishing (no stored token), and drafts the GitHub Release.

## License

MIT
