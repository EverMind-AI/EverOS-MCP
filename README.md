<div align="center" id="readme-top">

![EverOS banner](https://github.com/user-attachments/assets/8e217d39-5d15-4c6c-9b54-3e83add4e0f2)

<h3 align="center">🧠 Long-term memory for any MCP client — the official EverOS <b>MCP server</b></h3>

<p align="center">
  <a href="https://x.com/evermind"><img src="https://img.shields.io/badge/EverMind-000000?labelColor=gray&style=for-the-badge&logo=x&logoColor=white" alt="X"></a>
  <a href="https://huggingface.co/EverMind-AI"><img src="https://img.shields.io/badge/🤗_HuggingFace-EverMind-F5C842?labelColor=gray&style=for-the-badge" alt="HuggingFace"></a>
  <a href="https://discord.gg/gYep5nQRZJ"><img src="https://img.shields.io/badge/Discord-EverMind-404EED?labelColor=gray&style=for-the-badge&logo=discord&logoColor=white" alt="Discord"></a>
</p>

<p align="center">
  <a href="https://pypi.org/project/everos-mcp/"><img src="https://img.shields.io/pypi/v/everos-mcp?color=2DABC2&style=for-the-badge" alt="PyPI"></a>
  <img src="https://img.shields.io/pypi/pyversions/everos-mcp?style=for-the-badge" alt="Python">
  <img src="https://img.shields.io/badge/license-MIT-green?style=for-the-badge" alt="License">
</p>

[Website](https://evermind.ai) · [Documentation](https://docs.evermind.ai) · [Console](https://everos.evermind.ai) · [GitHub](https://github.com/EverMind-AI/EverOS-MCP)

</div>

<br>

> **Which package?** This is **everos-mcp**: plug EverOS memory into Claude Code, Claude
> Desktop, Cursor, Codex, or any other MCP client, with no code.
>
> Calling EverOS from your own Python code? Use the [`everos-cloud`](https://pypi.org/project/everos-cloud/) SDK.
> Want to self-host? Run the open-source [`everos`](https://pypi.org/project/everos/) server, then point this package at it.

# EverOS MCP Server

Your AI assistant forgets everything when the session ends. **everos-mcp** gives it a
memory that lasts: it remembers what you told it, builds a profile of how you work, and
learns from how past tasks were solved, across every session and every machine.

## Why everos-mcp

- **Remembers across sessions** — facts, decisions, and preferences you mention are stored
  and recalled by relevance whenever they matter, not dumped into every prompt.
- **Knows who you are** — EverOS distills a user profile (facts, traits, preferences) from
  your conversations, and the assistant loads it at the start of each session.
- **Learns from experience** — record how a task was solved, tool calls included, and EverOS
  distills it into reusable cases and skills the agent recalls next time.
- **Works on its own** — the server ships an autonomy protocol through MCP `instructions`,
  so capable hosts load the profile, store new facts, and recall context without being asked.
- **Safe by default** — a credential guard refuses to store secrets, recalled memories are
  fenced as data rather than instructions, and you can delete everything a session stored.
- **Cloud or self-hosted** — EverOS Cloud out of the box; one environment variable points it
  at your own EverOS deployment.

## Quick start

1. Get an API key from the [EverOS Console](https://everos.evermind.ai/api-keys).
2. Make sure [`uv`](https://docs.astral.sh/uv/getting-started/installation/) is installed
   (`uvx` runs the server without a manual install).
3. Add the server to your client:

**Claude Code**

```bash
claude mcp add everos -e EVEROS_API_KEY=sk-... -- uvx everos-mcp
```

**Codex**

```bash
codex mcp add everos --env EVEROS_API_KEY=sk-... -- uvx everos-mcp
```

**Claude Desktop, Cursor, and other JSON-configured clients** — add this to the client's
MCP config (`claude_desktop_config.json`, `~/.cursor/mcp.json`, …):

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

That's it. Tell your assistant something worth remembering, start a new session, and ask
about it.

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

Read-only tools carry the MCP `readOnlyHint` annotation, so hosts can run them without a
permission prompt; `forget_session` is marked destructive.

**Good to know**

- Profile updates and case/skill distillation run in an offline pipeline and land seconds
  to minutes after the write.
- A background save that fails is reported on the next tool result, so the model never
  silently believes something was remembered.
- Trajectories need more than three tool-call rounds to pass the distillation quality
  gate; shorter ones are stored as episodes but produce no case.

## Configuration

| Env var | Required | Default | Meaning |
|---|---|---|---|
| `EVEROS_API_KEY` | yes (cloud) | — | API key; issued per environment |
| `EVEROS_USER_ID` | no | `default-user` | Id owning the memories: one memory per API key by default, the same on every machine. Set it to keep several people apart under one key; up to 100 letters, digits and `_ . @ + -` |
| `EVEROS_BASE_URL` | no | `https://api.evermind.ai` | API endpoint; point at your own deployment for self-hosted EverOS |
| `EVEROS_APP_ID` / `EVEROS_PROJECT_ID` | no | `default` | Business scope |
| `EVEROS_SESSION_ID` | no | `mcp-<user_id>-<random>` | Session everything is stored under; a fresh one per server process. Setting a fixed value makes `forget_session` delete everything ever stored under it, by any run |
| `EVEROS_ASSISTANT_SENDER_ID` | no | `assistant-<user_id>` | Agent identity for trajectories and recalled experience. Per user by default; set the same value for everyone to pool agent experience across a team (their trajectories then become visible to each other) |

### Self-hosted / open-source EverOS

Set `EVEROS_BASE_URL` to your own [EverOS](https://github.com/EverMind-AI/EverOS)
deployment. No API key is required when the URL is not an evermind.ai host.

```bash
claude mcp add everos -e EVEROS_BASE_URL=http://127.0.0.1:8000 -- uvx everos-mcp
```

## Security

Every write path runs a credential guard before content leaves the process. It scans
every string in the payload — including trajectory tool-call arguments and tool results —
for high-confidence secret formats (API keys, AWS/GitHub/Slack/Stripe tokens, private
keys, JWTs, bearer tokens, URLs with embedded passwords, `password=`/`api_key:`-style
assignments) and refuses the write with no bypass flag. Long-term memory is not a safe
place for secrets; store a reference instead.

Recalled memories are returned marked as stored data, and the server instructions tell
the model never to follow directions found inside them.

Found a vulnerability? Please report it privately — see the
[security policy](https://github.com/EverMind-AI/EverOS-MCP/blob/main/SECURITY.md).

## Remote server (streamable HTTP)

The same package runs as a shared, hosted MCP server. It holds no credentials of its
own: every request brings the caller's EverOS API key, which is forwarded to the EverOS
API and never stored.

```bash
everos-mcp --transport http --host 0.0.0.0 --port 8765
```

Clients connect with their own key:

```bash
claude mcp add --transport http everos https://mcp.example.com/mcp \
  --header "Authorization: Bearer sk-..."
```

<details>
<summary><b>Request headers, server settings, and deployment notes</b></summary>

<br>

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
| `EVEROS_MCP_INTROSPECTION_URL` / `EVEROS_MCP_INTROSPECTION_SECRET` | — | OAuth mode (both required, together with the authorization server): bearer tokens are verified at this RFC 7662 endpoint (audience must be this server) and exchanged for the EverOS API key the user granted. The token itself is never forwarded upstream, as the MCP authorization spec requires. Contract: [`oauth.py`](https://github.com/EverMind-AI/EverOS-MCP/blob/main/src/everos_mcp/oauth.py) |

Deployment notes:

- Terminate TLS at the ingress; `GET /healthz` is the liveness probe.
- Each MCP session lives in the memory of the replica that created it. With more than
  one replica, route by the `Mcp-Session-Id` header (sticky sessions).
- An API key is the trust boundary: anyone holding a key can read and write every user id
  within that key's space (`X-EverOS-User-Id` is chosen by the caller). Give separate
  people separate keys when that matters.
- Conversations are isolated by (API key, user, MCP session): one caller never sees
  another's session, background-save notes, or trajectories.
- Hosts that only connect through OAuth (claude.ai connectors, ChatGPT) need an
  authorization server; set `EVEROS_MCP_AUTHORIZATION_SERVER` once one exists.

</details>

## Development

```bash
uv sync --dev
uv run ruff check . && uv run pytest      # offline: wire contract, tools, guard
EVEROS_API_KEY=... EVEROS_USER_ID=... python scripts/smoke_test.py   # live e2e
```

<details>
<summary><b>Releasing</b></summary>

<br>

Bump `version` in `pyproject.toml` and `__version__` in `src/everos_mcp/__init__.py`,
merge, then push a matching tag:

```bash
git tag v0.1.0 && git push origin v0.1.0
```

`.github/workflows/release.yml` tests, builds and smoke-tests the wheel, waits for
approval on the `release` environment, publishes to PyPI through Trusted Publishing (no
stored token), and drafts the GitHub Release.

</details>

<p align="right"><a href="#readme-top">back to top</a></p>

## EverMind Ecosystem

EverMind connects memory research, production-ready products, and practical
integrations into one open-source ecosystem.

<table>
<tr>
<th colspan="2">Products</th>
</tr>
<tr>
<td><strong><a href="https://github.com/EverMind-AI/EverOS">EverOS</a></strong></td>
<td>A local-first, Markdown-native long-term memory runtime for agents and users.</td>
</tr>
<tr>
<td><strong><a href="https://github.com/EverMind-AI/Raven">Raven</a></strong></td>
<td>A memory-first, self-improving agent harness with proactivity, context control, and skill evolution.</td>
</tr>
<tr>
<td><strong><a href="https://github.com/EverMind-AI/EverMe">EverMe (CLI)</a></strong></td>
<td>A CLI and agent plugin suite for cross-device, cross-agent personal memory.</td>
</tr>
<tr>
<th colspan="2">Research &amp; Evaluation</th>
</tr>
<tr>
<td><strong><a href="https://github.com/EverMind-AI/SkillCorpus">SkillCorpus</a></strong></td>
<td>Curated, retrieval-ready agent skill corpora with retrieval and evaluation tooling.</td>
</tr>
<tr>
<td><strong><a href="https://github.com/EverMind-AI/EverAlgo">EverAlgo</a></strong></td>
<td>Stateless extraction, ranking, parsing, and memory operators that power EverOS.</td>
</tr>
<tr>
<td><strong><a href="https://github.com/EverMind-AI/HyperMem">HyperMem</a></strong></td>
<td>Hypergraph-based hierarchical memory for coarse-to-fine long-term conversation retrieval.</td>
</tr>
<tr>
<td><strong><a href="https://github.com/EverMind-AI/MSA">MSA</a></strong></td>
<td>Memory Sparse Attention for scalable latent memory and 100M-token contexts.</td>
</tr>
<tr>
<td><strong><a href="https://github.com/EverMind-AI/EverMemBench">EverMemBench</a></strong></td>
<td>Evaluation of factual recall, applied reasoning, and personalized generalization in memory systems.</td>
</tr>
<tr>
<td><strong><a href="https://github.com/EverMind-AI/EvoAgentBench">EvoAgentBench</a></strong></td>
<td>Longitudinal evaluation of agent self-evolution, transfer efficiency, error avoidance, and skill use.</td>
</tr>
<tr>
<th colspan="2"><a href="https://github.com/EverMind-AI/plugins">Integrations</a></th>
</tr>
<tr>
<td><strong><a href="https://docs.openclaw.ai">OpenClaw</a></strong></td>
<td><a href="https://github.com/EverMind-AI/plugins/tree/main/openclaw">OpenClaw plugin</a> for automatic recall, capture, and session-memory lifecycle management.</td>
</tr>
<tr>
<td><strong><a href="https://github.com/NousResearch/hermes-agent">Hermes Agent</a></strong></td>
<td><a href="https://github.com/EverMind-AI/plugins/tree/main/hermes">Hermes plugin</a> for persistent memory across Hermes sessions.</td>
</tr>
<tr>
<td><strong><a href="https://github.com/deepseek-ai/DeepSeek-Harness">DeepSeek Harness</a></strong></td>
<td><a href="https://github.com/EverMind-AI/plugins/tree/main/dsh">DSH plugin</a> for memory-aware DeepSeek Harness agents.</td>
</tr>
<tr>
<td><strong><a href="https://dify.ai">Dify</a></strong></td>
<td><a href="https://github.com/EverMind-AI/plugins/tree/main/dify">Self-hosted</a> and <a href="https://github.com/EverMind-AI/plugins/tree/main/dify_cloud">cloud</a> tools for explicit memory search and storage in workflows and agents.</td>
</tr>
<tr>
<td><strong><a href="https://github.com/EverMind-AI/EverOS-MCP">MCP</a></strong></td>
<td>This server: EverOS memory for Claude Code, Claude Desktop, Cursor, Codex, and any MCP client.</td>
</tr>
</table>

Together, these projects form EverMind's research-to-runtime stack: methods
and benchmarks become reusable memory infrastructure, products, and agent
integrations.

## License

[MIT](https://github.com/EverMind-AI/EverOS-MCP/blob/main/LICENSE)
