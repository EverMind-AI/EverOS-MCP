# Security Policy

## Supported Versions

Security fixes are applied to the latest release only; older versions do not
receive backports.

| Version | Supported |
|---------|-----------|
| `0.1.x` (current) | ✅ |

## Reporting a Vulnerability

**Please do not report security vulnerabilities through public GitHub issues,
discussions, or pull requests.**

Instead, either use GitHub's
[private vulnerability reporting](https://github.com/EverMind-AI/EverOS-MCP/security/advisories/new)
or email **evermind@shanda.com** with:

- A description of the vulnerability and its potential impact
- Steps to reproduce, or a proof-of-concept
- The affected version / commit
- Any suggested mitigation, if you have one

We will acknowledge your report within **5 business days**, keep you informed of
progress, and aim to ship a fix or mitigation before any public disclosure.
Reporters are credited in the advisory and the release notes unless you prefer
to remain anonymous.

## Published Advisories

Confirmed issues are published as GitHub Security Advisories, each carrying the
affected version ranges and the release that fixes them:

<https://github.com/EverMind-AI/EverOS-MCP/security/advisories>

## Scope

In scope: this MCP server — its tools, the credential guard, session and user
isolation in remote (HTTP) mode, and the OAuth token handling.

Out of scope: vulnerabilities in EverOS Cloud or self-hosted EverOS themselves
(report those to the [EverOS repository](https://github.com/EverMind-AI/EverOS/security)),
and in the MCP clients that connect to this server.
