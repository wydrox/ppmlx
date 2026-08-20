# Security Policy

## Supported versions

| Version | Security support |
|---|---|
| Latest PyPI release | Yes |
| Current `main` branch | Best effort until the next release |
| Older releases | No, unless a maintainer announces an exception |

Use the latest released version when you test or report a vulnerability.

## Report a vulnerability

**Do not open a public issue for a security vulnerability.**

Email **rafal@ppmlx.dev** with:

- The affected PPMLX version and commit, when known.
- A concise description of the vulnerability.
- Reproduction steps or a minimal proof of concept.
- The expected and actual behavior.
- The potential impact and affected data.
- Any temporary mitigation that you found.

Do not include live credentials, private source code, personal data, or destructive payloads unless the maintainer asks for them through a secure channel.

### Response targets

- **48 hours:** Acknowledge the report.
- **7 days:** Provide an initial assessment and severity.
- **30 days:** Provide a fix, mitigation, or status update for a confirmed issue.

These are targets, not a guarantee.

## Scope

This policy covers:

- The PPMLX CLI and local API server.
- Agent IR and protocol adapters.
- Local tool-output normalization and continuation state.
- Request logging, memory, analytics, and configuration handling.
- Model download and registry integration owned by PPMLX.
- Build, package, release, and Homebrew update workflows in this repository.

Report an upstream defect directly to its maintainer when PPMLX does not add the vulnerable behavior.
Relevant upstream projects include MLX, mlx-lm, mlx-vlm, Hugging Face libraries, model repositories, and provider services.
You can still notify PPMLX when an upstream issue affects the default configuration.

## Current security posture

PPMLX is a local-first, single-user tool.
The server binds to `127.0.0.1` by default.
Loopback limits network exposure, but it does not authenticate one local process from another process under the same user account.

Do not expose the current server on `0.0.0.0`, a LAN, or the public internet as if it were a hardened multi-user service.
Non-loopback binding is explicit and remains the user's responsibility until PPMLX ships the required gateway-token controls.

The strict Agent IR runtime keeps tool execution in the harness and applies bounded parsing, identity, schema, and continuation checks.
Compatibility paths remain available for existing local clients and do not automatically have the same guarantees.

Remote model providers, provider authentication commands, and deterministic routing are not shipped in PPMLX 0.9.1.
PPMLX does not support copying a login token from another CLI or browser profile.

Read the detailed documents:

- [Threat model](docs/security/threat-model.md)
- [Privacy and data paths](docs/privacy.md)
- [Architecture decisions](docs/architecture/README.md)

## Secrets

- Do not put credentials in model aliases, provider URLs, issue reports, fixtures, screenshots, logs, or command examples.
- Treat `~/.ppmlx/config.toml`, environment files, shell history, databases, and backups as sensitive.
- A Hugging Face token can currently come from an explicit argument, configuration, or `HF_TOKEN`.
- Future provider secrets must use macOS Keychain and configuration references.
- PPMLX will not read another application's token cache.
- TLS verification must remain enabled by default.

If a secret appears in a commit or release artifact, revoke it before you remove it from history.

## Local data

PPMLX can store operational request metadata under `~/.ppmlx/`.
Memory capture is off by default, but an enabled memory mode can store content derived from messages and responses.
A local database is sensitive even when it contains no provider credentials.

The current release does not provide one unified retention and deletion command for all logs, memory, and graph data.
Back up local state before manual deletion or migration.

## Release integrity

The release workflow builds the wheel and source archive once, verifies their contents, records their digests, and reuses the same files for TestPyPI and PyPI.
External GitHub Actions in the current release workflows are pinned to commit hashes.
The Homebrew update reads the released version and PyPI source-archive digest.

A release must not reuse a published version number.
Yank a defective version and publish a new patch version.

## Known security work

The following controls remain incomplete or planned:

- Gateway authentication for remote-provider use and non-loopback binding.
- macOS Keychain provider-secret storage.
- Trusted provider URL policy and complete SSRF tests.
- Unified retention and deletion controls.
- Canonical memory events, durable outbox, and feedback-loop prevention.
- Dependency and secret scanning in CI.
- Full remote-provider redaction, rate-limit, cancellation, and disclosure tests.

A planned control is not a security claim.
The [threat model](docs/security/threat-model.md) records the current status of each control.
