# PPMLX Threat Model

This document covers the shipped local product and the accepted remote-gateway architecture.
It uses three control states:

- **Implemented:** The control exists in the current code path.
- **Partial:** Some controls exist, but a documented gap remains.
- **Planned:** The control is required before the related feature can ship.

A planned control is not a security claim.

## Security objectives

PPMLX must:

- Keep local inference local unless an explicit remote route applies.
- Keep tool execution and approval in the harness.
- Preserve tool names, arguments, identifiers, and result links after the accepted normalization boundary.
- Prevent secrets from entering configuration snapshots, analytics, logs, traces, packages, or release artifacts.
- Treat model output, tool output, memory, and caller input as untrusted.
- Fail closed when a required protocol or capability cannot be represented safely.
- Keep release artifacts reproducible between TestPyPI and PyPI.

## Trust boundaries

1. **Harness to local endpoint.** Claude Code, Codex, OpenCode, Pi, SDKs, and other local processes send requests to PPMLX.
2. **Gateway to local model runtime.** PPMLX converts validated requests into MLX engine input and receives untrusted model output.
3. **Gateway to local durable storage.** Request logs, memory, aliases, configuration, model files, and caches live under the user's account.
4. **Gateway to model registries.** Model discovery and downloads use external services such as Hugging Face.
5. **Gateway to future remote providers.** A future explicit route will send approved request data to a configured provider.
6. **Harness tool boundary.** PPMLX transports tool calls and results, but the harness owns permission and execution.
7. **Build and release boundary.** GitHub Actions, PyPI, TestPyPI, and the Homebrew tap handle release artifacts and metadata.

## Threats and controls

| Threat | State | Current controls | Remaining requirement |
|---|---|---|---|
| Untrusted local process calls the endpoint | Partial | Loopback is the default bind address. Request bodies and tool data have size limits. Strict Agent IR tool traffic uses bounded scope identifiers. | Add a gateway token before remote providers or privileged non-loopback use. Define principal separation for multiple local clients. |
| Server is exposed beyond loopback | Partial | Non-loopback binding is explicit. Strict Agent IR tool traffic rejects non-loopback callers. CORS defaults to localhost origins. | Require a gateway token for every non-loopback bind. Document TLS termination and firewall requirements. Review legacy endpoints before claiming remote-safe operation. |
| SSRF through images, provider URLs, or schemas | Partial | API vision requests reject remote image URLs, `file://` URLs, and bare local paths. Tool-schema validation rejects external references in the strict runtime. | Read provider base URLs only from trusted configuration. Keep TLS verification on. Add an explicit provider-host policy and SSRF tests before remote adapters ship. |
| Local file disclosure through multimodal input | Implemented for API path | The API accepts image data URLs but rejects local path references. The CLI can read a path only as an explicit interactive action. | Keep this distinction in tests and documentation when new multimodal protocols are added. |
| Provider or registry credential theft | Partial | Analytics and strict-runtime errors use allowlists and safe codes. Provider authentication is not shipped. | Store provider secrets in macOS Keychain. Keep only secret references in config. Remove secrets from exceptions, logs, traces, snapshots, and support bundles. Do not read another application's token cache. |
| Hugging Face token exposure | Partial | The token is used only for model-registry operations. | Replace plain configuration storage with a secret reference or Keychain integration. Until then, treat config and environment files as sensitive. |
| Prompt injection changes tool behavior | Partial | The harness remains the only tool executor. Tool results keep their role as untrusted model input. Strict normalization validates tool names, schemas, and call links. | Add provider-route disclosure policy, stronger prompt-injection tests, and canonical event provenance. A model response must never grant tool permission. |
| Malicious tool output changes instruction priority | Partial | Tool output remains a tool result. The harness controls execution and approval. Continuation links require stable call identity. | Apply route policy before sending a result to a remote provider. Add content-size and redaction tests for each provider path. |
| Tool-call parser invents intent | Partial | The strict Agent IR runtime uses explicit model profiles, strict parsing, a one-repair budget, schema validation, and safe errors. All shipped profiles remain repair-disabled until evidence exists. | Retire or clearly isolate compatibility parsing. Enable repair only for exact evaluated profiles. Keep family-name matches from creating capability claims. |
| Event or prompt data leaks through analytics | Implemented for current analytics contract | Analytics is disabled by default and accepts only allowlisted product/runtime metadata, counters, booleans, and safe labels. | Keep tests synchronized with every new event. Never add prompts, responses, raw errors, tool data, model output, project IDs, or session IDs. |
| Sensitive content persists in local logs | Partial | Ordinary request logging stores operational metadata rather than complete messages or responses. | Add unified retention controls, deletion commands, database permissions checks, and migration tests. |
| Memory stores secrets or poisoned facts | Partial | Memory is off by default. Existing code scopes data by project/session, rejects known secret patterns, tracks provenance in parts of the pipeline, and treats retrieved content as context. | Complete canonical events, durable outbox, explicit evidence rules, raw-event retention, redaction before persistence, feedback-loop prevention, and migration from 0.5.8. |
| Memory crosses project or remote-disclosure boundaries | Partial | Existing tests cover project namespace isolation. Remote routing is not shipped. | Add `local_only`, `remote_allowed`, and `secret` labels. Test every fallback and remote-memory path. Automatic injection must remain opt-in. |
| Raw model reasoning enters logs or memory | Partial | Ordinary request records use counts and timing rather than raw reasoning. The Agent IR treats reasoning as structured content. | Enforce redaction and retention rules in canonical events and every provider adapter. Do not project opaque reasoning into durable facts. |
| Denial of service through large input or output | Partial | HTTP body, token, tool-call, JSON byte, depth, node, string, and call-count limits exist in strict paths. Model memory checks and loaded-model limits reduce local pressure. | Add total event-stream limits to all new providers, cancellation tests, rate-limit handling, and configurable retention backpressure. |
| Dependency or action compromise | Partial | External GitHub Actions used by current release workflows are pinned to commit hashes. The release path checks artifacts and reuses them between repositories. | Add dependency scanning, secret scanning, and a documented response process. Review pinned actions on a schedule. |
| Release artifact substitution | Implemented for the current release workflow | The workflow builds once, checks the wheel and source archive, records digests, and uploads the same files to TestPyPI and PyPI. Homebrew uses PyPI source metadata. | Keep manual PyPI approval and compare published digests for every release. Add Apple Silicon MLX smoke evidence before final release. |
| Provider outage causes unsafe fallback | Planned | Remote providers and routing are not shipped. | Fallback must be explicit, deterministic, privacy-checked, and disabled from local to remote by default. Never change route after output or a tool call starts. |

## Current local endpoint limits

The default endpoint is a single-user local service.
Loopback does not authenticate one local process from another local process under the same account.
Malware or another process with user-level access can call the endpoint and read files that the user can read through an approved harness tool.

Do not bind PPMLX to `0.0.0.0` or a LAN address as if the current local API were a hardened multi-user service.
Use a separate authenticated reverse proxy and network policy only after reviewing the relevant endpoint and compatibility mode.

## Strict runtime and compatibility mode

The strict Agent IR runtime provides the intended security boundary for structured local tool use.
It preserves call identity, validates schemas, uses bounded parsers, and keeps execution in the harness.

Compatibility code remains in the product for older local clients.
Do not assume that compatibility parsing has the same guarantees as the strict runtime.
New provider and routing work must use Agent IR and adapter boundaries rather than extend legacy text parsing.

## Secret handling rules

- Never put provider credentials in a model alias, URL, error message, trace, snapshot, fixture, or package file.
- Never copy a token from another CLI, browser profile, or cache.
- Use environment variables only for CI or explicit temporary operation.
- Use macOS Keychain for shipped provider credentials.
- Keep TLS verification enabled by default.
- Redact authorization headers and secret-like values before persistence.
- Treat hashes of low-entropy secrets as sensitive identifiers.

## Tool and model rules

- A tool definition is not permission to run a tool.
- A tool call is an intention, not evidence that an action succeeded.
- A successful tool result is evidence of the result.
- A failed tool result is evidence of failure.
- PPMLX must reject an unknown tool, invalid schema, broken result link, changed call ID, or unsupported required field.
- A repaired syntax value must pass the same checks as a strict value.
- A capability profile must identify an exact evaluated model and tokenizer revision before publication.

## Memory rules

- Retrieved memory is untrusted content.
- Memory must not increase instruction priority.
- Memory-origin content must not create a duplicate of itself.
- A tool result has stronger evidence weight than an assistant claim.
- Project and session scope must be explicit.
- Remote routes must not receive restricted memory.
- Memory capture, read, and compaction controls must remain separate.

## Required security gates before remote providers

Remote-provider work cannot pass its phase gate until PPMLX has:

1. A provider interface with safe error and cancellation behavior.
2. Keychain-backed authentication and secret references.
3. A gateway token for remote-provider use and non-loopback binding.
4. Trusted provider URL configuration and TLS verification.
5. Route-level disclosure checks and explicit fallback.
6. Request, response, event, and retention limits.
7. Secret and dependency scans in CI.
8. Provider contract tests for rate limits, cancellation, malformed responses, and redaction.
9. Privacy documentation that matches the implemented route.

## Review and maintenance

Update this threat model when a pull request changes:

- An endpoint or trust boundary.
- Authentication or secret storage.
- Provider or registry network access.
- Tool parsing, execution, or continuation state.
- Memory capture, retrieval, compaction, or retention.
- Analytics or local diagnostics.
- Build, release, or package distribution.

A review must compare the document with shipped behavior.
A planned control must not be relabeled as implemented without code and tests.
