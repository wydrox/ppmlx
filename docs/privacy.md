# PPMLX Privacy and Data Paths

This document describes the shipped local product and the accepted remote-provider architecture.
It does not claim that unfinished provider, routing, or authentication features exist.

## Summary

PPMLX is local-first.
A local inference request stays on the Mac when it uses an MLX model.
PPMLX can still use the network for model discovery, model downloads, optional analytics, release checks, and other explicit network features.

Remote model routing is not shipped in PPMLX 0.9.1.
When remote routing is added, the selected provider will receive the request data required for that route.
The provider's own privacy and retention terms will then apply.

## Data classes

PPMLX can process these data classes:

- Instructions, messages, prompts, and model responses.
- Tool definitions, tool-call arguments, and tool results.
- Images supplied as data URLs.
- Model names, aliases, revisions, and local model files.
- Request identifiers, token counts, timings, and error codes.
- Project, session, and harness scope metadata.
- Memory events, candidates, facts, provenance, and graph data.
- Authentication data for model registries or future providers.

Treat prompts, tool data, memory, and configuration files as sensitive.
They can contain source code, file contents, personal data, and secrets.

## Shipped data paths

| Path | Activation | Data destination | Content sent or stored |
|---|---|---|---|
| Local MLX text, vision, or embedding inference | A local model handles the request | Local process memory and Apple Silicon model runtime | Request content needed for inference and the generated result |
| Model discovery and download | Registry refresh, `ppmlx pull`, or loading a model that is not local | Hugging Face services and local model storage | Model repository identifiers, download requests, network metadata, and an optional Hugging Face token |
| Optional analytics | The user enables analytics | Configured PostHog-compatible analytics endpoint | Anonymous installation identifier, PPMLX/runtime/platform metadata, numeric counters, booleans, and allowlisted event labels |
| Local request log | Logging is enabled | Local SQLite database under `~/.ppmlx/` | Request ID, endpoint, model metadata, stream flag, token counts, timing, status, safe error data, and message count |
| Memory capture and compaction | Memory mode is `shadow`, `compact`, or `inject` | Local memory database and local inference pipeline | Messages, response text, extracted candidates, facts, events, provenance, and graph projections |
| Local aliases and preferences | The user changes aliases or favorites | Files under `~/.ppmlx/` | Model aliases, repository identifiers, favorites, and configuration values |

The local request log does not intentionally store complete message bodies or complete responses.
The local memory system can store content derived from messages and responses when memory capture is enabled.
Memory mode is `off` by default.

## Images and local files

The API accepts image data URLs for local vision inference.
It rejects remote image URLs.
It also rejects `file://` URLs and bare local file paths in API requests.

The interactive CLI can permit a user-selected local image path.
This is an explicit local action and does not make the API a general file reader.

## Analytics

Analytics is disabled by default.
When enabled, PPMLX uses a generated installation identifier.
The analytics allowlist excludes prompts, responses, tool arguments, tool results, model output, raw errors, file paths, project IDs, session IDs, and provider credentials.

Some events contain counts or Boolean indicators such as token totals, latency, whether a project scope exists, or whether a feature is enabled.
Disabling analytics stops these events.

## Request logging

Request logging is local and enabled by default.
It records operational metadata for troubleshooting and performance analysis.
It does not send the log database to an analytics provider.

The current release does not provide one unified retention policy for all request, memory, and graph data.
Local records can remain until the user removes or migrates the relevant files under `~/.ppmlx/`.
Back up the directory before manual deletion.

## Memory

Memory is separate from the ordinary request log.
The default memory mode is `off`.

When memory capture is enabled, PPMLX can process and persist message and response content locally.
The memory code applies project and session scopes and rejects known secret patterns, but no secret detector is complete.
Do not enable memory capture for data that must never enter durable local storage.

Retrieved memory is untrusted context.
It must not grant tool permission or override higher-priority instructions.
The planned canonical-event migration will add stronger evidence, provenance, retention, and feedback-loop controls.

## Credentials

The current local product can read a Hugging Face token from an explicit argument, configuration, or the `HF_TOKEN` environment variable.
Treat `~/.ppmlx/config.toml` and environment files as secrets when they contain a token.

Remote-provider authentication is not shipped.
The accepted design requires macOS Keychain storage for provider secrets, secret references in configuration, environment variables for CI, and redaction from logs and traces.
PPMLX will not copy credentials from another CLI, browser profile, or token cache.

## Raw reasoning

PPMLX can process model reasoning or hidden-thinking output during generation.
Operational records can include reasoning token counts, character counts, or timing.
PPMLX does not intentionally store raw model reasoning in analytics or the ordinary request log.

A provider-specific reasoning block remains subject to its protocol and retention policy.
The future router must not inspect opaque reasoning to make a route decision.

## Future remote-provider path

Remote routing remains optional.
The accepted contract requires an explicit provider and route policy.
A future remote route can send:

- Instructions and messages required by the selected provider protocol.
- Tool definitions and accepted tool-call or tool-result data.
- Images when the provider and route permit images.
- Structured memory blocks only when their disclosure label permits the route.
- Generation settings and safe request metadata.

The future router must not:

- Enable local-to-remote fallback by default.
- Send `local_only` or `secret` memory to a remote provider.
- Change providers after output starts or after the first tool call.
- Reuse a token from another application.
- Store raw model reasoning as project memory.

## User responsibilities

- Keep the server on loopback unless you have a separate authentication and network-control design.
- Protect `~/.ppmlx/`, shell history, environment files, and backups.
- Review analytics and memory settings before processing sensitive data.
- Use test credentials in development.
- Do not assume that a local model or a future remote provider has reliable tool use without profile evidence.

## Changes to this document

Documentation must follow shipped behavior.
A pull request that changes a data path, retention rule, analytics field, authentication method, or remote route must update this document and its contract tests.
