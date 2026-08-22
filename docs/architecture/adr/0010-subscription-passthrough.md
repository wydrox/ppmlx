# ADR 0010: Subscription Passthrough

- Status: Accepted. This decision reverses the earlier rule that
  subscription authentication is out of scope.

## Context

Users want to mix models. They want Claude models through their Claude
subscription and other models (local MLX or remote providers) through
ppmlx, from the same harness session.

A harness custom-model setup that points ALL traffic at ppmlx removes the
direct path to the subscription. Without passthrough, an onboarded user
loses their subscription the moment they onboard. The user must switch
model definitions by hand, which is error-prone.

## Rejected alternatives

- Out of scope: ppmlx never touches subscription credentials. This was the
  earlier decision. Product review reversed it: it breaks the mixing-models
  workflow for every onboarded user.
- Fingerprint cloning: passthrough plus faithful reproduction of the
  official client's identity so the provider cannot distinguish the proxy
  traffic. REJECTED as detection evasion. ppmlx has no fingerprint cloning
  code.
- Partial tunnel: proxy only `messages` and skip telemetry or other
  endpoints. REJECTED. A partial tunnel breaks fidelity and can corrupt
  the Claude Code session state.

## Decision

ppmlx provides a full transparent tunnel for subscription models.

1. **Full tunnel, never partial.** ppmlx proxies ALL Anthropic endpoints
   that Claude Code uses: `messages` (including SSE streaming), `complete`,
   and telemetry. Forwarding is byte-faithful for request headers, request
   body, and response stream. Headers such as `user-agent`, `x-app`,
   `anthropic-beta`, and `x-stainless-*` are forwarded as received from
   Claude Code.

2. **Credentials are live-read, never copied.** ppmlx reads the
   credentials of the installed, logged-in Claude Code from `~/.claude`.
   It stores nothing in its own keychain or config. When Claude Code
   rotates credentials, ppmlx picks up the new value on the next read.
   Prerequisite: Claude Code must be installed AND logged in. This is the
   documented prerequisite for the feature.

3. **Memory capture with redaction.** Full request and response content
   flows into the memory pipeline with standard secret redaction (see
   ADR 0007). Provenance source is tagged `anthropic-subscription`.

4. **Default locked behind `[dangerous]`.**
   `[dangerous] subscription_passthrough` defaults to `false`. Enabling it
   prints a terms-of-service warning. Using a subscription outside Claude
   Code through this path is BLOCKED BY DEFAULT: the tunnel serves the
   user's own Claude Code session, not other harnesses.

## Security and privacy

Subscription credentials stay inside the Claude Code installation. ppmlx
reads them at request time and never writes them to its own keychain,
config, or logs. Full request and response content enters the memory
pipeline, so users must know that subscription turns are captured like any
other turn, with standard secret redaction applied (see ADR 0007).

## Compatibility effects

- Onboarded users keep their subscription models through the same ppmlx
  base URL that serves their other models. No manual switching.
- ppmlx now depends on the Claude Code credential file format under
  `~/.claude`. If that format changes, ppmlx must update its reader.
- Telemetry forwarding to Anthropic is deliberate. Fidelity of the tunnel
  takes priority over trimming telemetry.
- Harnesses other than Claude Code cannot use the tunnel unless the user
  changes the default-blocked setting.

## Consequences

- Proxying subscription traffic may breach provider terms of service up to
  account termination. The risk belongs to the user. Documentation states
  this warning wherever subscriptions are discussed.
- The mixing-models workflow works from one harness session without manual
  configuration switches.
