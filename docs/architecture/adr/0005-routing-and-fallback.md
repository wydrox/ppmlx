# ADR 0005: Routing and Fallback

- Status: Accepted
- Date: 2026-08-18
- Phase: 1, contract freeze

## Context

ppmlx routes one public model name to a local model or an external provider. A harness must get the same result for the same policy input.

Provider features differ. A model can support text but not tools, images, structured output, or the required stream events.

Automatic fallback can hide errors. It can also repeat text or tool calls after a stream starts.

## Decision

ppmlx uses a deterministic, capability-aware route policy. The policy has an explicit fallback list for each public model and harness.

The router makes one route decision from a versioned policy snapshot. ppmlx records a sanitized decision record for each attempt.

The words MUST, MUST NOT, SHOULD, SHOULD NOT, and MAY show the strength of each rule.

## Normative rules

### Route input

The route input MUST contain these values:

- the public model name.
- the harness name and exact version.
- the incoming protocol.
- the required capabilities.
- the route policy version.
- the `health_snapshot_id`.

The route input MUST also contain the request and session correlation identifiers.

Required capabilities MUST come from the request and harness contract. They MUST NOT come from a guess based only on the model name.

The capability set MUST cover text, images, tools, parallel tool calls, structured output, stream events, input size, and output size.

### Health snapshot

The router MUST resolve one immutable health snapshot before candidate selection. The caller MUST NOT select or replace the snapshot.

The snapshot MUST contain these values:

- `health_snapshot_id`.
- `captured_at` and `expires_at`.
- the provider, authentication profile, and model tuple for each remote candidate.
- the local runner and model tuple for each local candidate.
- the `healthy`, `unhealthy`, or `unknown` state for each candidate.
- a sanitized reason category and `checked_at` value for each state.

The route policy MUST set `health_snapshot_max_age_ms`. The default and maximum permitted value is 30,000 milliseconds.

At decision time, snapshot age MUST not exceed this limit. Its `expires_at` value MUST occur after decision time.

Otherwise, ppmlx MUST return a typed route-health error.

The snapshot MUST contain one state for every candidate in the route. The router MUST treat `unhealthy` and `unknown` candidates as unavailable.

The router MUST NOT read a later health value during candidate selection. It MUST use only the snapshot named by `health_snapshot_id`.

### Candidate selection

Each route entry MUST contain an ordered candidate list. Each candidate MUST name the provider, upstream model, and capability profile.

Each remote candidate MUST name one authentication profile. A local candidate MUST NOT require an authentication profile.

The router MUST examine candidates in policy order. It MUST select the first `healthy` candidate that has all required capabilities.

The router MUST make the same decision for the same route input, policy snapshot, and health snapshot.

The router MUST reject an incompatible candidate before it sends the request. It MUST return the missing capabilities in a normalized error.

A local MLX model and an external model MUST use the same candidate contract. Local inference MUST remain a normal route target.

### Route stability

ppmlx MUST assign a route decision identifier before the first provider attempt. All attempts for that decision MUST use the same policy snapshot.

Before the first output event, fallback MAY select the next permitted candidate. The candidate that sends the first output event becomes the pinned candidate.

ppmlx MUST keep the pinned candidate for the complete tool round trip. The round trip ends only after the final answer or a terminal error.

ppmlx MUST NOT transfer a tool-call continuation, tool result, or final-answer request to another candidate.

If the pinned candidate fails during the round trip, ppmlx MUST return a terminal provider error. It MUST NOT use another candidate.

A later independent turn MUST get a new route decision. A turn is independent only after the harness resolves all prior tool calls.

Load, price, or latency MUST NOT change candidate order during one route decision. A later policy version MAY use these inputs in a reproducible rule.

### Fallback

Fallback MUST be opt-in for each route. The policy MUST list each fallback candidate and the permitted error categories.

ppmlx MAY use fallback for a connection failure, a timeout, an upstream unavailable error, or a server error. It MAY use rate-limit fallback only when policy permits it.

ppmlx MUST NOT use fallback for these errors:

- authentication, permission, account, or billing failure.
- invalid request data.
- a missing capability.
- a provider safety or policy refusal.
- a tool contract error.
- a user cancellation.

ppmlx MUST NOT use fallback after it sends the first output event to the harness. This rule prevents duplicate text and duplicate tool calls.

Each candidate MUST have one attempt unless policy specifies a retry. The policy MUST set a total attempt limit and a total time limit.

Fallback MUST NOT form a loop. ppmlx MUST NOT try the same provider, authentication profile, and model tuple twice in one fallback chain.

### Provider errors

An adapter MUST map each provider error to a normalized provider error. The error MUST keep these sanitized values when available:

- provider and upstream model.
- provider status and error code.
- provider request identifier.
- retry delay.
- retry classification.
- route decision and attempt numbers.

The normalized category MUST distinguish authentication, permission, billing, rate limit, invalid request, safety refusal, timeout, connection, and server errors.

ppmlx MUST preserve the provider category. It MUST NOT report every provider failure as an internal server error.

Error text MUST NOT contain credentials, authorization headers, request bodies, or sensitive tool data.

### Decision records

A decision record MUST state why ppmlx selected, skipped, or rejected each candidate. It MUST include the policy version and capability result.

The record MUST include `health_snapshot_id`, snapshot age, snapshot expiry, and the state used for each candidate.

A decision record MUST NOT contain request content or credential values. It SHOULD contain enough data to reproduce candidate selection.

## Security and privacy

Route policy can refer to an authentication profile by name. It MUST NOT contain the profile secret.

Logs and metrics MUST use the public model name or authentication profile name. They MUST NOT expose provider tokens or subscription data.

ppmlx treats provider error text as untrusted data. It MUST sanitize this text before storage or display.

## Consequences

Routing stays predictable, and fixtures can test it. A user can see why ppmlx used a provider or stopped a fallback chain.

Some requests fail early when no candidate has all capabilities. This is safer than a silent capability loss.

ppmlx cannot silently recover after a stream starts. The harness must show the terminal error and decide if it will retry.

## Rejected alternatives

### Select the cheapest or fastest live model

This selection changes with live measurements. It does not give a reproducible route without a recorded snapshot.

### Try every provider for every error

This action hides account and request errors. It can also increase cost and repeat side effects.

### Continue a failed stream on another model

This action can duplicate text, tool calls, and tool side effects. The harness cannot safely join the two streams.

### Give local inference a separate route system

Two route systems would have different capability and error behavior. The common candidate contract keeps local inference as a first-class target.

## Compatibility effects

Each harness fixture MUST produce the same route decision for the same policy snapshot and `health_snapshot_id`.

Protocol adapters MUST pass required capabilities to the router. They MUST return normalized route and provider errors in the native harness protocol.

Existing configurations without fallback keep one candidate. ppmlx MUST NOT add an implicit external fallback during migration.
