# ADR 0001: Product Boundary

- Status: Accepted
- Date: 2026-08-18

## Context

ppmlx currently provides local inference through an OpenAI-compatible API.
The proxy direction adds remote providers and agent harnesses without removing local inference.

Each harness can use a different API protocol.
Each provider can also use a different API protocol, authentication method, and tool format.
A clear boundary prevents protocol code, credentials, and tool execution from mixing.

## Decision

ppmlx is one local model endpoint and request router.
The user configures each harness to use the same ppmlx base URL.
ppmlx can route each request to local MLX inference or an approved remote provider.

ppmlx owns protocol adaptation, Agent IR normalization, route selection, provider transport, and response translation.
It also owns route metadata, safe diagnostics, and provider capability checks.

The harness owns its tools, its working directory, user approval, and tool execution.
The provider owns model inference and provider-side usage records.

## Normative rules

### Product surface

- ppmlx MUST keep local MLX inference as a first-class provider.
- ppmlx MUST use one configured listener and one local base URL.
- The base URL MAY expose multiple HTTP paths for supported protocols.
- An ingress adapter MUST map a supported request to the Agent IR.
- An egress adapter MUST map the Agent IR to a selected provider protocol.

### Mapping and routes

- An adapter MUST preserve information that affects meaning, order, identity, or tool state.
- ppmlx MUST reject an unsupported mapping before it sends a request.
- ppmlx MUST NOT silently remove a tool, instruction, content block, or required option.
- ppmlx MUST NOT run a harness tool or infer that user approval exists.
- Each route MUST refer to a model alias.
- A remote route MUST also refer to an authentication profile, and the route MUST NOT contain a secret.

### Related contracts and diagnostics

- Provider failure behavior MUST follow the routing contract in ADR 0005.
- Memory behavior MUST follow the memory contracts in ADR 0006 and ADR 0007.
- ppmlx MUST report the selected route and provider without exposing a credential.
- A remote route MUST NOT become active until its adapter reports the required capabilities.
- A compatibility claim MUST name the harness version and protocol fixture that supports the claim.

## Security and privacy

ppmlx is a local trust boundary, but a remote route sends request data outside the computer.
The route decision must make this transfer visible in diagnostics and policy checks.

ppmlx must not read a harness token store, browser cookie store, or unrelated application data.
It must not send a provider credential to a harness or to a different provider.

Logs must use redaction rules from ADR 0007.
The endpoint must bind to localhost by default.
External binding requires explicit user configuration and local endpoint authentication.

## Consequences

Users get one stable endpoint for local and remote models.
Harness configuration does not need a separate provider URL for each model.

Protocol adapters and providers become separate modules.
The system needs capability checks and clear errors for mappings that are not lossless.
Some provider features remain unavailable until the Agent IR and both adapters can preserve them.

## Compatibility effects

- **Claude Code:** The Anthropic Messages adapter terminates at ppmlx, and Claude Code still runs its tools and approval flow.
- **Codex:** The OpenAI Responses adapter terminates at ppmlx, and Codex still owns its sandbox and tool execution.
- **OpenCode:** The OpenAI-compatible adapter terminates at ppmlx, and OpenCode still owns its tools and sessions.
- **Pi:** The OpenAI-compatible adapter terminates at ppmlx, and Pi still owns its tools and sessions.

Each harness can select a ppmlx model alias.
The same alias can select local inference or a remote provider through routing configuration.

## Rejected alternatives

### Separate proxy port

This option keeps local inference and remote routing on different base URLs.
It increases harness configuration and creates two product surfaces.

### Provider-specific harness configuration

This option configures each harness directly for each provider.
It removes the stable local control point and prevents common policy.

### ppmlx executes all tools

This option moves sandbox policy and user approval away from the harness.
It duplicates mature harness behavior and increases local security risk.

### Best-effort protocol conversion

This option drops unsupported fields and continues the request.
It can change agent behavior without a clear failure, so ppmlx rejects it.
