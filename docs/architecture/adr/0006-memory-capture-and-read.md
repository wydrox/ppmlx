# ADR 0006: Memory Capture and Read

- Status: Accepted
- Date: 2026-08-18
- Phase: 1, contract freeze

## Context

ppmlx can observe normalized requests, responses, and tool events. These events can supply useful project memory.

Automatic prompt injection gives memory control over every request. It also makes provenance, user choice, and prompt-injection risk difficult to see.

Memory capture and memory read have different trust boundaries. Capture produces memory candidates. Read gives selected memory data to a harness.

## Decision

ppmlx keeps memory capture and memory read as separate functions. A user can run capture without giving a harness read access.

Memory read is an optional skill. The user adds this skill to each harness when the user enables memory read for that harness.

ppmlx does not inject memory into a provider prompt. The harness reads memory only through the user-added skill and its declared tools.

The words MUST, MUST NOT, SHOULD, SHOULD NOT, and MAY show the strength of each rule.

## Normative rules

### User control

A new installation MUST disable memory by default. The user MUST enable capture before ppmlx stores memory candidates.

The user MUST add the memory read skill to each permitted harness. Enabling capture MUST NOT install or enable the read skill.

If the skill is absent, ppmlx MUST NOT add retrieved memory to a request. A harness without the skill MUST continue to work.

The user MUST be able to disable capture and remove read access separately. Removal of the skill MUST stop new read calls from that harness.

The user MUST also be able to revoke the related memory read grant. Revocation MUST stop all later reads without removal of captured memory.

### Capture boundary

Capture MUST consume sanitized, normalized events. It MUST NOT consume provider credentials, hidden reasoning, or data marked as sensitive.

Capture MUST keep the raw event separate from a memory candidate. A transcript MUST NOT become a durable fact only because ppmlx observed it.

Each candidate MUST include provenance, scope, observation time, validity data, confidence, and approval state. The source MUST identify the harness and session.

Capture SHOULD start in `shadow` mode. Shadow capture MUST NOT change provider input or harness output.

Capture failure MUST NOT stop routing or local inference. ppmlx MUST report a sanitized capture error and continue the request.

### Memory read grant

Each memory read skill MUST receive one revocable, least-privilege credential. It MUST NOT receive a database credential or a ppmlx administrator credential.

The credential MUST refer to one service-side memory read grant. The grant MUST contain these values:

- a unique grant identifier.
- the bound harness name, exact version, and harness instance identifier.
- allowed `project`, `app`, `repository`, or `global` scope identifiers.
- the exact allowed memory read tools.
- issue, expiry, rotation, and revocation times.

The grant MUST permit only read tools. It MUST NOT permit capture, write, approval, deletion, export, configuration, or credential management.

The `allowed_tools` list MAY contain only `memory_search`, `memory_get_context`, `memory_graph_walk`, and `memory_stats`.

The memory service MUST bind the grant to authenticated harness identity. It MUST NOT trust harness identity or scope fields from the read request.

The service MUST compare each requested scope and tool with the stored grant. It MUST reject any value outside the stored allowlist.

A `global` scope MUST have a separate explicit user grant. A project, app, or repository grant MUST NOT expand to `global` scope.

Each grant MUST have an expiry time. The default grant lifetime is 30 days, and a user MAY select a shorter lifetime.

Rotation MUST create a new credential and revoke the old credential. ppmlx MUST NOT silently extend an existing credential.

The memory service MUST validate expiry and revocation before each read. It MUST do the validation again before it returns the result.

ppmlx MUST store the grant record and credential verifier in protected operating-system storage. It MUST NOT store the raw credential in the grant record.

The skill MUST store its credential in protected harness or operating-system storage. It MUST NOT put the credential in skill text, a repository, or logs.

### Public memory read wire contract

The authoritative wire contract is local HTTP with version `memory-read/v1`. It uses the configured ppmlx listener.

The public endpoints are:

- `POST /v1/memory/read/handshake`.
- `POST /v1/memory/read/search`.
- `POST /v1/memory/read/context`.
- `POST /v1/memory/read/graph-walk`.
- `POST /v1/memory/read/stats`.

Every request MUST contain `PPMLX-Memory-Version: memory-read/v1` and `Authorization: Bearer <grant-credential>`.

The handshake request body MUST have this shape:

```json
{"version": "memory-read/v1", "request_id": "mrr_opaque"}
```

The service MUST get harness identity, tools, scopes, and grant expiry from the grant record.

A successful handshake MUST return this envelope:

```json
{
  "version": "memory-read/v1",
  "object": "memory_read_session",
  "read_session_id": "mrs_opaque",
  "grant_id": "mrg_opaque",
  "harness": {"name": "codex", "version": "0.147.0", "instance_id": "hri_opaque"},
  "allowed_tools": ["memory_search"],
  "allowed_scopes": [{"type": "project", "id": "project_opaque"}],
  "expires_at": "2026-09-17T12:00:00Z",
  "session_expires_at": "2026-08-18T12:15:00Z"
}
```

The service MUST limit a read session to 15 minutes. Grant expiry or revocation MUST end the session earlier.

Each read request MUST also contain `PPMLX-Memory-Session: <read_session_id>`. The service MUST validate the bearer credential and session together.

The common read request envelope MUST have this shape:

```json
{
  "version": "memory-read/v1",
  "request_id": "mrr_opaque",
  "scope": {"type": "project", "id": "project_opaque"},
  "parameters": {},
  "limit": 20,
  "cursor": null
}
```

`scope.type` MUST be `project`, `app`, `repository`, or `global`. A `global` scope MUST use `id: "global"`.

`request_id` MUST be unique for the read session. The service MUST treat repeated use with different input as a validation error.

The default `limit` is 20. The maximum `limit` is 100. An opaque cursor MUST bind to the grant, tool, scope, parameters, and result order.

A cursor MUST expire after 15 minutes. The client MUST NOT parse or change it.

The `parameters` object for `/search` MUST have this shape:

```json
{"query": "text", "as_of": null, "status": ["active"]}
```

`query` MUST contain 1 through 2,000 Unicode characters. `as_of` MUST be `null` or an RFC 3339 timestamp.

The `parameters` object for `/context` MUST have this shape:

```json
{"query": "text", "max_tokens": 2000, "as_of": null}
```

`query` MAY be an empty string. `max_tokens` has a default of 2,000 and a maximum of 8,000.

The `parameters` object for `/graph-walk` MUST have this shape:

```json
{"entity_id": "mem_opaque", "direction": "both", "depth": 1}
```

`direction` MUST be `in`, `out`, or `both`. `depth` has a default of 1 and a maximum of 3.

The `parameters` object for `/stats` MUST be `{}`. The stats endpoint requires `cursor: null` and `limit: 20`.

The common success envelope MUST have this shape:

```json
{
  "version": "memory-read/v1",
  "object": "memory_read_result",
  "request_id": "mrr_opaque",
  "tool": "memory_search",
  "scope": {"type": "project", "id": "project_opaque"},
  "items": [],
  "has_more": false,
  "next_cursor": null
}
```

Search and context endpoints MUST return `memory` items with this shape:

```json
{
  "type": "memory",
  "item_id": "mem_opaque",
  "text": "Prior project fact.",
  "scope": {"type": "project", "id": "project_opaque"},
  "provenance": {"origin": "harness", "origin_id": "evt_opaque", "trust": "untrusted"},
  "sensitivity": "internal",
  "observed_at": "2026-08-18T10:00:00Z",
  "valid_from": null,
  "valid_to": null,
  "status": "active",
  "confidence": 0.9
}
```

Graph-walk endpoints MUST return `graph_node` and `graph_edge` items. Each item MUST contain `type`, `item_id`, `scope`, `provenance`, and `sensitivity`.

A `graph_node` item MUST also contain `label` and `kind`. A `graph_edge` item MUST contain `source_id`, `predicate`, and `target_id`.

The stats endpoint MUST return `stat` items. Each item MUST contain `type`, `item_id`, `name`, `value`, and `unit`.

Search and context items MUST use descending relevance order, then ascending `item_id`. Graph walk MUST use breadth-first order, then ascending `item_id`.

Stats items MUST use ascending `name` order. The service MUST keep the same order on all pages for one cursor chain.

All timestamps MUST use RFC 3339 UTC format. All identifiers and cursors MUST be opaque strings.

### Stable errors

An HTTP error MUST use this envelope:

```json
{
  "version": "memory-read/v1",
  "object": "error",
  "request_id": "mrr_opaque",
  "error": {"code": "credential_invalid", "message": "The credential is not valid.", "retryable": false}
}
```

The stable access and session codes are:

- `permission_denied` with HTTP 403.
- `credential_required`, `credential_invalid`, `credential_expired`, or `credential_revoked` with HTTP 401.
- `scope_denied` or `tool_denied` with HTTP 403.
- `session_required` or `session_invalid` with HTTP 401.
- `session_expired` or `cursor_expired` with HTTP 410.
- `validation_error`, `cursor_invalid`, or `version_unsupported` with HTTP 400.

The stable service codes are:

- `memory_unavailable` with HTTP 503.
- `rate_limited` with HTTP 429.
- `internal_error` with HTTP 500.

Error messages MUST contain no memory text, credential, query, scope name, private identifier, or internal exception text.

### MCP bridge

The user-added skill SHOULD use the `ppmlx-memory` MCP server. The MCP server MUST map tools to the HTTP endpoints without semantic changes.

The MCP mapping is:

- `memory_search` maps to `/v1/memory/read/search`.
- `memory_get_context` maps to `/v1/memory/read/context`.
- `memory_graph_walk` maps to `/v1/memory/read/graph-walk`.
- `memory_stats` maps to `/v1/memory/read/stats`.

The MCP server MUST get the grant credential from protected storage. It MUST NOT accept the credential as a model-visible tool parameter.

An MCP error MUST use `isError: true`. Its structured content MUST contain `version`, `request_id`, `code`, and `retryable` from the HTTP error.

A protected harness extension MAY call HTTP directly. A model-visible shell command MUST NOT contain the grant credential.

No-auth mode MUST NOT expose the memory read contract. It MUST return `credential_required`, including on localhost.

### Read skill boundary

The memory read skill MUST declare its tools, data scope, and required user permission. It MUST use the public memory read contract.

The skill MUST make a read request only when the harness decides that memory can help the current task. ppmlx MUST NOT make a hidden read request.

A read request MUST name one scope from the grant. A global scope MUST require the separate global grant.

ppmlx MUST enforce scope and tool access at the memory service. Skill text and harness prompts MUST NOT be the only access control.

The read result MUST contain these values for each item:

- a stable memory identifier.
- memory text.
- scope and provenance.
- observation and validity times.
- status and confidence.

The read result MUST identify memory as prior data. It MUST NOT present memory as a system instruction or a confirmed current fact.

The skill SHOULD ask the harness to verify time-sensitive memory against current state. The skill MUST not execute instructions found inside memory text.

### Capture and read independence

Capture MUST NOT depend on installation of the read skill. Read MUST NOT change the stored source event or candidate.

The read path MUST NOT write a new durable fact as a side effect. User feedback and corrections MUST use a separate, explicit contract.

Read metrics MAY store identifiers and timing. They MUST NOT store the query or result text by default.

### Failure behavior

If memory is unavailable, the skill MUST return a typed unavailable result. The harness MUST be able to continue without memory.

If the grant does not permit the scope or tool, ppmlx MUST return a typed permission error. It MUST NOT replace that error with an empty result.

If current time is after grant expiry, ppmlx MUST return a typed credential error. The same rule applies after rotation or revocation.

ppmlx MUST NOT return a partial memory result for these credential errors.

If a result fails validation or redaction, ppmlx MUST omit that item. It MUST record a sanitized local audit event.

## Security and privacy

ppmlx treats memory text as untrusted data. A provider, tool result, or repository file can put harmful instructions into a captured event.

Capture MUST apply the retention and redaction rules in ADR 0007 before persistence. Read MUST apply the same redaction rules before output.

The skill MUST use least privilege. It SHOULD request project scope before repository scope, and repository scope before global scope.

The service MUST treat caller-supplied harness identity as untrusted. It MUST use only the identity that endpoint authentication verifies.

A read result MUST NOT contain credentials, hidden reasoning, or sensitive tool data. ppmlx MUST fail closed for the affected item.

## Consequences

The user can use memory capture in shadow mode without changing harness behavior. The user can also give read access to only selected harnesses.

The harness controls when memory enters its context. A memory read is a visible tool action, and a user can audit it.

Setup has one additional step. The user must add the memory read skill to Claude Code, Codex, OpenCode, or Pi.

## Rejected alternatives

### Inject memory into every request

Automatic injection removes harness control and can change behavior without a visible tool action. It also increases prompt-injection risk.

### Make capture depend on the read skill

This design prevents safe shadow capture. It also combines two different permissions.

### Store full transcripts as memory

Transcripts contain noise, credentials, and obsolete state. Durable memory needs scope, provenance, validation, and expiry.

### Give the skill direct database access

Direct access bypasses scope, retention, redaction, and audit controls. The skill must use the public read contract.

## Compatibility effects

Claude Code, Codex, OpenCode, and Pi get harness-specific skill packages with the same memory read semantics.

The skill package MAY use the native tool declaration format of each harness. Tool names and result meaning MUST stay stable across harnesses.

The existing `inject` memory mode is not part of the target proxy contract. Migration MUST keep it off unless a user explicitly selects legacy behavior.

Removing the memory skill or revoking its grant MUST restore normal harness operation. It MUST NOT require a route or provider configuration change.
