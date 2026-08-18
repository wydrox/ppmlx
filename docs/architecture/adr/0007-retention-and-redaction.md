# ADR 0007: Retention and Redaction

- Status: Accepted
- Date: 2026-08-18
- Phase: 1, contract freeze

## Context

The proxy can receive prompts, tool arguments, tool results, file content, and provider errors. This data can contain credentials and private information.

Logs, search indexes, embeddings, queues, and backups can make an accidental copy durable. Redaction after storage is too late.

Useful memory also becomes incorrect with time. ppmlx needs clear retention limits and complete deletion behavior.

## Decision

ppmlx applies redaction before any durable write. Raw request, response, and tool content has no retention by default.

ppmlx uses data classes with fixed default retention periods. A user can select a shorter period at any time.

Credentials and sensitive tool data never become memory. User approval does not override this rule.

The words MUST, MUST NOT, SHOULD, SHOULD NOT, and MAY show the strength of each rule.

## Normative rules

### Data classes and defaults

ppmlx MUST apply these default retention periods to transit and operational data:

| Data class | Default period | Stored content |
|---|---:|---|
| Raw request, response, and tool payload | 0 days | No durable copy |
| Sanitized route and error metadata | 30 days | Identifiers, timing, categories, and counts |
| Sanitized capture event | 30 days | Minimum source data for extraction and audit |

ppmlx MUST apply these default retention periods to memory and audit data:

| Data class | Default period | Stored content |
|---|---:|---|
| Unapproved memory candidate | 30 days | Redacted candidate and provenance |
| Approved memory item | 180 days | Redacted fact, scope, provenance, and validity |
| Security and deletion audit event | 90 days | Action metadata without content |
| Deletion tombstone | 30 days | Item identifier and deletion time |

An explicit earlier expiry for an item MUST override the default period. ppmlx MUST NOT return an expired or invalid item.

A user MAY select longer retention for approved memory. ppmlx MUST show the selected period and require an explicit configuration change.

No retention setting can permit storage of credentials, hidden reasoning, or sensitive tool data.

### Redaction before persistence

ppmlx MUST redact before it writes to a database, log, queue, cache, search index, embedding store, trace, metric label, or backup source.

ppmlx MUST detect and remove these credential classes at minimum:

- authorization headers and session cookies.
- API keys, access tokens, refresh tokens, and bearer tokens.
- passwords, passphrases, and one-time codes.
- private keys and signing secrets.
- provider subscription credentials.
- credentials in URLs, environment data, commands, and tool payloads.

ppmlx MUST redact a sensitive field by field name and by value pattern. A configured exact-value filter MAY add protection for known local secrets.

ppmlx MUST NOT store an unsalted hash of a credential. Such a hash can still identify a low-entropy secret.

### Agent IR classification and provenance

Content-bearing Agent IR objects MAY contain `sensitivity` and `provenance` from ADR 0002.

The `sensitivity` value MUST be `public`, `internal`, `confidential`, or `restricted`. An absent value MUST mean `restricted`.

The `provenance` object MUST contain `origin` and `trust`. It MAY contain `origin_id`.

The `origin` value MUST be `harness`, `provider`, `tool`, `memory`, `ppmlx`, or `unknown`.

The `trust` value MUST be `trusted`, `untrusted`, or `unknown`. Absent provenance MUST mean `{origin: "unknown", trust: "untrusted"}`.

The `trusted` value MUST NOT grant tool permission or increase instruction priority. Redaction and scope policy MUST still apply.

ppmlx MAY retain `public` and `internal` content when another retention rule permits it. It MUST still scan this content for credentials.

ppmlx MUST NOT retain `restricted` content. It MAY retain `confidential` content only under an explicit field permission.

A derived event or memory candidate MUST keep source provenance. It MUST inherit the most restrictive source sensitivity.

An `origin_id` MUST be a safe identifier. It MUST NOT contain a path, credential, request content, or personal value.

### Sensitive tool data

A tool adapter MUST classify its arguments, result, and content blocks with the Agent IR sensitivity and provenance fields.

If an adapter omits either field, ppmlx MUST apply the Agent IR fail-closed defaults. It MUST NOT infer a less restrictive class.

These tool data classes MUST be sensitive by default:

- keychain and credential-store output.
- environment variable values.
- authentication files and private keys.
- browser cookies and active session data.
- payment, health, legal, and private communication content.
- content outside the permitted project scope.

If markers are absent for a known sensitive source, ppmlx MUST skip persistence, indexing, embedding, logging, and memory capture.

ppmlx MUST fail closed for a keychain, credential store, environment source, authentication file, browser session, or private communication source.

A user MAY permit capture of a less-sensitive tool field. This permission MUST name the tool, field, scope, and retention period.

A field permission MUST NOT permit credential storage. The proxy MUST keep the credential exclusion rule.

### Failure behavior

If redaction cannot classify content safely, ppmlx MUST skip the durable write. It MUST continue the proxy request when possible.

ppmlx MUST record a sanitized redaction-failure count. It MUST NOT store the rejected content in the error record.

An adapter MUST NOT put raw provider error bodies into logs. It MUST first map and sanitize the error.

### Expiry and deletion

ppmlx MUST remove expired data from the primary store, search index, embedding store, cache, and pending extraction queue.

A user deletion MUST remove the same copies. The deletion MUST also remove derived candidates and memory items that depend only on the deleted source.

A deletion tombstone MUST contain no deleted text. It MUST exist only long enough to prevent an old queue item from restoring deleted data.

Local backups SHOULD use the same encryption and expiry policy. ppmlx MUST document any backup copy that cannot support immediate deletion.

### Scope and export

Every stored event, candidate, and memory item MUST have a scope. ppmlx MUST reject a write with no valid scope.

An export MUST use the same redaction and scope checks as a read. Export files MUST have restrictive local file permissions.

## Security and privacy

ppmlx follows data minimization. It stores only the data needed for routing, audit, or approved memory.

Redaction rules are a defense layer, not a reason to retain raw data. The 0-day raw payload rule remains the default.

Secret patterns and sensitive field rules MUST have tests. Test fixtures MUST use false credentials and sanitized personal data.

## Consequences

Some memory candidates lose context because ppmlx removes sensitive tool content. ppmlx requires this loss to keep private data out of storage.

Long-term memory requires expiry maintenance and deletion across indexes. The storage design must support item lineage.

Diagnostics have less raw data. Sanitized categories and correlation identifiers must supply the required support information.

## Rejected alternatives

### Store raw data and redact during read

This design leaves credentials in databases, indexes, and backups. A later read filter cannot remove all copies.

### Keep all approved memory without expiry

Old memory can become incorrect and can increase privacy risk. Explicit renewal is safer than silent permanent storage.

### Let user approval permit credential storage

Credentials are authentication material, not memory. ppmlx must use an authentication profile and its `secret_ref`.

### Redact only known field names

Credentials also occur in text, URLs, commands, and provider errors. Field and value checks are both necessary.

## Compatibility effects

All protocol adapters and harness adapters MUST use the Agent IR `sensitivity` and `provenance` fields.

A missing marker MUST use `restricted` and `{origin: "unknown", trust: "untrusted"}`. Known sensitive sources MUST then fail closed.

Contract fixtures MUST contain no real credential, private path, user name, or personal content. Fixture tests MUST scan all payloads and stream files.

Older stored memory MAY need a one-time scan and cleanup. Migration MUST stop if a safe redaction or scope decision is not possible.
