# ADR 0004: Provider Authentication

- Status: Accepted
- Date: 2026-08-18

## Context

Users can access model providers through API accounts, product subscriptions, or both.
These account types do not always grant the same service, model, quota, or API right.

Harnesses can also keep their own OAuth sessions and local credentials.
Reuse of those private credentials can violate provider policy and can expose user accounts.

## Decision

ppmlx uses named provider authentication profiles.
A remote route refers to one profile name, and the provider adapter resolves its credential at request time.
Local inference does not need a provider authentication profile.

The contract defines two provider credential classes:

- `api_key`: A user-supplied provider API key for an approved API endpoint.
- `oauth_session`: A ppmlx session from an official, documented provider authorization flow.

An API key and a product subscription are separate entitlements.
An active ChatGPT subscription does not by itself grant OpenAI API access.
ppmlx will not treat a subscription login as an API credential unless the provider officially supports that use.

An adapter can offer `oauth_session` only after the project verifies the official flow, scopes, token audience, and permitted client use.
Until that verification exists, the adapter must report subscription authentication as unsupported.

## Authentication profile

A profile contains non-secret configuration and a reference to secret storage.
It contains these identity fields:

- `name`: A local unique profile name.
- `provider`: The provider adapter identifier.
- `credential_type`: `api_key` or `oauth_session`.
- `secret_ref`: A reference to protected local storage.

It also contains these account and session fields:

- `account_label`: An optional local display label.
- `endpoint`: An approved provider endpoint or adapter default.
- `scopes`: The recorded OAuth scopes, when applicable.
- `expires_at`: The known OAuth expiry time, when applicable.

An OAuth profile also contains these session fields:

- `refresh_secret_ref`: The protected refresh token reference, when applicable.
- `refresh_expires_at`: The known refresh token expiry time, when applicable.
- `status`: `active`, `refreshing`, `reauthentication_required`, `revoked`, or `logged_out`.

Remote routes store only the profile name.
Request objects and the Agent IR do not contain provider secrets.

## Local endpoint authentication

Local endpoint authentication protects the ppmlx listener.
It does not authenticate ppmlx to a model provider.

The listener supports these credential types:

- `none`: No credential, with the strict loopback limits below.
- `local_token`: A random ppmlx token for one local principal.

OpenAI-compatible paths use `Authorization: Bearer <local_token>`.
These paths include `/v1/responses`, `/v1/chat/completions`, `/v1/completions`, and `/v1/embeddings`.

The Anthropic Messages path accepts `x-api-key: <local_token>`.
It can also accept `Authorization: Bearer <local_token>` for a tested harness profile.
If both headers exist, they must contain the same token.

ppmlx removes the local credential header before provider transport starts.
It never replaces that value with a provider credential in the same header object.

### Default-deny path policy

`GET /health` is the only public HTTP endpoint.
It returns liveness status only and does not return models, routes, providers, versions, memory data, or metrics.

In `local_token` mode, all other paths require the credential that their accepted path contract defines.
General paths use `local_token`.
Protected paths include model lists, token counts, metrics, memory, administration, and every inference path.

The policy includes `/v1/models`, `/v1/messages/count_tokens`, `/metrics`, and every path below `/v1/`.
It also includes each present or future memory endpoint.
ppmlx protects each new path until an accepted ADR explicitly makes it public.

The `/v1/memory/read/*` paths use `Authorization: Bearer <grant-credential>` from ADR 0006.
The grant credential is the accepted path credential and a `local_token` alone is not sufficient.
The memory paths do not require two `Authorization` headers.

### No-auth limits

ppmlx permits `none` only when every listener address is a loopback address or a local Unix socket.
ppmlx must reject startup if an external listener uses `none`.

No-auth mode uses one listener-scoped local principal.
It disables token checks only for requests that arrive through the permitted local listener.
It must reject non-loopback browser origins and must not trust forwarded client-address headers.
Diagnostics must show a persistent no-auth warning.

## OAuth session lifecycle

An OAuth login needs explicit user approval for the provider, account, scopes, and callback result.
The adapter stores access and refresh tokens as separate protected secrets.
It stores expiry, scope, audience, token generation, and account metadata without token values.

One profile can have only one active refresh operation.
Concurrent requests wait for that single-flight operation.
The default refresh threshold is five minutes before access token expiry.

A rotating refresh response replaces both tokens in one atomic operation.
The adapter increments the token generation before waiting requests continue.
An older refresh operation must not replace a newer generation.

Automatic refresh can use only the previously approved provider, account, audience, and scopes.
A scope increase, account change, or new consent screen needs explicit user approval.
An `invalid_grant`, revoked token, or expired refresh token changes the profile to `reauthentication_required`.

Logout first calls the official revocation endpoint when the provider supplies one.
ppmlx then removes local access and refresh tokens, even when remote revocation fails.
It reports a remote revocation failure and tells the user to revoke access in the provider account.

## Normative rules

### Separation and storage

- ppmlx MUST keep provider credentials separate from local endpoint credentials.
- ppmlx MUST keep credentials separate for each provider and authentication profile.
- A remote route MUST refer to a profile by name. It MUST NOT store a secret.
- ppmlx MUST use protected operating-system storage by default.

Protected store rules also apply:

- On macOS, ppmlx MUST use Keychain as the default protected store.
- A plaintext credential file MUST require explicit user action and a clear warning.

Credential file rules also apply:

- A plaintext credential file MUST allow access only to the current operating-system user.
- Environment references MAY supply API keys without persistent ppmlx storage.

### Local endpoint credentials

- `local_token` MUST contain at least 256 bits of random data before encoding.
- ppmlx MUST store only a verifier when later plaintext token recovery is not necessary.
- Token comparison MUST use a constant-time verifier.
- Local token rotation MUST create the replacement before it revokes the old token.
- An overlap period MUST require explicit configuration and MUST have an expiry time.
- Local token revocation MUST invalidate new requests without a server restart.

Path policy rules also apply:

- Access checks MUST use a default-deny path policy.
- In `local_token` mode, only `GET /health` MAY omit an accepted path credential.
- Model, token-count, metrics, memory, administration, and inference paths MUST require authentication.
- General protected paths MUST use `local_token`.
- Memory-read paths MUST use the ADR 0006 grant credential instead of `local_token`.
- No future route MAY become public without an accepted architecture decision.

### Session isolation

- ppmlx MUST NOT read a credential store that belongs to Claude Code, Codex, OpenCode, Pi, or a browser.
- ppmlx MUST NOT copy, import, or refresh an OAuth session from another application.
- ppmlx MUST NOT exchange browser cookies for provider tokens.
- An OAuth adapter MUST use an official documented flow and permitted client identity.
- An OAuth flow MUST validate `state` and use PKCE when the provider supports it.
- An adapter MUST request only the scopes that its route needs.

### Credential transport

- ppmlx MUST validate token audience and provider before each token use.
- A provider credential MUST go only to its configured provider endpoint.
- Redirects to an unapproved host MUST NOT receive an authorization header.
- ppmlx MUST redact keys, tokens, cookies, authorization codes, and refresh tokens from logs and errors.
- ppmlx MUST remove local endpoint credentials before it builds an upstream request.

### Provider credential rotation

- API key rotation MUST validate the new secret before an atomic profile switch.
- OAuth access and refresh tokens MUST use separate protected secret references.
- OAuth refresh MUST be single-flight for each profile.
- A refresh result MUST update token values, expiry, and generation atomically.
- A stale refresh result MUST NOT replace a newer token generation.
- A provider revocation MUST change the profile status before later requests can use it.

### Lifecycle and fallback

- ppmlx MUST support profile logout or deletion without removal of unrelated profiles.
- An expired OAuth session MUST cause a clear reauthentication error when the provider does not permit refresh.
- Authentication failure MUST NOT cause fallback to a profile for a different account.
- Authentication failure MUST NOT cause fallback from local inference to a remote provider.
- Logout MUST delete local OAuth tokens even if the remote revocation request fails.
- A request MUST NOT refresh a profile after logout or revocation.

### Approval

- A new OAuth login MUST require explicit user approval.
- A new provider account or larger scope set MUST require new approval.
- Automatic refresh MAY continue only the approved session with the same or smaller scopes.
- ppmlx MUST show the provider, account label, scopes, and endpoint before approval.
- A non-interactive process MUST NOT approve a new OAuth session by itself.

### Capability claims

- An adapter MUST report its supported credential classes before a route becomes active.
- ppmlx MUST NOT claim subscription support until automated tests cover the approved authorization flow.

## Failure codes

Local endpoint failures use these stable codes:

- `local_auth_missing`: A protected endpoint received no local credential.
- `local_auth_invalid`: The supplied local credential is not valid.
- `local_auth_conflict`: Two accepted headers contain different credentials.
- `local_auth_required`: Listener configuration requires a credential.

Provider profile failures use these stable codes:

- `provider_auth_expired`: The usable token expired and refresh is not available.
- `provider_auth_revoked`: The provider or user revoked the credential.
- `provider_auth_refresh_failed`: The approved refresh operation failed.
- `provider_auth_reapproval_required`: The session needs interactive approval.
- `provider_auth_unsupported`: The adapter does not support the requested credential class.

OpenAI-compatible paths return their code in `error.code` and use `error.type=authentication_error`.
The Anthropic Messages path returns its code in `error.code` and uses `error.type=authentication_error`.
Authentication error text must not identify which stored token, account, or secret value matched.

## Security and privacy

Provider tokens and API keys are high-value secrets.
The process must keep each secret for the shortest necessary time and must not put it in the Agent IR.

The protected store contains local tokens, provider API keys, OAuth access tokens, and OAuth refresh tokens.
Configuration files contain only secret references and non-secret metadata.

OAuth callback servers must bind to localhost and accept only the current authorization transaction.
The user must see the provider, account label, requested scopes, and callback result.

An endpoint override can send a credential to an attacker.
Each provider adapter must validate the endpoint scheme and host against explicit configuration.

## Consequences

Users can keep separate accounts and select them through route profiles.
They can remove one credential without changes to model aliases or other accounts.

Harnesses use stable local credentials when a provider key or OAuth session changes.
External listeners must use local authentication.
Token rotation and single-flight refresh add persistent state and lock management.

Some subscription-based access will remain unavailable.
This limit applies when a provider has no official proxy authorization flow.
The project needs provider-specific research and tests before it enables each OAuth adapter.

## Compatibility effects

- **Claude Code:** It sends a local token in `x-api-key` or a tested bearer header, and ppmlx does not reuse its provider session.
- **Codex:** It sends a local bearer token, and ppmlx does not reuse its Codex or ChatGPT session.
- **OpenCode:** It sends a local bearer token, and provider profiles stay inside ppmlx.
- **Pi:** It sends a local bearer token, and provider profiles stay inside ppmlx.

The harness does not receive the selected provider credential.
Provider changes do not require a harness login when its ppmlx endpoint credential stays valid.

## Rejected alternatives

### Import harness sessions

This option reads tokens from an installed harness.
It creates hidden account coupling and can break provider policy.

### Import browser sessions

This option uses cookies or browser storage as provider authentication.
It has a large account security risk and no stable service contract.

### Treat a subscription as API access

This option assumes that product payment grants API rights.
Providers often separate these products, so the assumption is unsafe.

### Store credentials in route configuration

This option puts secrets in files that users can copy, commit, or include in diagnostics.
Named secret references reduce this risk.

### Automatic cross-account fallback

This option can send private content to an unintended account or billing owner.
Authentication failure must remain explicit.
