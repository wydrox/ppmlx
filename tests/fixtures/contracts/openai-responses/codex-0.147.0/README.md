# Codex 0.147.0 fixture

- Harness: Codex CLI 0.147.0.
- Protocol: OpenAI Responses, version v1.
- Endpoint: `POST /v1/responses`.
- Status: Approved derived fixture from a real local capture.

## Capture

The capture used an ephemeral Codex run and a custom Responses provider. The provider sent requests to a local capture server.

The server sent a streamed `exec_command` call. Codex ran `printf fixture-ok` and sent the result with `call_capture_001`.

The server then sent the final streamed answer. Codex accepted both streams and completed the turn.

The Codex runtime added the host session envelope and all registered tool schemas. Sanitization removed that unrelated content after capture.

The fixture keeps the user message, used tool schema, wire metadata fields, and complete tool loop. Stable values replace volatile client IDs.
