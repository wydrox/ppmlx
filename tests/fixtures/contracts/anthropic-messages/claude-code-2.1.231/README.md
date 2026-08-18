# Claude Code 2.1.231 fixture

- Harness: Claude Code 2.1.231.
- Protocol: Anthropic Messages, version 2023-06-01.
- Endpoint: `POST /v1/messages`.
- Status: Real local harness capture.

## Capture

The capture used Claude Code in bare print mode. An isolated settings object sent requests to a local capture server.

The server sent a streamed `Bash` call. Claude Code ran `printf fixture-ok` and sent the result with `toolu_capture_001`.

The server then sent the final streamed answer. Claude Code accepted both streams and completed the turn.

Sanitization replaced the device ID and session ID. The request bodies keep the captured wire structure.
