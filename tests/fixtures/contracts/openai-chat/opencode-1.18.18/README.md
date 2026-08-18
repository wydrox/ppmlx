# OpenCode 1.18.18 fixture

- Harness: OpenCode 1.18.18.
- Protocol: OpenAI Chat Completions, version v1.
- Endpoint: `POST /v1/chat/completions`.
- Status: Real local harness capture.

## Capture

The capture used an isolated npm installation and an isolated OpenCode data directory. A local project file defined the capture provider.

The provider used `@ai-sdk/openai-compatible`. OpenCode sent the agent requests to a local capture server.

The server sent a streamed `bash` call. OpenCode ran `printf fixture-ok` and sent the result with `call_capture_001`.

The server then sent the final streamed answer. OpenCode accepted both streams and completed the turn.

Sanitization replaced personal paths, the temporary working directory, and the capture date. The separate title request is not in this fixture.
