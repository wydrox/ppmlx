# Pi 0.84.2 fixture

- Harness: Pi 0.84.2.
- Protocol: OpenAI Chat Completions, version v1.
- Endpoint: `POST /v1/chat/completions`.
- Status: Real local harness capture.

## Capture

The capture used an isolated Pi model file. It defined one OpenAI-compatible model and sent requests to a local capture server.

The server sent a streamed `bash` call. Pi ran `printf fixture-ok` and sent the result with `call_capture_001`.

The server then sent the final streamed answer. Pi accepted both streams and completed the turn.

Sanitization replaced the temporary working directory with `/workspace`. The request bodies keep the captured wire structure.
