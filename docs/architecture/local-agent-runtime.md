# Local Agent IR runtime

The Phase 4 runtime is an opt-in path for local MLX tool turns. It uses the protocol adapters and Agent IR contracts from Phases 2 and 3.

## Enable the runtime

Add this server configuration:

```toml
[server]
agent_runtime = "agent_ir"
continuation_ttl_seconds = 86400
```

You can use `PPMLX_AGENT_RUNTIME=agent_ir` and `PPMLX_CONTINUATION_TTL_SECONDS=86400` instead.

The default mode is `legacy`. An invalid runtime value does not send tool traffic to the legacy path.

## Supported path

The strict runtime accepts a request only when all these conditions are true:

- The client connects through the loopback listener.

- The request uses `stream=true`.

- The request includes one or more tools.

- The protocol is OpenAI Chat Completions, OpenAI Responses HTTP, or Anthropic Messages.

- The selected local model has a named output profile.

The profiles are `grok-openai-chat-v1`, `kimi-k2-v1`, `deepseek-v3-v1`, `qwen-json-v1`, `gemma4-v1`, and `lfm25-v1`.

The strict mode rejects Responses WebSocket tool requests. Non-tool requests and non-stream requests stay on the legacy runtime. A tool transcript cannot change these fields to enter the legacy runtime.

## Data flow

The runtime uses this order:

1. Decode the native request with a Phase 3 protocol adapter.

2. Validate the normalized request and continuation link.

3. Pin the local model route for the full tool round trip.

4. Compile the request for the local MLX text engine.

5. Normalize the complete model output with one selected profile.

6. Validate tool names, raw JSON arguments, and the selected tool schema.

7. Create Agent IR events with PPMLX call IDs.

8. Encode the events as native SSE data.

The runtime buffers one complete model turn before it sends SSE data. It does not claim live token streaming for tool turns.

## Tool ownership

The harness is the tool owner. ppmlx sends a tool call to the harness and waits for a later HTTP request that contains the result. ppmlx does not run the tool.

The in-memory ledger keeps only identity, digests, route data, state, and expiry data. It does not keep raw tool arguments or raw tool results. The runtime limits the active Agent IR record and removes it after a final answer, an abandoned continuation, or expiry.

The ledger keeps valid terminal tombstones for 24 hours. It has a limit of 16,384 total entries. It rejects a new tool call with a typed capacity error when it reaches this limit. It does not remove a valid tombstone to make space. An exact concurrent result retry joins the current continuation and receives the same response.

## Fail-closed rules

The strict runtime rejects these inputs before local generation when possible:

- A changed model, instruction, tool list, tool choice, generation value, or stream value in a continuation.

- An unknown tool, duplicate call, duplicate result, or result from another scope.

- A request that sets `store=true` or uses a provider option that the local runtime cannot apply.

- More than one call when the request disables parallel tool calls.

- A tool schema with an external reference or a high-complexity keyword.

- A strict tool schema or unsupported content type.

- A token limit above the server cap.

- A model family that has no named output profile.

String patterns in an accepted tool schema use a time-limited matcher. Request, output, argument, JSON, event, conversation, and replay data also have size or count limits.

The local runtime applies `parallel_tool_calls` and the Anthropic `disable_parallel_tool_use` option. It applies the Chat `stream_options.include_usage` option. The Claude Code `thinking.type=adaptive` and `output_config.effort=high` values enable local tokenizer thinking. The MLX engine removes hidden thinking before tool-output normalization. The `clear_thinking` context rule is therefore a no-op because the runtime does not keep hidden thinking.

## Current limits

- The runtime is local-only and has no remote provider route.

- It has no provider sign-in or OAuth flow.

- It does not read memory.

- It supports text input for this strict tool path. It rejects images and documents.

- It rejects Responses WebSocket tool requests.

- It keeps continuation state in process. A server restart expires an unfinished tool continuation.
