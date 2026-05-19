# ppmlx Memory as MCP Server — Architecture

## Why MCP

Pi already integrates with MemPalace via a dual CLI/MCP extension pattern. The same
MartMart MCP pattern (`martmart-mcp/index.ts`) spawns an MCP server, discovers tools,
and mirrors them as pi custom tools. ppmlx memory should follow the same pattern — but
with one critical difference: ppmlx memory would be a **shared** MCP server usable by
pi, Claude Code, Continue, Cursor, and any other MCP client simultaneously.

**Current:** ppmlx memory lives inside `ppmlx serve` (FastAPI, port 6767). Only
accessible when the full model server is running.

**Target:** Standalone `ppmlx-memory-mcp` server. No MLX model needed for retrieval.
Extraction can be on-demand or async. Multiple clients share one memory graph.

## Architecture

```
                    ┌─────────────────────────┐
                    │   ppmlx-memory-mcp      │
                    │   (stdio MCP server)    │
                    │                         │
  pi ──────────────►│ tools:                  │
  Claude Code ─────►│   memory_search         │◄── memory.db
  Cursor ──────────►│   memory_get_context    │    (SQLite)
  Continue ────────►│   memory_record_event   │
                    │   memory_graph_query    │
                    │   memory_extract        │◄── MLX (on-demand)
                    │   memory_list_sessions  │
                    │                         │
                    │ resources:              │
                    │   memory://{project}/   │
                    │     context             │
                    │     graph               │
                    └─────────────────────────┘
```

## Tools

### `memory_search`
Search active memory candidates and inferred edges.
```
INPUT:  query: str, project_id?: str, session_id?: str, limit?: int, scope?: str
OUTPUT: candidates[], inferred_edges[], total_count
```

### `memory_get_context`
Get compacted session context ready for injection into system prompt.
```
INPUT:  project_id: str, session_id?: str, max_tokens?: int, query?: str
OUTPUT: context_text (markdown), items[], tokens_used
```

### `memory_record_event`
Record a conversation turn for later extraction. Non-blocking — enqueues async job.
```
INPUT:  messages: list[{role,content}], project_id: str, session_id?: str, 
        model_alias?: str, response_text?: str
OUTPUT: event_id, queued_extraction_job_id
```

### `memory_extract`
Synchronously extract and store memories from messages. Uses the full v2 pipeline.
```
INPUT:  messages: list[{role,content}], project_id: str, session_id?: str
OUTPUT: candidates[], active_count, rejected_count, inferred_count, duration_ms
```

### `memory_graph_query`
Query the entity-relationship graph.
```
INPUT:  entity_name?: str, project_id?: str, include_inferred?: bool, limit?: int
OUTPUT: nodes[], edges[], inferred_edges[], connected_components
```

### `memory_list_sessions`
List sessions with memory data, ordered by recency.
```
INPUT:  project_id?: str, limit?: int
OUTPUT: sessions[{session_id, project_id, candidate_count, edge_count, last_event_at}]
```

### `memory_forget`
Remove a candidate from active memory.
```
INPUT:  candidate_id: str
OUTPUT: removed: bool
```

### `memory_stats`
Get database statistics.
```
INPUT:  project_id?: str
OUTPUT: events, candidates (by_status), entities, edges, inferred, compaction_ratio
```

## Resources

### `memory://{project_id}/context`
Live-rendered compacted session context. Clients subscribe and get updates
as new facts are extracted. Useful for system prompt injection.

### `memory://{project_id}/graph`
Current graph state as JSON. Nodes, edges, inferred edges. For visualization
or debugging.

## pi Extension

Follow the `martmart-mcp` pattern:

```typescript
// ~/.pi/agent/extensions/ppmlx-memory/index.ts

type PPMLXMemorySettings = {
  serverCommand?: string;     // default: "uvx ppmlx-memory-mcp"
  serverArgs?: string[];
  autoInjectContext?: boolean; // inject memory context into system prompt
  projectMapping?: Record<string, string>;  // cwd → project_id mapping
};
```

**Key behavior:**
- On session start: call `memory_get_context` for current project, inject into system prompt
- During session: after each user turn, call `memory_search` with user query, surface relevant facts
- On session end (or periodically): call `memory_extract` with the session transcript
- Write tools are background-queued (same pattern as MemPalace writes)

## Comparison: MemPalace vs ppmlx Memory

| | MemPalace | ppmlx Memory |
|---|---|---|
| Storage | File-based (drawers in dirs) | SQLite (candidates, edges, entities) |
| Structure | Unstructured notes + KG triples | Structured S-P-O + typed candidates |
| Extraction | Manual (agent writes drawers) | Automatic (LLM extraction pipeline) |
| Graph | Knowledge graph (manual edges) | Entity graph (deterministic projection) |
| Retrieval | BM25 search | FTS5 + structured query + namespace filtering |
| Inference | None | Transitive, co-occurrence, temporal |
| Context injection | System prompt (write policy) | Compact/inject mode with hot tail |

**Complementary, not competitive.** MemPalace is for unstructured, human-curated knowledge.
ppmlx memory is for automatic, structured, high-volume extraction from conversations.
They serve different layers of the memory stack.

## Implementation Plan

### Phase 1: MCP Server Core (1-2 days)
1. Extract `MemoryStore` + query methods into standalone package
2. Add MCP server boilerplate (Python MCP SDK or manual JSON-RPC over stdio)
3. Implement `memory_search`, `memory_stats`, `memory_list_sessions` (read-only, no model needed)

### Phase 2: Write Path (1 day)
4. Implement `memory_record_event` (queues async extraction)
5. Implement `memory_extract` (synchronous extraction with model)
6. Implement `memory_forget`

### Phase 3: Context Injection (1 day)
7. Implement `memory_get_context` (compacted session context)
8. Add resources: `memory://{project}/context`, `memory://{project}/graph`

### Phase 4: pi Extension (1 day)
9. Build pi extension following `martmart-mcp` pattern
10. Auto-inject context on session start
11. Background extraction on session end

### Phase 5: Multi-Client Testing (1 day)
12. Test with pi + Claude Code simultaneously
13. Test cross-project fact propagation
14. Performance testing with 100+ sessions

## Risks

1. **SQLite concurrency**: Multiple MCP clients writing to same DB. Mitigation: WAL mode + write lock (already implemented).
2. **Model availability**: Extraction requires loaded MLX model. Mitigation: lazy load, queue when unavailable, pure-code fallback.
3. **Tool conflict**: pi already has MemPalace tools. Users might confuse `memory_search` (ppmlx) with `mempalace_search`. Mitigation: distinct naming, different use cases.
