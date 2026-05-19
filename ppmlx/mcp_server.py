#!/usr/bin/env python3
"""ppmlx-memory-mcp — MCP server for ppmlx temporal memory graph.

Exposes memory search, context retrieval, graph walk, and event recording as
MCP tools.  Read-only tools require zero model calls — pure SQLite.  Write
tools enqueue async extraction jobs processed by a background worker thread.

Usage:
    uv run ppmlx-memory-mcp          # stdio mode (for MCP clients)
    ppmlx-memory-mcp --db /path/db   # custom database path
    ppmlx-memory-mcp --no-worker     # disable background extraction worker
"""

from __future__ import annotations
import asyncio
import json
import sys
import threading
import time
from pathlib import Path

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from ppmlx.memory_store import MemoryStore, get_memory_store
from ppmlx.context_reducer import build_handoff_context


# ---------------------------------------------------------------------------
# Server setup
# ---------------------------------------------------------------------------

server = Server("ppmlx-memory")
store: MemoryStore | None = None
_worker_thread: threading.Thread | None = None
_worker_stop: threading.Event | None = None
_worker_enabled: bool = True


def get_store() -> MemoryStore:
    global store
    if store is None:
        db_path = None
        for i, arg in enumerate(sys.argv):
            if arg == "--db" and i + 1 < len(sys.argv):
                db_path = Path(sys.argv[i + 1])
        store = MemoryStore(db_path) if db_path else get_memory_store()
    return store


def start_background_worker(poll_seconds: float = 2.0) -> None:
    """Start a daemon thread that processes extraction jobs from the queue.
    
    Runs until the server shuts down.  Claims jobs one at a time, processes
    them synchronously (model extraction), then sleeps when queue is empty.
    """
    global _worker_thread, _worker_stop
    if _worker_thread is not None:
        return  # already running
    
    _worker_stop = threading.Event()
    
    def _loop():
        s = get_store()
        # Lazy-load the extraction engine (requires model)
        engine = None
        consecutive_empty = 0
        
        while not _worker_stop.is_set():
            try:
                job = s.claim_extraction_job("mcp-worker")
                if job is None:
                    consecutive_empty += 1
                    # Exponential backoff when idle: 2s, 4s, 8s, max 30s
                    wait = min(poll_seconds * (2 ** min(consecutive_empty - 1, 4)), 30.0)
                    _worker_stop.wait(wait)
                    continue
                
                consecutive_empty = 0
                
                # Lazy-init engine on first job
                if engine is None:
                    from ppmlx.memory_engine import MemoryEngine
                    engine = MemoryEngine(store=s)
                
                # Process the claimed job
                event = dict(job.get("payload") or {})
                event_id = str(event.get("event_id") or job.get("source_event_id") or job["job_id"])
                event["event_id"] = event_id
                event.setdefault("request", {"messages": event.get("messages", [])})
                
                result = engine._extract_validate_store(event, suppress_extraction_errors=True)
                result["job_id"] = job["job_id"]
                s.complete_extraction_job(job["job_id"], result=result)
                
            except Exception as exc:
                try:
                    if job:
                        s.fail_extraction_job(job["job_id"], str(exc), retry=True)
                except Exception:
                    pass
    
    _worker_thread = threading.Thread(target=_loop, name="ppmlx-memory-worker", daemon=True)
    _worker_thread.start()


def stop_background_worker() -> None:
    """Signal the background worker to stop."""
    global _worker_stop
    if _worker_stop is not None:
        _worker_stop.set()


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="memory_search",
            description="Search active memory candidates and inferred edges by text query. Supports namespace filtering by project_id and session_id.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query text"},
                    "project_id": {"type": "string", "description": "Filter to project"},
                    "session_id": {"type": "string", "description": "Filter to session"},
                    "limit": {"type": "integer", "default": 20, "description": "Max results"},
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="memory_get_context",
            description="Get compacted session context ready for system prompt injection. Returns markdown-formatted prior conversation state.",
            inputSchema={
                "type": "object",
                "properties": {
                    "project_id": {"type": "string", "description": "Project namespace"},
                    "session_id": {"type": "string", "description": "Current session ID"},
                    "query": {"type": "string", "description": "Optional search query to focus context"},
                    "max_items": {"type": "integer", "default": 40},
                    "max_tokens": {"type": "integer", "default": 2000},
                },
                "required": ["project_id"],
            },
        ),
        Tool(
            name="memory_record_event",
            description="Record a conversation turn for async memory extraction. Non-blocking — enqueues background job.",
            inputSchema={
                "type": "object",
                "properties": {
                    "messages": {
                        "type": "array",
                        "description": "Conversation messages with role and content",
                        "items": {"type": "object"},
                    },
                    "project_id": {"type": "string", "description": "Project namespace"},
                    "session_id": {"type": "string", "description": "Session identifier"},
                    "response_text": {"type": "string", "description": "Assistant response text"},
                },
                "required": ["messages", "project_id"],
            },
        ),
        Tool(
            name="memory_graph_walk",
            description="Multi-hop graph traversal from an entity. Returns all reachable entities within N hops with paths and confidence decay.",
            inputSchema={
                "type": "object",
                "properties": {
                    "entity_name": {"type": "string", "description": "Starting entity name"},
                    "max_hops": {"type": "integer", "default": 2, "description": "Maximum traversal depth (1-5)"},
                    "include_inferred": {"type": "boolean", "default": True},
                },
                "required": ["entity_name"],
            },
        ),
        Tool(
            name="memory_stats",
            description="Get database statistics: event count, candidate counts by status, entity/edge/inferred counts.",
            inputSchema={
                "type": "object",
                "properties": {
                    "project_id": {"type": "string", "description": "Optional project filter"},
                },
            },
        ),
        Tool(
            name="memory_forget",
            description="Remove a candidate from active memory (mark as forgotten).",
            inputSchema={
                "type": "object",
                "properties": {
                    "candidate_id": {"type": "string", "description": "Candidate ID to forget"},
                },
                "required": ["candidate_id"],
            },
        ),
    ]


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    s = get_store()

    if name == "memory_search":
        rows = s.search(
            query=str(arguments.get("query", "")),
            status="active",
            limit=int(arguments.get("limit", 20)),
            project_id=arguments.get("project_id"),
            session_id=arguments.get("session_id"),
        )
        # Enrich with inferred edges
        inferred = []
        try:
            inf = s.query_inferred(status="active", limit=len(rows))
            inferred = [dict(i) for i in inf]
        except Exception:
            pass
        return [TextContent(type="text", text=json.dumps({
            "candidates": [_candidate_summary(r) for r in rows],
            "inferred_edges": inferred,
            "count": len(rows),
        }, ensure_ascii=False, indent=2))]

    elif name == "memory_get_context":
        result = build_handoff_context(
            query=arguments.get("query"),
            project_id=arguments.get("project_id"),
            session_id=arguments.get("session_id"),
            max_items=int(arguments.get("max_items", 40)),
            max_tokens=int(arguments.get("max_tokens", 2000)),
            store=s,
        )
        return [TextContent(type="text", text=json.dumps({
            "context": result.context,
            "tokens": result.tokens,
            "items_count": len(result.items),
        }, ensure_ascii=False, indent=2))]

    elif name == "memory_record_event":
        messages = arguments.get("messages", [])
        project_id = str(arguments.get("project_id", ""))
        session_id = str(arguments.get("session_id", ""))
        event_id = f"mcp-{project_id}-{session_id}-{_now_iso()}"

        s.record_event({
            "event_id": event_id,
            "endpoint": "/mcp/record_event",
            "app_id": None,
            "project_id": project_id,
            "session_id": session_id,
            "model_alias": "mcp-client",
            "model_repo": "mcp",
            "request": {"messages": messages},
            "response_text": str(arguments.get("response_text", "")),
            "metadata": {"source": "mcp"},
        })

        # Enqueue async extraction
        try:
            s.enqueue_extraction_job(
                {"messages": messages},
                source_event_id=event_id,
                priority=0,
            )
            queued = True
        except Exception:
            queued = False

        return [TextContent(type="text", text=json.dumps({
            "event_id": event_id,
            "queued_extraction": queued,
        }, ensure_ascii=False))]

    elif name == "memory_graph_walk":
        result = s.graph_walk(
            entity_name=str(arguments["entity_name"]),
            max_hops=int(arguments.get("max_hops", 2)),
            include_inferred=bool(arguments.get("include_inferred", True)),
        )
        return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False, indent=2))]

    elif name == "memory_stats":
        stats = s.stats()
        if arguments.get("project_id"):
            # Add project-specific counts via query
            pass
        return [TextContent(type="text", text=json.dumps(stats, ensure_ascii=False, indent=2, default=str))]

    elif name == "memory_forget":
        ok = s.forget_candidate(str(arguments["candidate_id"]))
        return [TextContent(type="text", text=json.dumps({"forgotten": ok}))]

    else:
        return [TextContent(type="text", text=json.dumps({"error": f"Unknown tool: {name}"}))]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _candidate_summary(candidate: dict) -> dict:
    """Compact candidate summary for MCP responses."""
    return {
        "candidate_id": candidate.get("candidate_id", ""),
        "type": candidate.get("type", ""),
        "subject": candidate.get("subject", ""),
        "predicate": candidate.get("predicate", ""),
        "object": candidate.get("object", ""),
        "text": candidate.get("text", ""),
        "scope": candidate.get("scope", ""),
        "confidence": candidate.get("confidence", 0),
        "salience": candidate.get("salience", 0),
        "source_quote": (candidate.get("source_quote", "") or "")[:120],
    }


def _now_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    """Run the MCP server on stdio with background extraction worker."""
    global _worker_enabled
    # Parse --no-worker flag
    if "--no-worker" in sys.argv:
        _worker_enabled = False
    
    async def _run():
        if _worker_enabled:
            start_background_worker()
        try:
            async with stdio_server() as (read_stream, write_stream):
                await server.run(read_stream, write_stream, server.create_initialization_options())
        finally:
            if _worker_enabled:
                stop_background_worker()
    
    asyncio.run(_run())


if __name__ == "__main__":
    main()
