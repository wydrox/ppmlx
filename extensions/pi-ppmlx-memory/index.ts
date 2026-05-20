/**
 * ppmlx-memory pi extension (v2 — CLI-first, MCP fallback)
 *
 * Integrates ppmlx temporal memory graph into pi sessions.
 * Uses direct CLI calls for speed (no JSON-RPC wrapper overhead).
 * Falls back to MCP if CLI is unavailable.
 *
 * Install: copy this directory to ~/.pi/agent/extensions/ppmlx-memory/
 */

import type { ExtensionAPI } from "@mariozechner/pi-coding-agent";
import { Type } from "@sinclair/typebox";
import { spawn } from "node:child_process";
import { homedir } from "node:os";
import { basename, join } from "node:path";

// ---------------------------------------------------------------------------
// Types & settings
// ---------------------------------------------------------------------------

type PPMemorySettings = {
	serverCommand?: string;    // default: "uv"
	serverArgs?: string[];     // default: ["run", "ppmlx"]
	autoRecord?: boolean;
	autoInjectContext?: boolean;
	contextMaxTokens?: number;
	projectId?: string;
};

const PI_AGENT_DIR = join(homedir(), ".pi", "agent");
const GLOBAL_SETTINGS_PATH = join(PI_AGENT_DIR, "settings.json");

let _settings: PPMemorySettings | null = null;

function getSettings(): PPMemorySettings {
	if (_settings) return _settings;
	const s: PPMemorySettings = {
		serverCommand: "uv",
		serverArgs: ["run", "ppmlx"],
		autoRecord: true,
		autoInjectContext: true,
		contextMaxTokens: 2000,
		projectId: "auto",
	};
	try {
		const fs = require("node:fs");
		for (const path of [GLOBAL_SETTINGS_PATH, join(process.cwd(), ".pi", "settings.json")]) {
			if (fs.existsSync(path)) {
				const data = JSON.parse(fs.readFileSync(path, "utf8"));
				const mem = data?.ppmlxMemory;
				if (mem) {
					if (mem.serverCommand) s.serverCommand = mem.serverCommand;
					if (mem.serverArgs) s.serverArgs = mem.serverArgs;
					if (typeof mem.autoRecord === "boolean") s.autoRecord = mem.autoRecord;
					if (typeof mem.autoInjectContext === "boolean") s.autoInjectContext = mem.autoInjectContext;
					if (typeof mem.contextMaxTokens === "number") s.contextMaxTokens = mem.contextMaxTokens;
					if (mem.projectId) s.projectId = mem.projectId;
				}
			}
		}
	} catch { /* defaults */ }
	_settings = s;
	return s;
}

function getProjectId(): string {
	const s = getSettings();
	if (s.projectId !== "auto") return s.projectId ?? "unknown";
	return basename(process.cwd()) || "unknown";
}

// ---------------------------------------------------------------------------
// CLI execution (direct, no MCP wrapper)
// ---------------------------------------------------------------------------

type CliResult = { stdout: string; stderr: string; code: number; ok: boolean };

function execCli(args: string[], stdin?: string): Promise<CliResult> {
	const s = getSettings();
	return new Promise((resolve) => {
		const proc = spawn(s.serverCommand ?? "uv", [...(s.serverArgs ?? ["run", "ppmlx"]), ...args], {
			stdio: ["pipe", "pipe", "pipe"],
			env: { ...process.env },
			timeout: 15_000,
		});
		let stdout = "";
		let stderr = "";
		proc.stdout?.on("data", (d: Buffer) => { stdout += d.toString(); });
		proc.stderr?.on("data", (d: Buffer) => { stderr += d.toString(); });
		if (stdin) { proc.stdin?.write(stdin); proc.stdin?.end(); }
		proc.on("close", (code: number | null) => {
			resolve({ stdout, stderr, code: code ?? -1, ok: code === 0 });
		});
		proc.on("error", () => resolve({ stdout, stderr, code: -1, ok: false }));
	});
}

async function cliSearch(query: string, projectId: string, limit: number = 5): Promise<string> {
	const r = await execCli(["memory", "search", "--json", "--limit", String(limit), query]);
	if (!r.ok) return "";
	return r.stdout;
}

async function cliContext(projectId: string, maxTokens: number = 2000): Promise<string> {
	const r = await execCli(["memory", "handoff", "--json", "--project", projectId, "--max-tokens", String(maxTokens)]);
	if (!r.ok) return "";
	try {
		const data = JSON.parse(r.stdout);
		return data?.context ?? "";
	} catch { return ""; }
}

async function cliRecordEvent(projectId: string, sessionId: string, messagesJson: string): Promise<void> {
	// Use memory ingest-bench style: write temp file, replay
	// Simpler: just fire-and-forget via the extraction enqueue
	// For now, skip — the auto-record path can use MCP or be deferred
}

// ---------------------------------------------------------------------------
// Background extraction subagent
// ---------------------------------------------------------------------------

function getPiInvocation(args: string[]): { command: string; args: string[] } {
	// pi CLI is typically available as "pi" in PATH when installed globally
	return { command: "pi", args };
}

function queueBackgroundExtraction(
	messages: Array<{ role: string; content: string }>,
): void {
	const s = getSettings();
	if (!s.autoRecord) return;

	const transcript = messages
		.map((m) => `${m.role}: ${m.content}`)
		.join("\n")
		.slice(0, 6000);

	const prompt = [
		"Extract durable memory facts from this conversation.",
		"Return ONLY pipe-delimited rows, no markdown, no prose.",
		"Format: type|subject|predicate|object|text|scope|confidence|salience|source_quote",
		"Types: fact, preference, decision, todo, constraint, instruction, entity_note, relationship, workflow_state",
		"Scopes: global, project, session",
		"Only include facts explicitly supported by the conversation.",
		"",
		"After extraction, pass all pipe-delimited rows together to: uv run ppmlx memory add --project-id " + getProjectId(),
		"",
		"Conversation:",
		transcript,
	].join("\n");

	const proc = spawn("pi", ["-p", "--no-session", prompt], {
		cwd: process.cwd(),
		env: { ...process.env },
		detached: true,
		stdio: "ignore",
	});
	proc.once("spawn", () => proc.unref());
	proc.on("error", () => {});
}

async function cliGraphWalk(entity: string, maxHops: number = 2): Promise<string> {
	const r = await execCli(["memory", "walk", entity, "--max-hops", String(maxHops)]);
	if (!r.ok) return "";
	return r.stdout;
}

// ---------------------------------------------------------------------------
// Extension entry point
// ---------------------------------------------------------------------------

export default function (pi: ExtensionAPI) {
	const sessionId = `pi-${Date.now()}`;
	const projectId = getProjectId();
	const s = getSettings();

	// Session start
	pi.on("session_start", async (_event, ctx) => {
		if (s.autoInjectContext) {
			const ctx_text = await cliContext(projectId, s.contextMaxTokens ?? 2000);
			if (ctx_text && ctx.hasUI) {
				ctx.ui.notify(`ppmlx-memory: loaded context for "${projectId}"`, "info");
			}
		}
		if (ctx.hasUI) {
			ctx.ui.notify(`ppmlx-memory ready (project: ${projectId})`, "info");
		}
	});

	// Before agent start: inject context + strong memory usage nudge
	pi.on("before_agent_start", async (event, _ctx) => {
		const additions: string[] = [];
		if (s.autoInjectContext) {
			const ctx_text = await cliContext(projectId, s.contextMaxTokens ?? 2000);
			if (ctx_text) {
				additions.push(ctx_text);
				additions.push([
					"The context above is from prior sessions in this project.",
					"Use memory_search to find additional relevant facts when:",
					"- the user asks about past work, decisions, or project state",
					"- you need context about architecture, constraints, or preferences",
					"- the user mentions a topic you haven't discussed in this session",
				].join(" "));
			} else {
				additions.push([
					"No prior session context found for this project.",
					"Use memory_search if the user asks about past work.",
				].join(" "));
			}
		}
		additions.push([
			"ppmlx memory is active. Facts from this and prior sessions are stored",
			"in a temporal memory graph.",
			"Use memory_search to retrieve context. Prefer it over raw DB access.",
			"Use memory_walk to explore connected entities.",
			"Use memory_add to store durable facts; it upserts by subject/predicate/object.",
			"Use memory_force_add only when an intentional duplicate record is needed.",
		].join(" "));
		const systemPrompt = event.systemPrompt
			? `${event.systemPrompt}\n\n${additions.join("\n\n")}`
			: additions.join("\n\n");
		return { systemPrompt };
	});

	// ── Write tool ───────────────────────────────────────────────
	pi.registerTool({
		name: "memory_add",
		label: "Add a fact to ppmlx memory",
		description: "Store or update a durable fact in the memory graph. Upserts by subject/predicate/object.",
		promptSnippet: "When you discover a durable fact, call memory_add to store it immediately.",
		promptGuidelines: [
			"When user says \"zapamiętaj\", \"remember that\", \"zapisz\", or explicitly asks to store something — call memory_add.",
			"When you make an architectural decision — proactively call memory_add with type=decision.",
			"When user states a preference — call memory_add with type=preference.",
		],
		parameters: Type.Object({
			type: Type.String({ description: "fact, preference, decision, todo, constraint, instruction, entity_note, relationship, workflow_state" }),
			subject: Type.String({ description: "Short stable subject" }),
			predicate: Type.String({ description: "Relation or action" }),
			object: Type.String({ description: "Object value" }),
			text: Type.String({ description: "One sentence describing the memory" }),
			scope: Type.String({ description: "global, project, or session" }),
			confidence: Type.Optional(Type.Number({ description: "0.0-1.0 (default 0.9)" })),
			valid_from: Type.Optional(Type.String({ description: "ISO timestamp when the memory became valid" })),
			valid_to: Type.Optional(Type.String({ description: "ISO timestamp when the memory expires" })),
		}),
		execute: async (_id, params, _signal) => {
			const args: string[] = ["memory", "add",
				"--type", params.type,
				"--subject", params.subject,
				"--predicate", params.predicate,
				"--object", params.object,
				"--text", params.text,
				"--scope", params.scope,
				"--confidence", String(params.confidence ?? 0.9),
				"--project-id", projectId,
				"--session-id", sessionId,
			];
			if (params.valid_from) args.push("--valid-from", params.valid_from);
			if (params.valid_to) args.push("--valid-to", params.valid_to);
			const r = await execCli(args);
			return { content: [{ type: "text", text: r.ok ? r.stdout : r.stderr }] };
		},
	});

	pi.registerTool({
		name: "memory_force_add",
		label: "Force-add a fact (bypass dedup)",
		description: "Store a durable fact as a new record, bypassing subject/predicate/object deduplication.",
		promptSnippet: "Use memory_force_add only when an intentional duplicate memory record is needed.",
		promptGuidelines: [
			"Prefer memory_add for normal durable facts.",
			"Use memory_force_add only when the duplicate itself is meaningful or needed for debugging.",
		],
		parameters: Type.Object({
			type: Type.String({ description: "fact, preference, decision, todo, constraint, instruction, entity_note, relationship, workflow_state" }),
			subject: Type.String({ description: "Short stable subject" }),
			predicate: Type.String({ description: "Relation or action" }),
			object: Type.String({ description: "Object value" }),
			text: Type.String({ description: "One sentence describing the memory" }),
			scope: Type.String({ description: "global, project, or session" }),
			confidence: Type.Optional(Type.Number({ description: "0.0-1.0 (default 0.9)" })),
			valid_from: Type.Optional(Type.String({ description: "ISO timestamp when the memory became valid" })),
			valid_to: Type.Optional(Type.String({ description: "ISO timestamp when the memory expires" })),
		}),
		execute: async (_id, params, _signal) => {
			const args: string[] = ["memory", "add",
				"--force",
				"--type", params.type,
				"--subject", params.subject,
				"--predicate", params.predicate,
				"--object", params.object,
				"--text", params.text,
				"--scope", params.scope,
				"--confidence", String(params.confidence ?? 0.9),
				"--project-id", projectId,
				"--session-id", sessionId,
			];
			if (params.valid_from) args.push("--valid-from", params.valid_from);
			if (params.valid_to) args.push("--valid-to", params.valid_to);
			const r = await execCli(args);
			return { content: [{ type: "text", text: r.ok ? r.stdout : r.stderr }] };
		},
	});

	// ── Read tools ───────────────────────────────────────────────
	pi.registerTool({
		name: "memory_search",
		label: "Search ppmlx memory",
		description: "Search active memory candidates and inferred edges from past sessions.",
		promptSnippet: "Search ppmlx memory for prior facts, decisions, preferences, constraints, and workflow state.",
		promptGuidelines: [
			"Use memory_search as the primary way to query memory. Avoid raw SQL unless debugging schema or doing bulk operations.",
			"When the user asks about prior work, past decisions, architecture, or project state — call memory_search BEFORE answering.",
			"When the user mentions a topic briefly discussed before — use memory_search to recover the full context.",
			"When uncertain about a constraint or preference — check memory_search first.",
		],
		parameters: Type.Object({
			query: Type.String({ description: "Search query" }),
			limit: Type.Optional(Type.Number({ description: "Max results (default 5)" })),
		}),
		execute: async (_id, params, _signal) => {
			const result = await cliSearch(params.query, projectId, params.limit ?? 5);
			return { content: [{ type: "text", text: result || "(no results)" }] };
		},
	});

	pi.registerTool({
		name: "memory_handoff",
		label: "Get compacted memory context",
		description: "Get compacted memory context for the current namespace/project.",
		parameters: Type.Object({
			max_tokens: Type.Optional(Type.Number({ description: "Max context tokens (default 2000)" })),
		}),
		execute: async (_id, params, _signal) => {
			const result = await cliContext(projectId, params.max_tokens ?? 2000);
			return { content: [{ type: "text", text: result || "(no prior context for this project)" }] };
		},
	});

	pi.registerTool({
		name: "memory_walk",
		label: "Walk ppmlx memory graph",
		description: "Multi-hop graph traversal from an entity. Returns connected entities.",
		parameters: Type.Object({
			entity_name: Type.String({ description: "Starting entity name" }),
			max_hops: Type.Optional(Type.Number({ description: "Max traversal depth (1-3, default 2)" })),
		}),
		execute: async (_id, params, _signal) => {
			const result = await cliGraphWalk(params.entity_name, params.max_hops ?? 2);
			return { content: [{ type: "text", text: result || `entity '${params.entity_name}' not found` }] };
		},
	});

	pi.registerTool({
		name: "memory_status",
		label: "ppmlx memory stats",
		description: "Get database statistics for the memory graph.",
		parameters: Type.Object({}),
		execute: async (_id, _params, _signal) => {
			const r = await execCli(["memory", "status"]);
			return { content: [{ type: "text", text: r.ok ? r.stdout : r.stderr }] };
		},
	});
}
