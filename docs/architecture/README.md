# ppmlx architecture decisions

This directory contains the accepted architecture decisions for the ppmlx proxy.
Phase 1 freezes these contracts before runtime work starts.

ppmlx keeps local MLX inference as a first-class route.
It also provides one local base URL for supported agent harnesses.
The local server can route a request to a local model or an approved remote provider.

## Accepted decisions

- [ADR 0001: Product Boundary](adr/0001-product-boundary.md)
- [ADR 0002: Agent IR](adr/0002-agent-ir.md)
- [ADR 0003: Tool Execution](adr/0003-tool-execution.md)
- [ADR 0004: Provider Authentication](adr/0004-provider-authentication.md)

The following accepted decisions complete the Phase 1 contract:

- [ADR 0005: Routing and Fallback](adr/0005-routing-and-fallback.md)
- [ADR 0006: Memory Capture and Read](adr/0006-memory-capture-and-read.md)
- [ADR 0007: Retention and Redaction](adr/0007-retention-and-redaction.md)
- [ADR 0008: Compatibility](adr/0008-compatibility.md)

## Accepted amendments

- [Amendment 0001: Bounded Tool-Argument Repair](amendments/0001-bounded-tool-argument-repair.md)

Amendment 0001 resolves the Phase 4 repair policy.
It amends ADR 0003 and clarifies the Agent IR acceptance boundary in ADR 0002.
A later accepted amendment takes precedence only for the rules that it explicitly changes.

The [contract fixture manifest](../../tests/fixtures/contracts/manifest.json) records the exact harness and protocol versions for Phase 1.
The [Agent IR v1 JSON Schema](schema/agent-ir-v1.schema.json) defines the normalized request and event shapes.
The [local Agent IR runtime guide](local-agent-runtime.md) defines the Phase 4 implementation boundary and its limits.

## Contract rules

- These ADRs and amendments define contracts.
- A contract document does not add runtime behavior by itself.
- A later phase must add contract tests before it changes a frozen contract.
- A new contract decision must replace or explicitly amend an accepted decision.
- An implementation must not silently change an accepted decision.
- An amendment must name the decisions and rules that it changes.

Protocol rules apply to all adapters:

- Protocol adapters must keep information that affects request or response meaning.
- Unsupported behavior must cause a clear error.
- An adapter must not silently remove data.
- Provider credentials must stay separate from harness credentials and local endpoint credentials.

## Terms

**Harness** means Claude Code, Codex, OpenCode, Pi, or another supported agent client.

**Provider** means a local inference engine or an approved remote model service.

**Adapter** means a component that maps one protocol to or from the Agent IR.

**Agent IR** means the internal, protocol-neutral representation in ADR 0002.

**Model profile** means a versioned contract for model capabilities, tokenizer behavior, output normalization, and optional bounded repair.

**Local endpoint** means one ppmlx base URL and listener. It can expose multiple compatible HTTP paths.

**Tool owner** means the harness that receives a tool call and runs the tool.
