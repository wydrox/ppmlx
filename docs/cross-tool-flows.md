# ppmlx Memory: Cross-Tool Integration Flows

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    ppmlx-memory-mcp                     │
│                  (localhost, stdio/HTTP)                │
│                                                         │
│  memory.db ←── extraction pipeline (v2, async)          │
│            ←── graph projection + inference             │
│            ──→ search / context retrieval               │
└──────┬──────────────┬──────────────┬───────────────────┘
       │              │              │
       ▼              ▼              ▼
    ┌─────┐      ┌──────────┐   ┌──────────┐
    │ pi  │      │Claude Code│   │  Codex   │
    │     │      │          │   │          │
    │ LLM:│      │ LLM:     │   │ LLM:     │
    │ppmlx│      │Anthropic │   │ OpenAI   │
    └─────┘      └──────────┘   └──────────┘
```

ppmlx memory działa jako **osobny proces** — nie wymaga inference. Retrieval to czysty SQLite.
Ekstrakcja używa MLX tylko gdy potrzebna (async, w tle). Każdy klient (pi, Claude Code, Codex)
widzi ten sam graf pamięci dla tego samego `project_id`.

---

## Flow 1: pi (full local stack)

pi używa ppmlx zarówno do inferencji JAK i do pamięci. Oba procesy lokalne.

```
SESSION START
─────────────────────────────────────────────────────────
1. pi ładuje extensję ppmlx-memory
2. pi → memory_get_context("ppmlx", session="...")
3. ppmlx zwraca skompaktowany kontekst:
   "Current workflow: implementing memory v2 pipeline
    Last decision: use gemma-4-e4b-it-optiq for extraction
    Constraint: extraction prompt <500 tokens
    Open todo: add few-shot examples"
4. pi wstrzykuje to do system promptu przed pierwszą wiadomością

USER TURN
─────────────────────────────────────────────────────────
5. User: "lets add the dense chunker now"
6. pi → memory_search("dense chunker", project="ppmlx")
7. ppmlx zwraca relewantne fakty:
   "fact: ppmlx uses qwen3-embedding for embeddings"
   "decision: dense_chunker uses sliding windows"
8. pi używa tego jako dodatkowego kontekstu przy generowaniu odpowiedzi
9. pi → ppmlx (inference): generuje odpowiedź

AFTER TURN (async, background)
─────────────────────────────────────────────────────────
10. pi → memory_record_event(messages, project="ppmlx", session="...")
11. ppmlx odpala pipeline v2:
    dense_chunk → contrastive → classify → extract → validate → graph → inference
12. Nowe fakty zapisane w memory.db

SESSION END
─────────────────────────────────────────────────────────
13. pi → memory_extract(full_transcript, project="ppmlx")
14. Pełna ekstrakcja całej sesji (dla faktów które umknęły per-turn)

NEXT SESSION (następny dzień)
─────────────────────────────────────────────────────────
15. pi → memory_get_context("ppmlx")
16. ppmlx zwraca:
    "Current workflow: dense_chunker implemented, contrastive retriever fixed
     Last commit: abc123 — fix empty index handling
     Decision: use gemma-4-e4b-it-optiq for extraction, qwen3-embedding for embeddings
     Constraint: all embedding errors handled gracefully"
17. pi kontynuuje pracę z pełnym kontekstem poprzedniej sesji
```

**W tym flow:** pi robi inference przez ppmlx (lokalnie, gemma-4-e4b), a pamięć działa
jako side-effect. User nie widzi pipeline'u — po prostu pi "pamięta" między sesjami.

---

## Flow 2: Claude Code (remote LLM, local memory)

Claude Code używa API Anthropic do inferencji. ppmlx-memory działa lokalnie tylko jako
warstwa pamięci.

```
SESSION START
─────────────────────────────────────────────────────────
1. Claude Code startuje w ~/dev/tview
2. Claude Code → ppmlx-memory: memory_get_context("tview")
3. ppmlx zwraca:
   "Current workflow: fixing invite-candidate dialog
    Last validation: pnpm build ✅, eslint ✅
    Decision: use Convex for auth, not Clerk directly
    Constraint: brutalist design system — no rounded corners"
4. Claude Code wstrzykuje to do system promptu (via CLAUDE.md lub podobny mechanizm)

USER TURN
─────────────────────────────────────────────────────────
5. User: "the invite dialog is not showing validation errors"
6. Claude Code → Anthropic API: wysyła prompt z wstrzykniętym kontekstem pamięci
7. Claude Code (po otrzymaniu odpowiedzi) → ppmlx-memory:
   memory_record_event(messages, project="tview")

ALTERNATIVNIE (via MCP tools):
─────────────────────────────────────────────────────────
8. Claude Code może bezpośrednio wywołać tool:
   memory_search("invite dialog validation", project="tview")
   → zwraca relewantne fakty o invite dialog
9. Claude używa tego w swojej odpowiedzi

SESSION END
─────────────────────────────────────────────────────────
10. Claude Code → ppmlx-memory: memory_extract(full_transcript, project="tview")
```

**W tym flow:** Claude Code robi inference przez Anthropic API, ale transkrypt
konwersacji jest wysyłany do lokalnego ppmlx-memory dla ekstrakcji. Kontekst z
poprzednich sesji (zarówno pi jak i Claude Code) jest dostępny dla tego samego
`project_id`.

---

## Flow 3: Codex (OpenAI CLI, local memory)

Identyczny pattern jak Claude Code — inference przez OpenAI API, pamięć lokalnie.

```
Codex → OpenAI API (inference)
Codex → ppmlx-memory (context retrieval)
Codex → ppmlx-memory (transcript extraction, async)
```

---

## Cross-Tool Memory Example

Najsilniejszy use case: developer używa różnych narzędzi do tego samego projektu.

```
DZIEŃ 1 — pi (ppmlx project)
─────────────────────────────────────────────────────────
pi: "zaimplementuj dense chunker"
pi: "użyj qwen3-embedding do embeddingów"
→ memory.db: fact: ppmlx uses qwen3-embedding:0.6b-4bit-dwq
→ memory.db: decision: dense_chunker uses sliding windows with 200-token stride

DZIEŃ 2 — Claude Code (ppmlx project)
─────────────────────────────────────────────────────────
Claude Code → memory_get_context("ppmlx")
→ "fact: ppmlx uses qwen3-embedding:0.6b-4bit-dwq"
→ "decision: dense_chunker uses sliding windows with 200-token stride"
Claude Code: "ah, już macie qwen3-embedding, to użyjmy go też do contrastive"

DZIEŃ 3 — Codex (ppmlx project)
─────────────────────────────────────────────────────────
Codex → memory_get_context("ppmlx")  
→ wszystkie fakty z pi i Claude Code dostępne
Codex: "wdrażam MCP server, widzę że contrastive retriever już działa"
```

**Kluczowa wartość:** Pamięć nie jest per-narzędzie. Jest per-projekt. Niezależnie
czy używasz pi, Claude Code, Cursor, czy Codex — wszystkie czytają i piszą do
tego samego grafu.

---

## Implementation Requirements

### Dla pi (natywna integracja):
- Extensja `~/.pi/agent/extensions/ppmlx-memory/` (wzorzec: martmart-mcp)
- Auto-inject context na start sesji
- Auto-extract po każdej turze (async, background)
- Konfiguracja: `project_id = cwd` lub mapowanie w settings.json

### Dla Claude Code / Codex (MCP):
- `ppmlx-memory-mcp` jako MCP server (stdio)
- Tools: `memory_search`, `memory_get_context`, `memory_record_event`, `memory_extract`
- Resources: `memory://{project}/context`, `memory://{project}/graph`
- Konfiguracja w `~/.claude/claude_desktop_config.json` lub `~/.codex/config.json`

### Wspólne:
- Jeden `memory.db` (domyślnie `~/.ppmlx/memory.db`)
- Project ID = nazwa katalogu lub explicit mapping
- Session ID = data+lub UUID per sesję
- Async extraction — nigdy nie blokuje głównej interakcji
