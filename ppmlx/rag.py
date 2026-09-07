"""RAG pipeline — Chat with your documents, privacy-first.

Provides document loading, chunking, vector storage (SQLite),
and retrieval-augmented generation using ppmlx's own embeddings and LLM.
"""
from __future__ import annotations

import hashlib
import math
import sqlite3
import struct
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _get_rag_db_path() -> Path:
    try:
        from ppmlx.config import get_ppmlx_dir
        return get_ppmlx_dir() / "rag.db"
    except ImportError:
        return Path.home() / ".ppmlx" / "rag.db"


DEFAULT_CHUNK_SIZE = 500      # characters (~125 tokens)
DEFAULT_CHUNK_OVERLAP = 50    # characters
DEFAULT_TOP_K = 5


# ---------------------------------------------------------------------------
# Document loading
# ---------------------------------------------------------------------------

SUPPORTED_EXTENSIONS = {
    ".txt", ".md", ".markdown",
    ".py", ".js", ".ts", ".jsx", ".tsx",
    ".rs", ".go", ".java", ".c", ".cpp", ".h", ".hpp",
    ".rb", ".php", ".swift", ".kt", ".scala",
    ".sh", ".bash", ".zsh",
    ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg",
    ".html", ".css", ".xml", ".csv",
    ".r", ".sql", ".lua",
    ".tex", ".rst",
    ".pdf",
}


def _read_pdf(path: Path) -> str:
    """Extract text from a PDF file using the built-in pdfminer or fallback."""
    try:
        from pdfminer.high_level import extract_text
        return extract_text(str(path))
    except ImportError:
        raise ImportError(
            "PDF support requires 'pdfminer.six'.\n"
            "Install with: pip install pdfminer.six"
        )


def load_document(path: Path) -> str:
    """Load a single document and return its text content."""
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {suffix}")

    if suffix == ".pdf":
        return _read_pdf(path)

    # Text-based files
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="latin-1")


def discover_files(directory: Path) -> list[Path]:
    """Recursively find all supported files in a directory."""
    if directory.is_file():
        if directory.suffix.lower() in SUPPORTED_EXTENSIONS:
            return [directory]
        return []

    files = []
    for p in sorted(directory.rglob("*")):
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS:
            # Skip hidden dirs/files
            parts = p.relative_to(directory).parts
            if any(part.startswith(".") for part in parts):
                continue
            files.append(p)
    return files


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[str]:
    """Split text into overlapping chunks using recursive character splitting."""
    if not text.strip():
        return []

    # Try splitting by decreasing granularity
    separators = ["\n\n", "\n", ". ", " ", ""]

    def _split(text: str, seps: list[str]) -> list[str]:
        if not seps:
            # Base case: hard split at chunk_size
            hard_chunks: list[str] = []
            for i in range(0, len(text), chunk_size - chunk_overlap):
                chunk = text[i : i + chunk_size]
                if chunk.strip():
                    hard_chunks.append(chunk.strip())
            return hard_chunks

        sep = seps[0]
        remaining_seps = seps[1:]

        if sep:
            parts = text.split(sep)
        else:
            return _split(text, remaining_seps)

        # Merge small parts into chunks
        chunks: list[str] = []
        current = ""
        for part in parts:
            candidate = (current + sep + part).strip() if current else part.strip()
            if len(candidate) <= chunk_size:
                current = candidate
            else:
                if current.strip():
                    chunks.append(current.strip())
                # If the part itself is too long, recurse with finer separator
                if len(part) > chunk_size:
                    sub_chunks = _split(part, remaining_seps)
                    chunks.extend(sub_chunks)
                    current = ""
                else:
                    current = part.strip()
        if current.strip():
            chunks.append(current.strip())

        # Apply overlap: prepend tail of previous chunk to each chunk
        if chunk_overlap > 0 and len(chunks) > 1:
            overlapped = [chunks[0]]
            for i in range(1, len(chunks)):
                prev_tail = chunks[i - 1][-chunk_overlap:]
                merged = prev_tail + " " + chunks[i]
                overlapped.append(merged.strip())
            return overlapped

        return chunks

    return _split(text, separators)


# ---------------------------------------------------------------------------
# Vector encoding helpers (pure Python, no numpy required at import time)
# ---------------------------------------------------------------------------

def _encode_vector(vec: list[float]) -> bytes:
    """Encode a float vector as a compact bytes blob (little-endian float32)."""
    return struct.pack(f"<{len(vec)}f", *vec)


def _decode_vector(blob: bytes) -> list[float]:
    """Decode a bytes blob back to a float vector."""
    n = len(blob) // 4
    return list(struct.unpack(f"<{n}f", blob))


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors (pure Python)."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# ---------------------------------------------------------------------------
# Vector store (SQLite-backed)
# ---------------------------------------------------------------------------

_RAG_SCHEMA = """
CREATE TABLE IF NOT EXISTS collections (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL UNIQUE,
    embed_model TEXT NOT NULL,
    created_at  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%f', 'now'))
);

CREATE TABLE IF NOT EXISTS documents (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    collection_id   INTEGER NOT NULL REFERENCES collections(id) ON DELETE CASCADE,
    source_path     TEXT NOT NULL,
    file_hash       TEXT NOT NULL,
    ingested_at     TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%f', 'now')),
    UNIQUE(collection_id, source_path)
);

CREATE TABLE IF NOT EXISTS chunks (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id   INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index   INTEGER NOT NULL,
    content       TEXT NOT NULL,
    embedding     BLOB NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_documents_collection ON documents(collection_id);
"""


class VectorStore:
    """SQLite-backed vector store for RAG document chunks."""

    def __init__(self, db_path: Path | None = None):
        self._path = db_path or _get_rag_db_path()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self._path))
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._conn.executescript(_RAG_SCHEMA)
        return self._conn

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    # ── Collection management ───────────────────────────────────────────

    def create_collection(self, name: str, embed_model: str) -> int:
        """Create a new collection. Returns its ID."""
        conn = self._get_conn()
        cur = conn.execute(
            "INSERT INTO collections (name, embed_model) VALUES (?, ?)",
            (name, embed_model),
        )
        conn.commit()
        return cur.lastrowid  # type: ignore[return-value]

    def get_collection(self, name: str) -> dict[str, Any] | None:
        """Get collection metadata by name."""
        conn = self._get_conn()
        # The cursor type does not expose ``row_factory`` to mypy, but the
        # connection does. Use a short-lived typed connection for row mapping.
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        row = cur.execute(
            "SELECT * FROM collections WHERE name = ?", (name,)
        ).fetchone()
        if row is None:
            return None
        return dict(row)

    def list_collections(self) -> list[dict[str, Any]]:
        """List all collections with their doc/chunk counts."""
        conn = self._get_conn()
        rows = conn.execute("""
            SELECT c.id, c.name, c.embed_model, c.created_at,
                   COUNT(DISTINCT d.id) as doc_count,
                   COUNT(ch.id) as chunk_count
            FROM collections c
            LEFT JOIN documents d ON d.collection_id = c.id
            LEFT JOIN chunks ch ON ch.document_id = d.id
            GROUP BY c.id
            ORDER BY c.name
        """).fetchall()
        return [
            {
                "id": r[0], "name": r[1], "embed_model": r[2],
                "created_at": r[3], "doc_count": r[4], "chunk_count": r[5],
            }
            for r in rows
        ]

    def delete_collection(self, name: str) -> bool:
        """Delete a collection and all its data. Returns True if it existed."""
        conn = self._get_conn()
        coll = self.get_collection(name)
        if coll is None:
            return False
        conn.execute("DELETE FROM chunks WHERE document_id IN (SELECT id FROM documents WHERE collection_id = ?)", (coll["id"],))
        conn.execute("DELETE FROM documents WHERE collection_id = ?", (coll["id"],))
        conn.execute("DELETE FROM collections WHERE id = ?", (coll["id"],))
        conn.commit()
        return True

    # ── Document ingestion ──────────────────────────────────────────────

    def _file_hash(self, path: Path) -> str:
        """Compute a fast content hash for change detection."""
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()[:16]

    def _document_exists(self, collection_id: int, source_path: str, file_hash: str) -> bool:
        """Check if a document with the same path and hash already exists."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT file_hash FROM documents WHERE collection_id = ? AND source_path = ?",
            (collection_id, source_path),
        ).fetchone()
        return row is not None and row[0] == file_hash

    def add_document(
        self,
        collection_id: int,
        source_path: str,
        file_hash: str,
        chunks: list[str],
        embeddings: list[list[float]],
    ) -> int:
        """Add a document with its chunks and embeddings. Returns doc ID."""
        conn = self._get_conn()

        # Remove old version if it exists (allows re-ingestion)
        conn.execute(
            """DELETE FROM chunks WHERE document_id IN (
                SELECT id FROM documents WHERE collection_id = ? AND source_path = ?
            )""",
            (collection_id, source_path),
        )
        conn.execute(
            "DELETE FROM documents WHERE collection_id = ? AND source_path = ?",
            (collection_id, source_path),
        )

        cur = conn.execute(
            "INSERT INTO documents (collection_id, source_path, file_hash) VALUES (?, ?, ?)",
            (collection_id, source_path, file_hash),
        )
        doc_id: int = cur.lastrowid  # type: ignore[assignment]

        for i, (chunk_text, embedding) in enumerate(zip(chunks, embeddings)):
            conn.execute(
                "INSERT INTO chunks (document_id, chunk_index, content, embedding) VALUES (?, ?, ?, ?)",
                (doc_id, i, chunk_text, _encode_vector(embedding)),
            )

        conn.commit()
        return doc_id

    def get_documents(self, collection_id: int) -> list[dict[str, Any]]:
        """List documents in a collection."""
        conn = self._get_conn()
        rows = conn.execute(
            """SELECT d.id, d.source_path, d.file_hash, d.ingested_at,
                      COUNT(ch.id) as chunk_count
               FROM documents d
               LEFT JOIN chunks ch ON ch.document_id = d.id
               WHERE d.collection_id = ?
               GROUP BY d.id
               ORDER BY d.source_path""",
            (collection_id,),
        ).fetchall()
        return [
            {"id": r[0], "source_path": r[1], "file_hash": r[2],
             "ingested_at": r[3], "chunk_count": r[4]}
            for r in rows
        ]

    # ── Search ──────────────────────────────────────────────────────────

    def search(
        self,
        collection_id: int,
        query_embedding: list[float],
        top_k: int = DEFAULT_TOP_K,
    ) -> list[dict[str, Any]]:
        """Find top-k most similar chunks by cosine similarity."""
        conn = self._get_conn()
        rows = conn.execute(
            """SELECT ch.id, ch.content, ch.embedding, d.source_path
               FROM chunks ch
               JOIN documents d ON d.id = ch.document_id
               WHERE d.collection_id = ?""",
            (collection_id,),
        ).fetchall()

        if not rows:
            return []

        scored = []
        for row in rows:
            chunk_embedding = _decode_vector(row[2])
            sim = _cosine_similarity(query_embedding, chunk_embedding)
            scored.append({
                "chunk_id": row[0],
                "content": row[1],
                "source": row[3],
                "similarity": sim,
            })

        scored.sort(key=lambda x: x["similarity"], reverse=True)
        return scored[:top_k]


# ---------------------------------------------------------------------------
# RAG pipeline
# ---------------------------------------------------------------------------

def _get_embed_fn(embed_model: str):
    """Return a function that takes a list of texts and returns embeddings."""
    from ppmlx.engine_embed import get_embed_engine
    engine = get_embed_engine()

    def embed(texts: list[str]) -> list[list[float]]:
        return engine.encode(embed_model, texts)

    return embed


def ingest(
    path: Path,
    collection_name: str = "default",
    embed_model: str = "embed:all-minilm",
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    embed_fn=None,
    store: VectorStore | None = None,
) -> dict[str, Any]:
    """Ingest documents from a file or directory into a collection.

    Returns stats: {files_processed, chunks_created, files_skipped, errors}.
    """
    if embed_fn is None:
        embed_fn = _get_embed_fn(embed_model)

    if store is None:
        store = VectorStore()

    # Get or create collection
    coll = store.get_collection(collection_name)
    if coll is None:
        coll_id = store.create_collection(collection_name, embed_model)
    else:
        coll_id = coll["id"]
        embed_model = coll["embed_model"]  # use collection's embed model

    files = discover_files(path)
    stats: dict[str, Any] = {
        "files_processed": 0, "chunks_created": 0, "files_skipped": 0, "errors": []
    }

    for file_path in files:
        try:
            file_hash = store._file_hash(file_path)
            source = str(file_path.resolve())

            # Skip if unchanged
            if store._document_exists(coll_id, source, file_hash):
                stats["files_skipped"] += 1
                continue

            text = load_document(file_path)
            chunks = chunk_text(text, chunk_size, chunk_overlap)
            if not chunks:
                stats["files_skipped"] += 1
                continue

            # Batch embed
            embeddings = embed_fn(chunks)
            store.add_document(coll_id, source, file_hash, chunks, embeddings)

            stats["files_processed"] += 1
            stats["chunks_created"] += len(chunks)

        except Exception as e:
            stats["errors"].append({"file": str(file_path), "error": str(e)})

    return stats


def retrieve(
    query: str,
    collection_name: str = "default",
    top_k: int = DEFAULT_TOP_K,
    embed_fn=None,
    store: VectorStore | None = None,
) -> list[dict[str, Any]]:
    """Retrieve top-k relevant chunks for a query."""
    if store is None:
        store = VectorStore()

    coll = store.get_collection(collection_name)
    if coll is None:
        return []

    embed_model = coll["embed_model"]
    if embed_fn is None:
        embed_fn = _get_embed_fn(embed_model)

    query_embedding = embed_fn([query])[0]
    return store.search(coll["id"], query_embedding, top_k=top_k)


def build_rag_prompt(
    query: str,
    contexts: list[dict[str, Any]],
) -> str:
    """Build a prompt with retrieved context injected."""
    if not contexts:
        return query

    context_parts = []
    for i, ctx in enumerate(contexts, 1):
        source = Path(ctx["source"]).name
        context_parts.append(f"[{i}] (from {source}):\n{ctx['content']}")

    context_block = "\n\n".join(context_parts)

    return (
        "Use the following context to answer the question. "
        "If the context doesn't contain relevant information, say so.\n\n"
        f"--- Context ---\n{context_block}\n--- End Context ---\n\n"
        f"Question: {query}"
    )
