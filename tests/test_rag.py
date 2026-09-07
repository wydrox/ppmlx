"""Tests for the RAG pipeline — document loading, chunking, vector store, retrieval."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Ensure ppmlx.rag module mock is available for CLI tests
for mod_name in ["ppmlx.models", "ppmlx.engine", "ppmlx.db",
                 "ppmlx.config", "ppmlx.memory", "ppmlx.modelfile",
                 "ppmlx.quantize", "ppmlx.engine_embed", "ppmlx.engine_vlm",
                 "ppmlx.registry"]:
    if mod_name not in sys.modules:
        sys.modules[mod_name] = MagicMock()

from ppmlx.rag import (
    chunk_text,
    load_document,
    discover_files,
    VectorStore,
    ingest,
    retrieve,
    build_rag_prompt,
    _encode_vector,
    _decode_vector,
    _cosine_similarity,
    SUPPORTED_EXTENSIONS,
)

import hashlib as _hashlib


def _mock_embed_fn(texts: list[str]) -> list[list[float]]:
    """Deterministic mock embedding: hash text to produce a fixed-length vector."""
    results = []
    for t in texts:
        h = _hashlib.md5(t.encode()).hexdigest()
        vec = [int(h[i:i+2], 16) / 255.0 for i in range(0, 16, 2)]
        results.append(vec)
    return results


# ---------------------------------------------------------------------------
# chunk_text tests
# ---------------------------------------------------------------------------

class TestChunkText:
    def test_empty_string(self):
        assert chunk_text("") == []

    def test_whitespace_only(self):
        assert chunk_text("   \n\n   ") == []

    def test_short_text_single_chunk(self):
        text = "Hello world"
        chunks = chunk_text(text, chunk_size=500, chunk_overlap=0)
        assert len(chunks) == 1
        assert chunks[0] == "Hello world"

    def test_long_text_splits(self):
        # Build a text longer than chunk_size
        text = "word " * 200  # 1000 chars
        chunks = chunk_text(text, chunk_size=100, chunk_overlap=0)
        assert len(chunks) > 1
        # All chunks should be non-empty
        for c in chunks:
            assert len(c.strip()) > 0

    def test_paragraph_splitting(self):
        text = "Paragraph one content here.\n\nParagraph two content here.\n\nParagraph three."
        chunks = chunk_text(text, chunk_size=500, chunk_overlap=0)
        # Short enough to fit in one chunk
        assert len(chunks) >= 1

    def test_overlap_produces_more_text(self):
        text = "First section content. " * 50
        chunks_no_overlap = chunk_text(text, chunk_size=100, chunk_overlap=0)
        chunks_with_overlap = chunk_text(text, chunk_size=100, chunk_overlap=20)
        # Overlap chunks should contain some repeated content
        assert len(chunks_with_overlap) >= len(chunks_no_overlap)


# ---------------------------------------------------------------------------
# Vector encoding tests
# ---------------------------------------------------------------------------

class TestVectorEncoding:
    def test_roundtrip(self):
        vec = [0.1, 0.2, 0.3, -0.5, 1.0]
        blob = _encode_vector(vec)
        decoded = _decode_vector(blob)
        assert len(decoded) == len(vec)
        for a, b in zip(vec, decoded):
            assert abs(a - b) < 1e-6

    def test_empty_vector(self):
        vec: list[float] = []
        blob = _encode_vector(vec)
        assert _decode_vector(blob) == []


# ---------------------------------------------------------------------------
# Cosine similarity tests
# ---------------------------------------------------------------------------

class TestCosineSimilarity:
    def test_identical_vectors(self):
        v = [1.0, 2.0, 3.0]
        assert abs(_cosine_similarity(v, v) - 1.0) < 1e-6

    def test_orthogonal_vectors(self):
        assert abs(_cosine_similarity([1.0, 0.0], [0.0, 1.0])) < 1e-6

    def test_opposite_vectors(self):
        assert abs(_cosine_similarity([1.0, 0.0], [-1.0, 0.0]) + 1.0) < 1e-6

    def test_zero_vector(self):
        assert _cosine_similarity([0.0, 0.0], [1.0, 2.0]) == 0.0


# ---------------------------------------------------------------------------
# Document loading tests
# ---------------------------------------------------------------------------

class TestDocumentLoading:
    def test_load_txt_file(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("Hello world")
        assert load_document(f) == "Hello world"

    def test_load_md_file(self, tmp_path):
        f = tmp_path / "test.md"
        f.write_text("# Title\nContent")
        assert load_document(f) == "# Title\nContent"

    def test_load_py_file(self, tmp_path):
        f = tmp_path / "test.py"
        f.write_text("print('hello')")
        assert load_document(f) == "print('hello')"

    def test_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_document(tmp_path / "nonexistent.txt")

    def test_unsupported_extension(self, tmp_path):
        f = tmp_path / "test.xyz"
        f.write_text("data")
        with pytest.raises(ValueError, match="Unsupported"):
            load_document(f)

    def test_discover_files(self, tmp_path):
        (tmp_path / "a.txt").write_text("a")
        (tmp_path / "b.py").write_text("b")
        (tmp_path / "c.jpg").write_text("img")  # unsupported
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "d.md").write_text("d")
        # Hidden dir should be skipped
        hidden = tmp_path / ".hidden"
        hidden.mkdir()
        (hidden / "e.txt").write_text("e")

        files = discover_files(tmp_path)
        names = {f.name for f in files}
        assert "a.txt" in names
        assert "b.py" in names
        assert "d.md" in names
        assert "c.jpg" not in names
        assert "e.txt" not in names

    def test_discover_single_file(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hello")
        assert discover_files(f) == [f]


# ---------------------------------------------------------------------------
# VectorStore tests
# ---------------------------------------------------------------------------

class TestVectorStore:
    def test_create_and_get_collection(self, tmp_path):
        store = VectorStore(tmp_path / "test.db")
        coll_id = store.create_collection("test", "embed:all-minilm")
        assert coll_id is not None

        coll = store.get_collection("test")
        assert coll is not None
        assert coll["name"] == "test"
        assert coll["embed_model"] == "embed:all-minilm"

    def test_list_collections(self, tmp_path):
        store = VectorStore(tmp_path / "test.db")
        store.create_collection("coll1", "embed:all-minilm")
        store.create_collection("coll2", "embed:all-minilm")
        colls = store.list_collections()
        assert len(colls) == 2
        names = {c["name"] for c in colls}
        assert names == {"coll1", "coll2"}

    def test_delete_collection(self, tmp_path):
        store = VectorStore(tmp_path / "test.db")
        store.create_collection("test", "embed:all-minilm")
        assert store.delete_collection("test") is True
        assert store.get_collection("test") is None
        assert store.delete_collection("test") is False

    def test_add_document_and_search(self, tmp_path):
        store = VectorStore(tmp_path / "test.db")
        coll_id = store.create_collection("test", "embed:all-minilm")

        chunks = ["Python is a programming language", "Java is also a language"]
        # Fake embeddings: first chunk has high x, second has high y
        embeddings = [[0.9, 0.1, 0.0], [0.1, 0.9, 0.0]]

        doc_id = store.add_document(coll_id, "/tmp/test.txt", "abc123", chunks, embeddings)
        assert doc_id is not None

        # Query similar to first chunk
        results = store.search(coll_id, [0.95, 0.05, 0.0], top_k=1)
        assert len(results) == 1
        assert "Python" in results[0]["content"]

        # Query similar to second chunk
        results = store.search(coll_id, [0.05, 0.95, 0.0], top_k=1)
        assert len(results) == 1
        assert "Java" in results[0]["content"]

    def test_document_re_ingestion(self, tmp_path):
        store = VectorStore(tmp_path / "test.db")
        coll_id = store.create_collection("test", "embed:all-minilm")

        # Ingest once
        store.add_document(coll_id, "/tmp/test.txt", "hash1", ["old content"], [[0.5, 0.5]])

        # Re-ingest with new hash
        store.add_document(coll_id, "/tmp/test.txt", "hash2", ["new content"], [[0.8, 0.2]])

        docs = store.get_documents(coll_id)
        assert len(docs) == 1
        assert docs[0]["file_hash"] == "hash2"

    def test_search_empty_collection(self, tmp_path):
        store = VectorStore(tmp_path / "test.db")
        coll_id = store.create_collection("test", "embed:all-minilm")
        results = store.search(coll_id, [0.5, 0.5], top_k=5)
        assert results == []


# ---------------------------------------------------------------------------
# Ingest pipeline tests (with mocked embeddings)
# ---------------------------------------------------------------------------

class TestIngestPipeline:
    def test_ingest_single_file(self, tmp_path):
        doc_file = tmp_path / "test.txt"
        doc_file.write_text("This is a test document with some content for RAG.")

        store = VectorStore(tmp_path / "rag.db")
        stats = ingest(
            path=doc_file,
            collection_name="test",
            embed_fn=_mock_embed_fn,
            store=store,
        )

        assert stats["files_processed"] == 1
        assert stats["chunks_created"] >= 1
        assert stats["errors"] == []

    def test_ingest_directory(self, tmp_path):
        (tmp_path / "a.txt").write_text("Document A content " * 20)
        (tmp_path / "b.md").write_text("# Document B\n\nSome markdown content " * 20)
        sub = tmp_path / "subdir"
        sub.mkdir()
        (sub / "c.py").write_text("# Python file\nprint('hello')\n" * 10)

        store = VectorStore(tmp_path / "rag.db")
        stats = ingest(
            path=tmp_path,
            collection_name="test",
            embed_fn=_mock_embed_fn,
            store=store,
        )

        assert stats["files_processed"] == 3
        assert stats["chunks_created"] >= 3
        assert stats["errors"] == []

    def test_ingest_skips_unchanged_files(self, tmp_path):
        doc_file = tmp_path / "test.txt"
        doc_file.write_text("Same content")

        store = VectorStore(tmp_path / "rag.db")

        # First ingest
        stats1 = ingest(path=doc_file, collection_name="test",
                        embed_fn=_mock_embed_fn, store=store)
        assert stats1["files_processed"] == 1

        # Second ingest — should skip
        stats2 = ingest(path=doc_file, collection_name="test",
                        embed_fn=_mock_embed_fn, store=store)
        assert stats2["files_skipped"] == 1
        assert stats2["files_processed"] == 0

    def test_ingest_reprocesses_changed_files(self, tmp_path):
        doc_file = tmp_path / "test.txt"
        doc_file.write_text("Original content")

        store = VectorStore(tmp_path / "rag.db")
        ingest(path=doc_file, collection_name="test",
               embed_fn=_mock_embed_fn, store=store)

        # Change the file
        doc_file.write_text("Updated content")
        stats = ingest(path=doc_file, collection_name="test",
                       embed_fn=_mock_embed_fn, store=store)
        assert stats["files_processed"] == 1


# ---------------------------------------------------------------------------
# Retrieve tests
# ---------------------------------------------------------------------------

class TestRetrieve:
    def test_retrieve_from_empty_collection(self, tmp_path):
        store = VectorStore(tmp_path / "rag.db")
        results = retrieve("query", collection_name="nonexistent", store=store)
        assert results == []

    def test_retrieve_returns_relevant_chunks(self, tmp_path):
        store = VectorStore(tmp_path / "rag.db")

        # Ingest some documents
        (tmp_path / "doc.txt").write_text("Python programming is great.\n\nJava is verbose.")
        ingest(path=tmp_path / "doc.txt", collection_name="test",
               embed_fn=_mock_embed_fn, store=store)

        # Retrieve
        results = retrieve("Python", collection_name="test",
                           embed_fn=_mock_embed_fn, store=store, top_k=2)
        assert len(results) >= 1
        for r in results:
            assert "content" in r
            assert "source" in r
            assert "similarity" in r


# ---------------------------------------------------------------------------
# RAG prompt building tests
# ---------------------------------------------------------------------------

class TestBuildRagPrompt:
    def test_no_context(self):
        prompt = build_rag_prompt("What is Python?", [])
        assert prompt == "What is Python?"

    def test_with_context(self):
        contexts = [
            {"content": "Python is a language.", "source": "/tmp/doc.txt", "similarity": 0.9},
            {"content": "It was created by Guido.", "source": "/tmp/doc2.txt", "similarity": 0.8},
        ]
        prompt = build_rag_prompt("What is Python?", contexts)
        assert "Python is a language." in prompt
        assert "It was created by Guido." in prompt
        assert "What is Python?" in prompt
        assert "Context" in prompt


# ---------------------------------------------------------------------------
# CLI integration tests
# ---------------------------------------------------------------------------

class TestRagCli:
    def test_rag_help(self):
        from typer.testing import CliRunner
        from ppmlx.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["rag", "--help"])
        assert result.exit_code == 0
        assert "rag" in result.output.lower() or "document" in result.output.lower()

    def test_rag_list_empty(self):
        from typer.testing import CliRunner
        from ppmlx.cli import app

        runner = CliRunner()
        # Patch VectorStore to return no collections
        with pytest.MonkeyPatch.context() as mp:
            mock_store_cls = MagicMock()
            mock_store_instance = MagicMock()
            mock_store_instance.list_collections.return_value = []
            mock_store_cls.return_value = mock_store_instance

            mp.setattr("ppmlx.rag.VectorStore", mock_store_cls)
            result = runner.invoke(app, ["rag", "list"])
            assert result.exit_code == 0
            assert "No RAG collections" in result.output

    def test_rag_ingest_missing_path(self):
        from typer.testing import CliRunner
        from ppmlx.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["rag", "ingest", "/nonexistent/path/xyz"])
        assert result.exit_code == 1

    def test_rag_rm_not_found(self):
        from typer.testing import CliRunner
        from ppmlx.cli import app

        runner = CliRunner()
        with pytest.MonkeyPatch.context() as mp:
            mock_store_cls = MagicMock()
            mock_store_instance = MagicMock()
            mock_store_instance.get_collection.return_value = None
            mock_store_cls.return_value = mock_store_instance

            mp.setattr("ppmlx.rag.VectorStore", mock_store_cls)
            result = runner.invoke(app, ["rag", "rm", "nonexistent"])
            assert result.exit_code == 1
