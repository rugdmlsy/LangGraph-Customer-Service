"""
tests/test_rag.py

Unit tests for the RAG pipeline components.

Tests cover:
    - SlidingWindowChunker: boundary conditions and chunk overlap correctness.
    - Embedder: output shape, normalisation, empty input handling.
    - Retriever: hybrid_search logic, rerank ordering, BM25 scoring.

Mocking strategy:
    - Milvus collection: mock collection.search() to return controlled hits.
    - SentenceTransformer: mock encode() to return deterministic vectors.
    - CrossEncoder: mock predict() to return controlled scores.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from rag.chunker import SlidingWindowChunker
from rag.embedder import Embedder
from rag.retriever import Retriever


# --------------------------------------------------------------------------- #
# Chunker tests                                                                #
# --------------------------------------------------------------------------- #


class TestSlidingWindowChunker:
    """Tests for SlidingWindowChunker."""

    def test_chunker_short_text_returns_single_chunk(self):
        """
        Text shorter than chunk_size should return a single-element list.

        How to implement:
            chunker = SlidingWindowChunker(chunk_size=512, overlap=100)
            text = "短文本"
            chunks = chunker.chunk(text)
            assert len(chunks) == 1
            assert chunks[0] == text

        TODO: implement test body
        """
        # TODO: implement
        pass

    def test_chunker_sliding_window(self):
        """
        Verify that consecutive chunks share exactly `overlap` characters.

        How to implement:
            chunk_size, overlap = 100, 20
            chunker = SlidingWindowChunker(chunk_size=chunk_size, overlap=overlap)
            text = "A" * 250
            chunks = chunker.chunk(text)
            # Second chunk should start where first chunk ends minus overlap
            stride = chunk_size - overlap
            assert chunks[1][:overlap] == chunks[0][-overlap:]

        TODO: implement test body
        """
        # TODO: implement
        pass

    def test_chunker_empty_text_returns_empty_list(self):
        """
        Empty or whitespace-only input should return [].

        How to implement:
            chunker = SlidingWindowChunker()
            assert chunker.chunk("") == []
            assert chunker.chunk("   ") == []

        TODO: implement test body
        """
        # TODO: implement
        pass

    def test_chunker_chunk_size_must_exceed_overlap(self):
        """
        SlidingWindowChunker should raise ValueError if chunk_size <= overlap.

        How to implement:
            with pytest.raises(ValueError):
                SlidingWindowChunker(chunk_size=100, overlap=100)

        TODO: implement test body
        """
        # TODO: implement
        pass

    def test_chunker_with_metadata_returns_correct_structure(self):
        """
        chunk_with_metadata should attach source and chunk_id to each chunk.

        How to implement:
            chunker = SlidingWindowChunker(chunk_size=50, overlap=10)
            result = chunker.chunk_with_metadata("A" * 120, source="test.txt")
            assert all("text" in r and "chunk_id" in r and "source" in r for r in result)
            assert all(r["source"] == "test.txt" for r in result)
            assert [r["chunk_id"] for r in result] == list(range(len(result)))

        TODO: implement test body
        """
        # TODO: implement
        pass


# --------------------------------------------------------------------------- #
# Embedder tests                                                               #
# --------------------------------------------------------------------------- #


class TestEmbedder:
    """Tests for the Embedder class."""

    @patch("rag.embedder.Embedder._load_model")
    def test_embedder_output_shape(self, mock_load):
        """
        embed() should return one vector per input text.

        How to implement:
            embedder = Embedder()
            # Mock the loaded model's encode() method
            fake_vectors = np.random.rand(3, 384).astype(np.float32)
            embedder.model = MagicMock()
            embedder.model.encode.return_value = fake_vectors
            result = embedder.embed(["a", "b", "c"])
            assert len(result) == 3
            assert len(result[0]) == 384

        TODO: implement test body
        """
        # TODO: implement
        pass

    @patch("rag.embedder.Embedder._load_model")
    def test_embedder_empty_input_returns_empty_list(self, mock_load):
        """
        embed([]) should return [] without calling the model.

        How to implement:
            embedder = Embedder()
            embedder.model = MagicMock()
            result = embedder.embed([])
            assert result == []
            embedder.model.encode.assert_not_called()

        TODO: implement test body
        """
        # TODO: implement
        pass

    @patch("rag.embedder.Embedder._load_model")
    def test_embedder_embed_query_returns_single_vector(self, mock_load):
        """
        embed_query() should return a flat list (not a list of lists).

        How to implement:
            embedder = Embedder()
            embedder.model = MagicMock()
            embedder.model.encode.return_value = np.random.rand(1, 384).astype(np.float32)
            result = embedder.embed_query("test query")
            assert isinstance(result, list)
            assert isinstance(result[0], float)
            assert len(result) == 384

        TODO: implement test body
        """
        # TODO: implement
        pass

    @patch("rag.embedder.Embedder._load_model")
    def test_embedder_vectors_are_normalised(self, mock_load):
        """
        With normalize_embeddings=True, vectors should have unit L2 norm.

        How to implement:
            embedder = Embedder()
            # Return pre-normalised vectors from mock
            vec = np.array([[1/np.sqrt(2), 1/np.sqrt(2)] + [0]*382], dtype=np.float32)
            embedder.model = MagicMock()
            embedder.model.encode.return_value = vec
            result = embedder.embed(["test"])
            norm = sum(x**2 for x in result[0]) ** 0.5
            assert abs(norm - 1.0) < 1e-5

        TODO: implement test body
        """
        # TODO: implement
        pass


# --------------------------------------------------------------------------- #
# Retriever tests                                                              #
# --------------------------------------------------------------------------- #


class TestRetriever:
    """Tests for the hybrid retriever."""

    @pytest.fixture
    def mock_knowledge_base(self):
        """Mock KnowledgeBase with a mock Milvus collection."""
        kb = MagicMock()
        # Simulate a Milvus search result
        hit = MagicMock()
        hit.id = 1
        hit.score = 0.92
        hit.entity.get.side_effect = lambda k: {"text": "FAQ text", "source": "faq.txt"}.get(k)
        kb.collection.search.return_value = [[hit]]
        return kb

    @pytest.fixture
    def mock_embedder(self):
        """Mock Embedder that returns a deterministic vector."""
        embedder = MagicMock()
        embedder.embed_query.return_value = [0.1] * 384
        return embedder

    def test_retriever_hybrid_search(self, mock_knowledge_base, mock_embedder):
        """
        hybrid_search should call both dense_search and sparse_search and
        return a non-empty list.

        How to implement:
            retriever = Retriever(mock_knowledge_base, mock_embedder)
            retriever.dense_search  = MagicMock(return_value=[{"id": 1, "text": "a", "score": 0.9}])
            retriever.sparse_search = MagicMock(return_value=[{"id": 1, "text": "a", "score": 5.0}])
            results = retriever.hybrid_search("test query", top_k=5)
            retriever.dense_search.assert_called_once()
            retriever.sparse_search.assert_called_once()
            assert len(results) > 0

        TODO: implement test body
        """
        # TODO: implement
        pass

    def test_retriever_rerank_sorts_by_score(self, mock_knowledge_base, mock_embedder):
        """
        rerank() should return documents sorted by cross-encoder score descending.

        How to implement:
            retriever = Retriever(mock_knowledge_base, mock_embedder)
            retriever.reranker = MagicMock()
            retriever.reranker.predict.return_value = np.array([0.3, 0.9, 0.6])
            docs = [{"text": "low"}, {"text": "high"}, {"text": "mid"}]
            result = retriever.rerank("q", docs, top_n=2)
            assert result[0]["text"] == "high"
            assert len(result) == 2

        TODO: implement test body
        """
        # TODO: implement
        pass

    def test_retriever_dense_search_parses_milvus_results(self, mock_knowledge_base, mock_embedder):
        """
        dense_search() should correctly parse pymilvus Hit objects.

        How to implement:
            retriever = Retriever(mock_knowledge_base, mock_embedder)
            results = retriever.dense_search([0.1] * 384, top_k=1)
            assert len(results) == 1
            assert results[0]["text"] == "FAQ text"
            assert results[0]["score"] == 0.92

        TODO: implement test body
        """
        # TODO: implement
        pass
