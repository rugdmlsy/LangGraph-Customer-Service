"""
rag/retriever.py

Hybrid retrieval engine using Milvus native hybrid search + cross-encoder reranking.

Retrieval strategy:
    Stage 1  — Milvus hybrid_search: combines dense ANN (HNSW/COSINE) and sparse
               (SPARSE_INVERTED_INDEX/IP) search server-side via RRFRanker.
    Stage 2  — Cross-encoder reranking: fine-tuned cross-encoder rescores top
               candidates for higher precision.

Why Milvus-native hybrid search over client-side RRF?
    Server-side fusion avoids transferring full candidate lists over the network
    and keeps the ID spaces consistent (Milvus auto-IDs, not corpus indices).
"""

from __future__ import annotations

import logging
from typing import Any
from pymilvus import AnnSearchRequest, RRFRanker

logger = logging.getLogger(__name__)


class Retriever:
    """
    Hybrid retriever: Milvus dense + sparse search fused with RRF + cross-encoder reranker.

    Attributes:
        knowledge_base: KnowledgeBase instance (owns the Milvus collection).
        embedder:       Embedder for query vectorisation (dense + sparse).
        reranker:       CrossEncoder model (lazy-loaded on first rerank call).
        rrf_k:          RRF smoothing constant (default 60).
    """

    def __init__(
        self,
        knowledge_base: Any,
        embedder: Any,
        reranker_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        rrf_k: int = 60,
    ) -> None:
        self.knowledge_base = knowledge_base
        self.embedder = embedder
        self.reranker_model_name = reranker_model_name
        self.rrf_k = rrf_k
        self.reranker = None

    def dense_search(self, query_vec: list[float], top_k: int = 10) -> list[dict]:
        """
        Dense ANN search in Milvus.

        Returns:
            List of {"id", "text", "source", "score"} dicts sorted by descending score.
        """
        search_params = {"metric_type": "COSINE", "params": {"ef": 64}}
        results = self.knowledge_base.collection.search(
            data=[query_vec],
            anns_field="dense_vector",
            param=search_params,
            limit=top_k,
            output_fields=["text", "source"],
        )[0]
        return [
            {
                "id":     hit.id,
                "text":   hit.entity.get("text"),
                "source": hit.entity.get("source"),
                "score":  hit.score,
            }
            for hit in results
        ]

    def sparse_search(self, query: str, top_k: int = 10) -> list[dict]:
        """
        Sparse search using the Milvus SPARSE_INVERTED_INDEX (IP metric).

        Returns:
            List of {"id", "text", "source", "score"} dicts sorted by descending score.
        """
        query_sparse = self.embedder.embed_query_sparse(query)
        search_params = {"metric_type": "IP", "params": {}}
        results = self.knowledge_base.collection.search(
            data=[query_sparse],
            anns_field="sparse_vector",
            param=search_params,
            limit=top_k,
            output_fields=["text", "source"],
        )[0]
        return [
            {
                "id":     hit.id,
                "text":   hit.entity.get("text"),
                "source": hit.entity.get("source"),
                "score":  hit.score,
            }
            for hit in results
        ]

    def hybrid_search(self, query: str, top_k: int = 10) -> list[dict]:
        """
        Milvus native hybrid search: dense + sparse fused server-side with RRFRanker.

        Uses AnnSearchRequest for each vector field and collection.hybrid_search()
        to perform server-side RRF fusion, avoiding client-side score normalisation.

        Returns:
            List of {"id", "text", "source", "score"} dicts sorted by descending RRF score.
        """
        query_dense  = self.embedder.embed_query(query)
        query_sparse = self.embedder.embed_query_sparse(query)

        dense_req = AnnSearchRequest(
            data=[query_dense],
            anns_field="dense_vector",
            param={"metric_type": "COSINE", "params": {"ef": 64}},
            limit=top_k,
        )
        sparse_req = AnnSearchRequest(
            data=[query_sparse],
            anns_field="sparse_vector",
            param={"metric_type": "IP", "params": {}},
            limit=top_k,
        )
        results = self.knowledge_base.collection.hybrid_search(
            reqs=[dense_req, sparse_req],
            rerank=RRFRanker(k=self.rrf_k),
            limit=top_k,
            output_fields=["text", "source"],
        )[0]
        return [
            {
                "id":     hit.id,
                "text":   hit.entity.get("text"),
                "source": hit.entity.get("source"),
                "score":  hit.score,
            }
            for hit in results
        ]

    def rerank(self, query: str, docs: list[dict], top_n: int = 3, method: str = "Cross-Encoder") -> list[dict]:
        """
        Rerank candidate documents using a cross-encoder model.

        Args:
            query:  User query string.
            docs:   Candidate documents from hybrid_search().
            top_n:  Number of documents to return after reranking.
            method: Reranking method ("Cross-Encoder" only currently).

        Returns:
            Top-n documents re-sorted by cross-encoder relevance score,
            each with an added "rerank_score" field.
        """
        if method != "Cross-Encoder":
            raise NotImplementedError(f"Reranking method '{method}' not implemented.")
        if self.reranker is None:
            from sentence_transformers import CrossEncoder
            self.reranker = CrossEncoder(self.reranker_model_name)
        pairs = [(query, doc["text"]) for doc in docs]
        scores = self.reranker.predict(pairs)
        for doc, score in zip(docs, scores):
            doc["rerank_score"] = float(score)
        return sorted(docs, key=lambda d: d["rerank_score"], reverse=True)[:top_n]
