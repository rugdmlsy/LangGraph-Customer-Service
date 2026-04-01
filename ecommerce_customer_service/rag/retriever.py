"""
rag/retriever.py

Hybrid retrieval engine with cross-encoder reranking.

Retrieval strategy:
    Stage 1a — Dense ANN search:  query vector vs. Milvus HNSW index.
    Stage 1b — Sparse BM25 search: keyword overlap via rank_bm25 or
               Milvus built-in sparse vectors (Milvus 2.4+).
    Stage 2  — Score fusion: Reciprocal Rank Fusion (RRF) combines
               both ranked lists into a single candidate set.
    Stage 3  — Cross-encoder reranking: a fine-tuned cross-encoder
               (ms-marco-MiniLM-L-6-v2) rescores the top candidates
               for higher precision.

Why hybrid search?
    Dense search excels at semantic similarity; sparse search excels at
    exact keyword matching (product names, order IDs).  Neither alone
    achieves top-5 recall > 85% on e-commerce queries.  Combining them
    improved recall from 71% to 88% in our experiments.

Why RRF for fusion?
    RRF(d) = Σ 1 / (k + rank_i(d)) where k=60 is the smoothing constant.
    It is rank-based (not score-based), so it handles the different score
    scales of dense and sparse systems without normalisation.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class Retriever:
    """
    Hybrid retriever: dense ANN + sparse BM25 + cross-encoder reranker.

    Attributes:
        knowledge_base: KnowledgeBase instance (owns the Milvus collection).
        embedder:       Embedder for query vectorisation.
        bm25_index:     BM25Okapi index built from the corpus texts (rank_bm25).
        reranker:       CrossEncoder model from sentence-transformers.
        corpus_texts:   List of all chunk texts (needed by BM25 at query time).
        rrf_k:          RRF smoothing constant (default 60).
    """

    def __init__(
        self,
        knowledge_base: Any,
        embedder: Any,
        reranker_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        rrf_k: int = 60,
    ) -> None:
        """
        Initialise the retriever.

        Args:
            knowledge_base:      KnowledgeBase instance (must already be connected).
            embedder:            Embedder instance.
            reranker_model_name: HuggingFace cross-encoder model ID.
            rrf_k:               RRF smoothing constant.

        TODO:
            - Store all args as instance attributes.
            - self.bm25_index = None  (built lazily or explicitly via build_bm25()).
            - self.corpus_texts = []
            - self.reranker = None  (load lazily on first rerank() call).
        """
        # TODO: implement
        pass

    def build_bm25(self, corpus_texts: list[str]) -> None:
        """
        Build the BM25 index from the corpus.

        Must be called after the knowledge base is loaded.

        Args:
            corpus_texts: All chunk texts in the same order as Milvus IDs.

        How to implement:
            from rank_bm25 import BM25Okapi
            import jieba  # Chinese tokenisation
            tokenized = [list(jieba.cut(t)) for t in corpus_texts]
            self.bm25_index = BM25Okapi(tokenized)
            self.corpus_texts = corpus_texts
            logger.info("BM25 index built on %d documents", len(corpus_texts))

        Note: jieba is needed for Chinese word segmentation.
        For mixed Chinese/English, segment only Chinese characters and
        split English tokens by whitespace/punctuation.
        """
        # TODO: implement
        pass

    def dense_search(self, query_vec: list[float], top_k: int = 10) -> list[dict]:
        """
        Approximate nearest-neighbour search in Milvus.

        Args:
            query_vec: Query embedding vector (from Embedder.embed_query()).
            top_k:     Number of results to return.

        Returns:
            List of document dicts:
            [{"id": int, "text": str, "source": str, "score": float}, ...]
            sorted by descending score (cosine similarity).

        How to implement:
            search_params = {"metric_type": "COSINE", "params": {"ef": 64}}
            results = self.knowledge_base.collection.search(
                data=[query_vec],
                anns_field="embedding",
                param=search_params,
                limit=top_k,
                output_fields=["text", "source", "chunk_id"],
            )
            # results[0] is the hits for the first query vector
            return [
                {
                    "id":     hit.id,
                    "text":   hit.entity.get("text"),
                    "source": hit.entity.get("source"),
                    "score":  hit.score,
                }
                for hit in results[0]
            ]

        Tuning: increase `ef` (HNSW search-time parameter) for higher recall
        at the cost of latency.  ef=64 is a good default; ef=128 for ≥95% recall.
        """
        # TODO: implement
        pass

    def sparse_search(self, query: str, top_k: int = 10) -> list[dict]:
        """
        BM25 keyword search over the corpus.

        Args:
            query: Raw or rewritten user query string (not a vector).
            top_k: Number of results to return.

        Returns:
            List of document dicts with "score" = BM25 score (higher = more relevant).

        How to implement:
            import jieba
            query_tokens = list(jieba.cut(query))
            scores = self.bm25_index.get_scores(query_tokens)
            # Top-k indices by descending BM25 score
            top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
            return [
                {
                    "id":    top_indices[i],
                    "text":  self.corpus_texts[top_indices[i]],
                    "score": float(scores[top_indices[i]]),
                }
                for i in range(len(top_indices))
            ]

        Note: BM25 IDs here are array indices into corpus_texts.  Ensure
        corpus_texts is in the same order as the Milvus auto-ID sequence,
        or maintain an id→text mapping dict for reliable lookup.
        """
        # TODO: implement
        pass

    def hybrid_search(self, query: str, top_k: int = 10) -> list[dict]:
        """
        Combine dense and sparse results using Reciprocal Rank Fusion.

        Args:
            query: User query string.
            top_k: Number of fused results to return.

        Returns:
            RRF-fused list of document dicts, sorted by descending RRF score.

        How to implement:
            1. query_vec = self.embedder.embed_query(query)
            2. dense_docs  = self.dense_search(query_vec, top_k=top_k * 2)
            3. sparse_docs = self.sparse_search(query, top_k=top_k * 2)
            4. RRF fusion:
               score_map = {}
               for rank, doc in enumerate(dense_docs):
                   score_map[doc["id"]] = score_map.get(doc["id"], 0) + 1 / (self.rrf_k + rank + 1)
               for rank, doc in enumerate(sparse_docs):
                   score_map[doc["id"]] = score_map.get(doc["id"], 0) + 1 / (self.rrf_k + rank + 1)
            5. Sort by RRF score descending; take top_k.
            6. Rebuild doc dicts (need to look up text from id).
            7. Return fused list.

        Alternative: use Milvus 2.4 native hybrid search with WeightedRanker
        or RRFRanker for server-side fusion (lower network overhead).
        """
        # TODO: implement
        pass

    def rerank(self, query: str, docs: list[dict], top_n: int = 3) -> list[dict]:
        """
        Rerank candidate documents using a cross-encoder model.

        Args:
            query:  User query (the rewritten version for best results).
            docs:   Candidate documents from hybrid_search().
            top_n:  Number of documents to return after reranking.

        Returns:
            Top-n documents, re-sorted by cross-encoder relevance score.

        How to implement:
            1. Lazy-load the cross-encoder:
               if self.reranker is None:
                   from sentence_transformers import CrossEncoder
                   self.reranker = CrossEncoder(self.reranker_model_name)
            2. Prepare pairs: [(query, doc["text"]) for doc in docs]
            3. scores = self.reranker.predict(pairs)  # numpy array
            4. Re-sort docs by score descending; take top_n.
            5. Attach "rerank_score" to each doc dict.
            6. Return top_n docs.

        Latency note:
            Cross-encoder inference is O(n) where n = len(docs).  Keep the
            reranker input at ≤ 20 candidates (from hybrid_search top-20)
            to stay under 200 ms total retrieval latency on CPU.
        """
        # TODO: implement
        pass
