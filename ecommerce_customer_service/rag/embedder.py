"""
rag/embedder.py

Dense vector encoding using SentenceTransformers.

Why SentenceTransformers?
    - Pre-trained multilingual models (e.g. paraphrase-multilingual-MiniLM-L12-v2)
      handle Chinese and English in a single model.
    - Simple API: model.encode(texts) → numpy array.
    - Can be fine-tuned on domain-specific query-passage pairs using
      MultipleNegativesRankingLoss or CosineSimilarityLoss in < 1 hour on a
      single GPU.

Performance tips:
    - Use GPU if available (device="cuda").
    - Batch encoding (encode(texts, batch_size=64)) is much faster than one-by-one.
    - Normalise embeddings (normalize_embeddings=True) so that cosine similarity
      equals dot product, compatible with Milvus COSINE metric.
"""

from __future__ import annotations

import logging
from typing import Union

import numpy as np

logger = logging.getLogger(__name__)
from sentence_transformers import SentenceTransformer


class Embedder:
    """
    Wrapper around a SentenceTransformers model for batch text encoding.

    Attributes:
        model_name: HuggingFace model ID or local path.
        model:      Loaded SentenceTransformer instance (lazy-loaded on first use).
        device:     "cuda" or "cpu" depending on availability.
        batch_size: Number of texts encoded per forward pass.

    Example:
        embedder = Embedder("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        vectors  = embedder.embed(["退款政策是什么？", "How to track my order?"])
        query_vec = embedder.embed_query("快递在哪？")
    """
    model_name: str
    model: SentenceTransformer | None
    batch_size: int
    device: str | None

    def __init__(
        self,
        model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        batch_size: int = 64,
        device: str | None = None,
    ) -> None:
        """
        Initialise the embedder.  Model loading is deferred to the first call.

        Args:
            model_name: SentenceTransformers model identifier.
                        Recommended alternatives:
                        - "BAAI/bge-m3"                      (higher quality, larger)
                        - "moka-ai/m3e-base"                  (strong Chinese performance)
                        - "sentence-transformers/all-MiniLM-L6-v2" (English-only, fast)
            batch_size: Encoding batch size.  Tune to fill GPU VRAM without OOM.
            device:     Torch device string.  None = auto-detect (GPU > CPU).
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self.model = None  # Lazy-loaded on first encode request

    def _load_model(self) -> None:
        """
        Load the SentenceTransformer model into memory.

        Called automatically on first encode request.
        """
        if self.model is not None:
            return  # Model already loaded
        self.model = SentenceTransformer(self.model_name, device=self.device)
        logger.info(f"Loaded embedder model '{self.model_name}' on device '{self.model.device}'")
        # Optional: self.model.half() if using GPU and compatible with Milvus index metric

    def embed(self, texts: list[str]) -> list[list[float]]:
        """
        Batch-encode a list of texts into dense vectors.

        Args:
            texts: List of strings to encode.  May contain mixed Chinese/English.

        Returns:
            List of embedding vectors, one per input text.
            Each vector is a list of floats of length == model embedding dim
            (e.g. 384 for MiniLM-L12-v2).
        """
        if not texts:
            return []
        if self.model is None:
            self._load_model()
        try:
            assert self.model is not None  # For type checker
            vectors = self.model.encode(
                texts,
                batch_size=self.batch_size,
                normalize_embeddings=True,
                show_progress_bar=len(texts) > 100,
            )
            return vectors.tolist()
        except RuntimeError as e:
            if "CUDA out of memory" in str(e) and self.batch_size > 1:
                logger.warning(f"CUDA OOM with batch_size={self.batch_size}, retrying with smaller batch size")
                self.batch_size //= 2
                return self.embed(texts)  # Retry with smaller batch size
            else:
                logger.error(f"Error during embedding: {e}")
                raise

    def embed_query(self, query: str) -> list[float]:
        """
        Encode a single query string into a dense vector.

        Args:
            query: Query text (typically the rewritten user query).

        Note:
            Some asymmetric models (e.g. BGE) require a query prefix like
            "Represent this sentence: " for query-side encoding.  Check the
            model card and add the prefix here if needed.
        """
        return self.embed([query])[0]

    @property
    def embedding_dim(self) -> int:
        """
        Return the output dimensionality of the loaded model.

        Used by KnowledgeBase.create_collection() to set the vector field dimension.

        TODO:
            if self.model is None: self._load_model()
            return self.model.get_sentence_embedding_dimension()
        """
        if self.model is None:
            self._load_model()
        assert self.model is not None  # For type checker
        dim = self.model.get_sentence_embedding_dimension()
        assert dim is not None
        return dim