"""
rag/embedder.py

Dense + sparse vector encoding using SentenceTransformers or BGE-M3.

BGE-M3 mode (model_name contains "bge-m3"):
    Uses pymilvus BGEM3EmbeddingFunction which produces both dense and sparse
    (SPLADE-style) vectors in a single forward pass.

Other models:
    Dense via SentenceTransformers; sparse via jieba TF (token hash → frequency).
"""

from __future__ import annotations

import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)
from sentence_transformers import SentenceTransformer
from pymilvus.model.hybrid import BGEM3EmbeddingFunction


class Embedder:
    """
    Wrapper for batch text encoding with optional hybrid (dense + sparse) output.

    Attributes:
        model_name: HuggingFace model ID or local path.
        model:      SentenceTransformer instance (non-BGE-M3 models).
        bgem3_fn:   BGEM3EmbeddingFunction instance (BGE-M3 models).
        device:     "cuda" or "cpu".
        batch_size: Texts encoded per forward pass.
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
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = self._resolve_device(device)
        self.model = None
        self.bgem3_fn = None

    def _resolve_device(self, device: str | None) -> str:
        if device is None or device == "auto":
            resolved = "cuda" if torch.cuda.is_available() else "cpu"
        elif device == "cuda" and not torch.cuda.is_available():
            logger.warning(
                "EMBEDDING_DEVICE='cuda' but CUDA is not available; falling back to 'cpu'."
            )
            resolved = "cpu"
        else:
            resolved = device
        if resolved == "cuda" and self.batch_size < 512:
            self.batch_size = 512
            logger.info("GPU detected — auto-set EMBEDDING_BATCH_SIZE=512")
        logger.info("Embedding device: %s (batch_size=%d)", resolved, self.batch_size)
        return resolved

    def _load_model(self) -> None:
        if self.model is not None or self.bgem3_fn is not None:
            return
        if "bge-m3" in self.model_name.lower():
            self.bgem3_fn = BGEM3EmbeddingFunction(
                model_name=self.model_name,
                device=self.device,
                use_fp16=(self.device == "cuda"),
                batch_size=self.batch_size,
            )
            logger.info("Loaded BGE-M3 embedder '%s' on device '%s'", self.model_name, self.device)
        else:
            self.model = SentenceTransformer(self.model_name, device=self.device)
            logger.info("Loaded embedder '%s' on device '%s'", self.model_name, self.device)

    def embed(self, texts: list[str], show_progress: bool | None = None) -> list[list[float]]:
        """Encode texts into dense vectors."""
        if not texts:
            return []
        self._load_model()
        if self.bgem3_fn is not None:
            result = self.bgem3_fn(texts)
            dense = np.array(result["dense"], dtype="float32")
            return dense.tolist()
        if show_progress is None:
            show_progress = len(texts) > 100
        try:
            assert self.model is not None
            vectors = self.model.encode(
                texts,
                batch_size=self.batch_size,
                normalize_embeddings=True,
                show_progress_bar=show_progress,
            )
            return vectors.tolist()
        except RuntimeError as e:
            if "CUDA out of memory" in str(e) and self.batch_size > 1:
                logger.warning("CUDA OOM with batch_size=%d, retrying with smaller batch", self.batch_size)
                self.batch_size //= 2
                return self.embed(texts, show_progress=show_progress)
            raise

    def embed_sparse(self, texts: list[str]) -> list[dict[int, float]]:
        """Encode texts into sparse vectors as list of {token_id: weight} dicts."""
        if not texts:
            return []
        self._load_model()
        if self.bgem3_fn is not None:
            sparse_mat = self.bgem3_fn(texts)["sparse"]
            return [self._csr_row_to_dict(sparse_mat[i]) for i in range(sparse_mat.shape[0])]
        return self._jieba_sparse(texts)

    def embed_hybrid(self, texts: list[str]) -> dict[str, list]:
        """
        Encode texts into both dense and sparse vectors in one pass.
        Returns {"dense": list[list[float]], "sparse": list[dict[int, float]]}.
        Preferred over calling embed() + embed_sparse() separately.
        """
        if not texts:
            return {"dense": [], "sparse": []}
        self._load_model()
        if self.bgem3_fn is not None:
            result = self.bgem3_fn(texts)
            dense = np.array(result["dense"], dtype="float32")
            sparse_mat = result["sparse"]
            return {
                "dense": dense.tolist(),
                "sparse": [self._csr_row_to_dict(sparse_mat[i]) for i in range(sparse_mat.shape[0])],
            }
        dense = self.embed(texts)
        sparse = self._jieba_sparse(texts)
        return {"dense": dense, "sparse": sparse}

    def embed_query(self, query: str) -> list[float]:
        """Encode a single query string into a dense vector."""
        return self.embed([query])[0]

    def embed_query_sparse(self, query: str) -> dict[int, float]:
        """Encode a single query string into a sparse vector."""
        return self.embed_sparse([query])[0]

    def _csr_row_to_dict(self, row) -> dict[int, float]:
        """Convert a scipy csr_matrix row to {col_index: value} dict."""
        coo = row.tocoo()
        return {int(j): float(v) for j, v in zip(coo.col, coo.data)}

    def _jieba_sparse(self, texts: list[str]) -> list[dict[int, float]]:
        """Fallback sparse representation via jieba TF with token hash IDs."""
        import jieba
        import hashlib
        result = []
        for text in texts:
            tokens = list(jieba.cut(text))
            if not tokens:
                result.append({0: 1.0})
                continue
            tf: dict[int, float] = {}
            for t in tokens:
                key = int(hashlib.md5(t.encode()).hexdigest()[:8], 16) % (1 << 20)
                tf[key] = tf.get(key, 0.0) + 1.0
            total = float(len(tokens))
            result.append({k: v / total for k, v in tf.items()})
        return result

    @property
    def embedding_dim(self) -> int:
        """Return the dense embedding dimensionality of the loaded model."""
        self._load_model()
        if self.bgem3_fn is not None:
            return self.bgem3_fn.dim["dense"]
        assert self.model is not None
        dim = self.model.get_embedding_dimension()
        assert dim is not None
        return dim
