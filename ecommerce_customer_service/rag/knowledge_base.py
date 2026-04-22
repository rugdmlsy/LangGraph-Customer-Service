"""
rag/knowledge_base.py

Knowledge base management backed by Milvus vector database.

Ingestion pipeline:
    raw file → chunk → embed (dense + sparse) → insert into Milvus → build index

Collection schema:
    id            INT64 primary key (auto_id)
    text          VARCHAR chunk text
    source        VARCHAR origin file / URL
    chunk_id      INT64 position within source doc
    dense_vector  FLOAT_VECTOR  ANN searched with HNSW + COSINE
    sparse_vector SPARSE_FLOAT_VECTOR  searched with SPARSE_INVERTED_INDEX + IP

Both fields are indexed for Milvus native hybrid search (AnnSearchRequest + RRFRanker).
"""

from __future__ import annotations

import logging
from pathlib import Path
from pymilvus import connections
from pymilvus import CollectionSchema, FieldSchema, DataType, Collection
from pymilvus import utility
from rag.chunker import SlidingWindowChunker
from rag.embedder import Embedder
import fitz, time
from tqdm import tqdm

logger = logging.getLogger(__name__)


class KnowledgeBase:
    """
    High-level interface to the Milvus knowledge base.

    Typical usage (ingestion):
        kb = KnowledgeBase(embedder, chunker)
        kb.connect()
        kb.create_collection("ecommerce_faq", dim=1024)
        kb.load_from_file("data/raw/faq.txt")
        kb.build_index()

    Typical usage (query) — called by Retriever:
        kb.connect()
        # Retriever uses kb.collection directly via hybrid_search()
    """

    def __init__(
        self,
        embedder: Embedder,
        chunker: SlidingWindowChunker,
        host: str = "localhost",
        port: int = 19530,
        collection_name: str = "ecommerce_faq",
        batch_size: int = 1000,
    ) -> None:
        self.embedder = embedder
        self.chunker = chunker
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.collection: Collection | None = None
        self.batch_size = batch_size

    def connect(self) -> None:
        """Establish connection to the Milvus server (idempotent)."""
        connections.connect(alias="default", host=self.host, port=self.port)
        logger.info("Connected to Milvus at %s:%s", self.host, self.port)

    def create_collection(self, name: str, dim: int) -> None:
        """
        Create the collection with dense + sparse vector fields.

        If the collection already exists, loads it without recreating.
        """
        if utility.has_collection(name):
            self.collection = Collection(name)
            self.collection.load()
            logger.info("Loaded existing collection '%s'", name)
            return
        fields = [
            FieldSchema("id",            DataType.INT64,              is_primary=True, auto_id=True),
            FieldSchema("text",          DataType.VARCHAR,             max_length=65535),
            FieldSchema("source",        DataType.VARCHAR,             max_length=512),
            FieldSchema("chunk_id",      DataType.INT64),
            FieldSchema("dense_vector",  DataType.FLOAT_VECTOR,        dim=dim),
            FieldSchema("sparse_vector", DataType.SPARSE_FLOAT_VECTOR),
        ]
        schema = CollectionSchema(fields)
        self.collection = Collection(name, schema, consistency_level="Session")
        logger.info("Created collection '%s' with dim=%d", name, dim)

    def insert(
        self,
        texts: list[str],
        dense_vectors: list[list[float]],
        sparse_vectors: list[dict[int, float]],
        metadata: list[dict],
        flush: bool = False,
    ) -> list[int]:
        """
        Insert text chunks with their dense and sparse embeddings.

        Args:
            texts:          Raw chunk strings.
            dense_vectors:  Dense embeddings (from Embedder.embed or embed_hybrid).
            sparse_vectors: Sparse vectors as list of {token_id: weight} dicts
                            (from Embedder.embed_sparse or embed_hybrid).
            metadata:       Parallel list of {"source": str, "chunk_id": int} dicts.
            flush:          Flush after insert; set False when batching (flush once at end).

        Returns:
            List of auto-assigned Milvus primary key IDs.
        """
        if self.collection is None:
            self.collection = Collection(self.collection_name)

        all_ids = []
        total = len(texts)

        for i in range(0, total, self.batch_size):
            batch_texts   = texts[i : i + self.batch_size]
            batch_meta    = metadata[i : i + self.batch_size]
            batch_dense   = dense_vectors[i : i + self.batch_size]
            batch_sparse  = sparse_vectors[i : i + self.batch_size]

            data = [
                batch_texts,
                [m["source"]   for m in batch_meta],
                [m["chunk_id"] for m in batch_meta],
                batch_dense,
                batch_sparse,
            ]
            mr = self.collection.insert(data)
            all_ids.extend(mr.primary_keys)

        if flush:
            self.collection.flush()

        return all_ids

    def build_index(self) -> None:
        """
        Build HNSW index on dense_vector and SPARSE_INVERTED_INDEX on sparse_vector.
        Must be called after all data is inserted.
        """
        dense_index = {
            "metric_type": "COSINE",
            "index_type":  "HNSW",
            "params": {"M": 16, "efConstruction": 200},
        }
        sparse_index = {"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "IP"}
        assert self.collection is not None
        self.collection.create_index("dense_vector",  dense_index)
        self.collection.create_index("sparse_vector", sparse_index)
        self.collection.load()
        logger.info("Index built and collection loaded into memory.")

    def load_from_file(self, file_path: str | Path) -> int:
        """
        End-to-end ingestion: read file → chunk → embed (dense + sparse) → insert.

        Supports .txt and .pdf formats.

        Returns:
            Total number of chunks inserted.
        """
        file = Path(file_path)
        if not file.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        file_size_mb = file.stat().st_size / (1024 * 1024)
        print(f"  处理文件: {file.name} ({file_size_mb:.1f} MB)")
        start = time.perf_counter()

        if file.suffix == ".pdf":
            with fitz.open(file) as doc:
                full_text = "\n".join(str(page.get_text()) for page in doc)
        elif file.suffix == ".txt":
            with open(file, "r", encoding="utf-8", errors="replace") as f:
                full_text = f.read()
        else:
            raise ValueError(f"Unsupported file format: {file.suffix}")

        chunks_with_meta = self.chunker.chunk_with_metadata(full_text, str(file))
        texts = [c["text"]     for c in chunks_with_meta]
        meta  = [{"source": c["source"], "chunk_id": c["chunk_id"]} for c in chunks_with_meta]

        total_batches = (len(texts) + self.batch_size - 1) // self.batch_size
        pbar = tqdm(
            range(0, len(texts), self.batch_size),
            total=total_batches,
            desc=f"  Embedding {file.name}",
            unit="batch",
            ncols=80,
        )
        for i in pbar:
            batch_texts = texts[i : i + self.batch_size]
            batch_meta  = meta[i : i + self.batch_size]
            embeddings  = self.embedder.embed_hybrid(batch_texts)
            self.insert(
                batch_texts,
                embeddings["dense"],
                embeddings["sparse"],
                batch_meta,
                flush=False,
            )
            pbar.set_postfix({"chunks": f"{min(i + self.batch_size, len(texts))}/{len(texts)}"})

        assert self.collection is not None
        self.collection.flush()
        elapsed = time.perf_counter() - start
        logger.info(
            "Ingested %d chunks from '%s' in %.1f s (%.0f chunks/s)",
            len(chunks_with_meta), file.name, elapsed, len(chunks_with_meta) / max(elapsed, 1),
        )
        return len(chunks_with_meta)

    def drop_collection(self) -> None:
        """Drop the collection from Milvus. Data is permanently lost."""
        utility.drop_collection(self.collection_name)
        self.collection = None
        logger.warning("Dropped collection '%s'", self.collection_name)

    def has_collection(self, name: str) -> bool:
        return bool(utility.has_collection(name))
