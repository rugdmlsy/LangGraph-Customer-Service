"""
rag/knowledge_base.py

Knowledge base management backed by Milvus vector database.

This module owns the full ingestion pipeline:
    raw file → chunk → embed → insert into Milvus → build index → ready to query

Milvus concepts used here:
    Collection: analogous to a SQL table; stores vectors + scalar metadata fields.
    Index:      HNSW graph built on the vector field for ANN search.
                IVF_FLAT is an alternative with lower memory footprint.
    Partition:  (optional) logical sub-collections within one collection;
                use to separate FAQ, manual, and chat history data sources.

Connection management:
    Use pymilvus.connections.connect() once at startup (via connect()).
    PyMilvus maintains a connection pool internally; no manual pooling needed.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
from pymilvus import connections
from pymilvus import CollectionSchema, FieldSchema, DataType, Collection
from pymilvus import utility
from rag.chunker import SlidingWindowChunker
from rag.embedder import Embedder
import fitz, time

logger = logging.getLogger(__name__)


class KnowledgeBase:
    """
    High-level interface to the Milvus knowledge base.

    Attributes:
        host:           Milvus server host.
        port:           Milvus server port.
        collection_name: Active Milvus collection name.
        embedder:        Embedder instance for vector generation.
        chunker:         SlidingWindowChunker instance for document splitting.
        collection:      pymilvus.Collection object (set after connect()).

    Typical usage (ingestion):
        kb = KnowledgeBase(embedder, chunker)
        kb.connect()
        kb.create_collection("ecommerce_faq", dim=384)
        kb.load_from_file("data/raw/faq.txt")
        kb.build_index()

    Typical usage (query) — called by Retriever:
        kb.connect()
        results = kb.collection.search(...)
    """

    def __init__(
        self,
        embedder: Embedder,
        chunker: SlidingWindowChunker,
        host: str = "localhost",
        port: int = 19530,
        collection_name: str = "ecommerce_faq",
    ) -> None:
        """
        Initialise the knowledge base manager.

        Args:
            embedder:        Embedder instance.  Needed by load_from_file().
            chunker:         SlidingWindowChunker instance.
            host:            Milvus host (override with settings.MILVUS_HOST).
            port:            Milvus port.
            collection_name: Name of the Milvus collection to operate on.
        """
        self.embedder = embedder
        self.chunker = chunker
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.collection = None  # Set after connect()

    def connect(self) -> None:
        """
        Establish connection to the Milvus server.

        Call this once at application startup (e.g. in FastAPI lifespan event).
        Subsequent calls are safe (idempotent in pymilvus >= 2.3).
        """
        connections.connect(alias="default", host=self.host, port=self.port)
        logger.info("Connected to Milvus at %s:%s", self.host, self.port)

    def create_collection(self, name: str, dim: int) -> None:
        """
        Create a Milvus collection with the standard FAQ schema.

        Schema fields:
            id        (INT64,  primary key, auto_id=True)
            text      (VARCHAR, max_length=4096)  — raw chunk text
            source    (VARCHAR, max_length=512)   — origin file / URL
            chunk_id  (INT64)                     — position within source doc
            embedding (FLOAT_VECTOR, dim=dim)     — dense vector

        Args:
            name: Collection name (stored as self.collection_name).
            dim:  Embedding dimensionality (must match embedder output).
        """
        if utility.has_collection(name):
            self.collection = Collection(name)
            self.collection.load()
            logger.info("Loaded existing collection '%s'", name)
            return
        fields = [
            FieldSchema("id",        DataType.INT64,         is_primary=True, auto_id=True),
            FieldSchema("text",      DataType.VARCHAR,        max_length=4096),
            FieldSchema("source",    DataType.VARCHAR,        max_length=512),
            FieldSchema("chunk_id",  DataType.INT64),
            FieldSchema("embedding", DataType.FLOAT_VECTOR,   dim=dim),
        ]
        schema = CollectionSchema(fields)
        self.collection = Collection(name=name, schema=schema)
        logger.info("Created collection '%s' with dim=%d", name, dim)

    def insert(
        self,
        texts: list[str],
        embeddings: list[list[float]],
        metadata: list[dict],
    ) -> list[int]:
        """
        Insert text chunks and their embeddings into the collection.

        Args:
            texts:      List of raw chunk strings.
            embeddings: Parallel list of dense vectors (from Embedder.embed()).
            metadata:   Parallel list of dicts with keys "source" and "chunk_id".

        Returns:
            List of auto-assigned Milvus primary key IDs.
            
        Batch insert tip: for large corpora, call insert() in batches of
        ~1 000 chunks to avoid gRPC message size limits and OOM errors.
        """
        data = [
            texts,
            [m["source"]   for m in metadata],
            [m["chunk_id"] for m in metadata],
            embeddings,
        ]
        if self.collection is None:
            self.collection = Collection(self.collection_name)  # lazy load if not connected
        mr = self.collection.insert(data)
        self.collection.flush()   # ensure data is persisted to segment
        return mr.primary_keys

    def build_index(self) -> None:
        """
        Build the ANN index on the embedding field.

        Must be called after all data is inserted and before collection.search().

        HNSW tuning:
            M (graph edges per node): higher → better recall, more memory.
            efConstruction: higher → better index quality, slower build.
            Recommended: M=16, efConstruction=200 for production;
            M=8, efConstruction=64 for fast prototyping.
        """
        index_params = {
            "metric_type": "COSINE",
            "index_type":  "HNSW",
            "params": {"M": 16, "efConstruction": 200},
        }
        assert self.collection is not None
        self.collection.create_index(field_name="embedding", index_params=index_params)
        self.collection.load()
        logger.info("Index built and collection loaded into memory.")

    def load_from_file(self, file_path: str | Path) -> int:
        """
        End-to-end ingestion pipeline: read → chunk → embed → insert.

        Supports file formats:
            .txt   — plain text, one document per file
            .pdf   — PDF documents (requires pdfplumber dependency)
            to be extended:
                .jsonl — one JSON object per line with "text" and "source" keys
                .json  — list of {"text": str, "source": str} objects

        Args:
            file_path: Path to the input file.

        Returns:
            Total number of chunks inserted.
        """
        file = Path(file_path)
        if not file.exists():
            logger.error("File not found: %s", file_path)
            raise FileNotFoundError(f"File not found: {file_path}")
        start = time.perf_counter()
        if file.suffix == ".pdf":
            with fitz.open(file) as doc:
                rawtexts = []
                for page in doc:
                    rawtexts.append(page.get_text())
                full_text = "\n".join(rawtexts)
        elif file.suffix == ".txt":
            with open(file, "r") as f:
                full_text = f.read()
        else:
            logger.error("Unsupported file format: %s", file.suffix)
            raise ValueError(f"Unsupported file format: {file.suffix}")
        chunks_with_meta = self.chunker.chunk_with_metadata(full_text, str(file))
        texts = [c["text"] for c in chunks_with_meta]
        meta = [{"source": c["source"], "chunk_id": c["chunk_id"]} for c in chunks_with_meta]
        embeddings = self.embedder.embed(texts)
        self.insert(texts, embeddings, meta)
        elapsed = time.perf_counter() - start
        logger.info("Ingested %d chunks from file '%s', elapsed: %.2f seconds",
                    len(chunks_with_meta), file_path, elapsed)
        return len(chunks_with_meta)    

    def drop_collection(self) -> None:
        """
        Drop the collection from Milvus.  Use with caution — data is lost.
        """
        utility.drop_collection(self.collection_name)
        self.collection = None
        logger.warning("Dropped collection '%s'", self.collection_name)
