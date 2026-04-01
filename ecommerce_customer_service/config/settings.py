"""
config/settings.py

Central configuration management for the entire system.

All runtime parameters — model paths, database URIs, queue config, tuning knobs —
live here so they can be overridden via environment variables or a .env file
without touching application code.

Implementation approach:
    Use pydantic-settings BaseSettings, which automatically reads from environment
    variables (uppercased field names) and from a .env file when
    `model_config = SettingsConfigDict(env_file=".env")` is declared.
    Alternatively, use Python dataclasses + python-dotenv for a lighter dependency.

Usage:
    from config import settings
    print(settings.MILVUS_HOST)
"""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application-wide settings loaded from environment variables / .env file.

    Group fields into logical sections with comments so contributors can
    quickly understand which subsystem each knob controls.

    How to implement:
        1. Declare each field with a type annotation and a default value.
        2. Use Field(description="...") to self-document each knob.
        3. Sensitive values (API keys, passwords) should have no defaults so
           the application fails fast if they are missing in production.
        4. Add validators (@field_validator) for fields that need range checks
           (e.g. CHUNK_SIZE must be > 0).
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ------------------------------------------------------------------ #
    # LLM configuration                                                    #
    # ------------------------------------------------------------------ #

    LLM_MODEL_NAME: str = Field(
        default="Qwen/Qwen2.5-7B-Instruct",
        description=(
            "HuggingFace model ID or local path to the LLM. "
            "Supports Qwen2.5 and Llama-3 family models. "
            "For local quantised models use the absolute GGUF path."
        ),
    )

    LLM_API_BASE: str = Field(
        default="http://localhost:8000/v1",
        description=(
            "Base URL of the OpenAI-compatible serving endpoint (e.g. vLLM, Ollama). "
            "Leave empty to use the HuggingFace transformers pipeline directly."
        ),
    )

    LLM_API_KEY: str = Field(
        default="EMPTY",
        description="API key for the LLM endpoint. Use 'EMPTY' for local servers.",
    )

    LLM_MAX_TOKENS: int = Field(default=1024, description="Maximum tokens to generate per call.")

    LLM_TEMPERATURE: float = Field(default=0.1, description="Sampling temperature (0 = greedy).")

    # ------------------------------------------------------------------ #
    # Embedding model                                                       #
    # ------------------------------------------------------------------ #

    EMBEDDING_MODEL_NAME: str = Field(
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        description=(
            "SentenceTransformers model for dense embeddings. "
            "Multilingual variant recommended for Chinese + English mixed queries. "
            "Alternative: BAAI/bge-m3 for higher quality at larger compute cost."
        ),
    )

    EMBEDDING_BATCH_SIZE: int = Field(
        default=64,
        description="Number of texts encoded in one forward pass. Tune based on GPU VRAM.",
    )

    EMBEDDING_DIM: int = Field(
        default=384,
        description=(
            "Output dimensionality of the embedding model. "
            "Must match the Milvus collection schema. "
            "paraphrase-multilingual-MiniLM-L12-v2 → 384; bge-m3 → 1024."
        ),
    )

    # ------------------------------------------------------------------ #
    # Chunking                                                             #
    # ------------------------------------------------------------------ #

    CHUNK_SIZE: int = Field(
        default=512,
        description="Maximum number of characters (or tokens) per text chunk.",
    )

    CHUNK_OVERLAP: int = Field(
        default=100,
        description="Overlap between consecutive sliding-window chunks to preserve context.",
    )

    # ------------------------------------------------------------------ #
    # Retrieval                                                            #
    # ------------------------------------------------------------------ #

    RETRIEVAL_TOP_K: int = Field(
        default=10,
        description="Number of candidates returned by the first-stage retriever before reranking.",
    )

    RERANK_TOP_N: int = Field(
        default=3,
        description="Final number of documents kept after cross-encoder reranking.",
    )

    RERANKER_MODEL_NAME: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        description="Cross-encoder model used for reranking retrieved documents.",
    )

    HYBRID_SEARCH_ALPHA: float = Field(
        default=0.5,
        description=(
            "Weight of dense scores in the RRF fusion formula. "
            "0 = pure sparse (BM25), 1 = pure dense. 0.5 balances both."
        ),
    )

    # ------------------------------------------------------------------ #
    # Milvus vector database                                               #
    # ------------------------------------------------------------------ #

    MILVUS_HOST: str = Field(default="localhost", description="Milvus server hostname.")

    MILVUS_PORT: int = Field(default=19530, description="Milvus gRPC port.")

    MILVUS_COLLECTION_NAME: str = Field(
        default="ecommerce_faq",
        description="Default collection name for FAQ knowledge base.",
    )

    MILVUS_INDEX_TYPE: str = Field(
        default="HNSW",
        description=(
            "Vector index type. HNSW for high-recall production use; "
            "IVF_FLAT for smaller datasets or limited RAM."
        ),
    )

    MILVUS_METRIC_TYPE: str = Field(
        default="COSINE",
        description="Distance metric. COSINE for normalised embeddings; IP or L2 otherwise.",
    )

    # ------------------------------------------------------------------ #
    # Redis                                                                #
    # ------------------------------------------------------------------ #

    REDIS_HOST: str = Field(default="localhost", description="Redis server hostname.")

    REDIS_PORT: int = Field(default=6379, description="Redis server port.")

    REDIS_PASSWORD: str = Field(default="", description="Redis AUTH password. Empty = no auth.")

    REDIS_DB: int = Field(default=0, description="Redis logical database index.")

    REDIS_QUEUE_KEY: str = Field(
        default="customer_service:queue",
        description="Redis list key used as the main request queue (LPUSH / BRPOP pattern).",
    )

    REDIS_LONG_TERM_MEMORY_TTL: int = Field(
        default=86400 * 30,
        description="TTL in seconds for long-term user memory keys (default: 30 days).",
    )

    # ------------------------------------------------------------------ #
    # Kafka (optional, alternative to Redis queue)                         #
    # ------------------------------------------------------------------ #

    KAFKA_BOOTSTRAP_SERVERS: str = Field(
        default="localhost:9092",
        description="Comma-separated list of Kafka broker addresses.",
    )

    KAFKA_TOPIC: str = Field(
        default="customer_service_requests",
        description="Kafka topic for incoming customer service requests.",
    )

    KAFKA_GROUP_ID: str = Field(
        default="agent_workers",
        description="Consumer group ID for Kafka workers.",
    )

    USE_KAFKA: bool = Field(
        default=False,
        description="Switch to Kafka queue instead of Redis. Requires running Kafka broker.",
    )

    # ------------------------------------------------------------------ #
    # Thread pool / concurrency                                            #
    # ------------------------------------------------------------------ #

    THREAD_POOL_MAX_WORKERS: int = Field(
        default=10,
        description=(
            "Maximum threads in ThreadPoolExecutor for concurrent message processing. "
            "Rule of thumb: 2–4× number of CPU cores for I/O-bound LLM API calls."
        ),
    )

    # ------------------------------------------------------------------ #
    # API server                                                           #
    # ------------------------------------------------------------------ #

    API_HOST: str = Field(default="0.0.0.0", description="FastAPI bind host.")

    API_PORT: int = Field(default=8080, description="FastAPI bind port.")

    API_WORKERS: int = Field(
        default=1,
        description=(
            "Number of Uvicorn worker processes. "
            "Keep at 1 when the in-process queue worker thread is used."
        ),
    )

    # ------------------------------------------------------------------ #
    # Logging                                                              #
    # ------------------------------------------------------------------ #

    LOG_LEVEL: str = Field(default="INFO", description="Python logging level.")

    LOG_FORMAT: str = Field(
        default="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        description="Log format string.",
    )
