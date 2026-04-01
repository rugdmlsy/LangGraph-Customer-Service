"""
rag package

Provides the full Retrieval-Augmented Generation pipeline:

    Chunker       – sliding-window text segmentation
    Embedder      – dense vector encoding with SentenceTransformers
    KnowledgeBase – Milvus collection management (insert, index, load)
    Retriever     – hybrid (dense + sparse) search with cross-encoder reranking
"""

from rag.chunker import SlidingWindowChunker
from rag.embedder import Embedder
from rag.knowledge_base import KnowledgeBase
from rag.retriever import Retriever

__all__ = ["SlidingWindowChunker", "Embedder", "KnowledgeBase", "Retriever"]
