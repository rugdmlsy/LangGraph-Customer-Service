"""
rag/chunker.py

Text chunking utilities for the knowledge base ingestion pipeline.

Chunking strategy — sliding window:
    Split long documents into overlapping chunks of fixed character/token size.
    Overlap ensures that information near chunk boundaries is not lost; the
    same sentence appears in two consecutive chunks so at least one chunk will
    be retrieved.

    chunk_size = 512 characters  (approx 256 Chinese tokens)
    overlap    = 100 characters

Why character-based rather than token-based?
    Tokenisation depends on the model; using characters is model-agnostic and
    avoids loading the tokeniser at ingestion time.  For high-accuracy slicing
    on CJK text, switch to a tokeniser-aware splitter (e.g. LangChain's
    RecursiveCharacterTextSplitter with `separators=["。", "！", "？", "\\n"]`).
"""

from __future__ import annotations

import logging
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader

logger = logging.getLogger(__name__)


class SlidingWindowChunker:
    """
    Splits raw text into overlapping character-level windows.

    Attributes:
        chunk_size: Maximum number of characters per chunk.
        overlap:    Number of characters shared between consecutive chunks.

    Example:
        chunker = SlidingWindowChunker(chunk_size=512, overlap=100)
        chunks  = chunker.chunk("很长的商品描述文本 ...")
        # → ["第1段...", "第2段...（前100字与第1段相同）", ...]
    """
    chunk_size: int
    overlap: int
    mode: str 

    def __init__(self, chunk_size: int = 512, overlap: int = 100) -> None:
        """
        Initialise chunker with window parameters.

        Args:
            chunk_size: Number of characters per chunk.  Must be > overlap.
            overlap:    Overlap in characters between consecutive chunks.
                        Controls context continuity at chunk boundaries.
        """
        if chunk_size <= overlap:
            raise ValueError("chunk_size must be greater than overlap")
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.mode = "char"  # For now, only character-based chunking is implemented

    def chunk(self, text: str) -> list[str]:
        """
        Split `text` into overlapping windows and return the list of chunks.

        Args:
            text: Raw document text (may contain newlines, HTML tags, etc.).
                  Callers should strip HTML / markdown before passing here.

        Returns:
            List of chunk strings.  Empty list if text is empty or whitespace.
            Each chunk has at most `chunk_size` characters.

        Optimisation for CJK text:
            Before the loop, split on sentence-ending punctuation
            (。！？\n) to avoid cutting mid-sentence.  Then greedily
            pack sentences into windows up to chunk_size characters.
            This "sentence-aware" variant improves retrieval quality
            because each chunk is semantically complete.

        Performance note:
            This function runs at ingestion time (offline), so optimising
            for throughput (vectorised string ops) matters more than
            latency.  For very large corpora, consider multiprocessing.
        """
        text_splitter = RecursiveCharacterTextSplitter(
            separators=[
                "\n\n",
                "\n",
                " ",
                ".",
                ",",
                "\u200b",  # Zero-width space
                "\uff0c",  # Fullwidth comma
                "\u3001",  # Ideographic comma
                "\uff0e",  # Fullwidth full stop
                "\u3002",  # Ideographic stop
                "",
            ],
            chunk_size=self.chunk_size, chunk_overlap=self.overlap,
        )
        chunks = text_splitter.split_text(text)
        return chunks

    def chunk_with_metadata(self, text: str, source: str) -> list[dict]:
        """
        Chunk `text` and attach source metadata to each chunk.

        Args:
            text:   Document text to chunk.
            source: Identifier for the source document (filename, URL, doc_id).

        Returns:
            List of dicts: [{"text": str, "chunk_id": int, "source": str}, ...]

        This method is a convenience wrapper used by KnowledgeBase.load_from_file().
        """
        chunks = self.chunk(text)
        return [{"text": c, "chunk_id": i, "source": source} 
                for i, c in enumerate(chunks)]
