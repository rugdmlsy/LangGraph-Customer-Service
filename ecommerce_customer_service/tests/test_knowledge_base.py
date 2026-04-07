from config import settings
from rag import SlidingWindowChunker, Embedder, KnowledgeBase
embedder = Embedder(settings.EMBEDDING_MODEL_NAME)
chunker  = SlidingWindowChunker()
kb = KnowledgeBase(embedder, chunker)
kb.connect()
kb.create_collection("test_col", dim=384)
kb.load_from_file("data/故障排除.txt")
kb.build_index()