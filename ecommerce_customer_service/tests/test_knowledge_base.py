from config import settings
from rag import SlidingWindowChunker, Embedder, KnowledgeBase
import pytest
from unittest.mock import MagicMock


embedder = Embedder(settings.EMBEDDING_MODEL_NAME)
chunker  = SlidingWindowChunker()
kb = KnowledgeBase(embedder, chunker)
kb.connect()
kb.create_collection("test_col", dim=1024)
kb.load_from_file("ecommerce_customer_service/data/故障排除.txt")
kb.build_index()

@pytest.fixture
def mock_milvus():
    client = MagicMock()
    client.create_collection.return_value = None
    client.load.return_value = None
    return client