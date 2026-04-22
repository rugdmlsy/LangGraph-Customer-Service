import os, sys
# 获取项目根目录
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import torch
print(f"[device] CUDA available: {torch.cuda.is_available()}", flush=True)
if torch.cuda.is_available():
    print(f"[device] GPU: {torch.cuda.get_device_name(0)}", flush=True)

from config import settings
from rag.embedder import Embedder
from rag.chunker import SlidingWindowChunker
from rag.knowledge_base import KnowledgeBase
from pathlib import Path



# 初始化组件
embedder = Embedder(
    model_name=settings.EMBEDDING_MODEL_NAME,
    batch_size=settings.EMBEDDING_BATCH_SIZE,
    device=settings.EMBEDDING_DEVICE
)
chunker = SlidingWindowChunker(
    chunk_size=settings.CHUNK_SIZE,
    overlap=settings.CHUNK_OVERLAP
)
kb = KnowledgeBase(
    embedder=embedder,
    chunker=chunker,
    host=settings.MILVUS_HOST,
    port=settings.MILVUS_PORT,
    collection_name=settings.MILVUS_COLLECTION_NAME,
    batch_size=settings.MILVUS_BATCH_SIZE
)

def build_milvus():
    """构建 Milvus 知识库：连接 → 创建 Collection → 导入数据 → 建立索引"""

    # 连接并建立 Collection
    kb.connect()

    # 如果 collection 已存在但没有索引，先删除
    if kb.has_collection(settings.MILVUS_COLLECTION_NAME):
        print(f"Collection '{settings.MILVUS_COLLECTION_NAME}' 已存在，删除后重建...")
        kb.drop_collection()

    kb.create_collection(settings.MILVUS_COLLECTION_NAME, dim=embedder.embedding_dim)

    # 逐文件导入
    data_dir = Path("ecommerce_customer_service/data/cleaned")
    total = 0
    for file_path in data_dir.iterdir():
        if file_path.suffix in (".txt", ".pdf"):
            count = kb.load_from_file(file_path)
            print(f"[+] {file_path.name}: 导入 {count} 个 chunks")
            total += count

    # 建立向量索引（导入完成后执行一次）
    kb.build_index()
    print(f"\n知识库构建完成，共 {total} 个 chunks，HNSW 索引已建立。")
    
if __name__ == "__main__":
    build_milvus()
    