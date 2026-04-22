# 多 Agent 电商客服系统 — 使用文档

本文档覆盖系统完整生命周期：环境搭建 → 知识库构建 → 服务启动 → 对话测试 → 评测。

---

## 目录

1. [环境准备](#1-环境准备)
2. [基础设施启动（Milvus / Redis）](#2-基础设施启动)
3. [配置](#3-配置)
4. [知识库构建](#4-知识库构建)
5. [启动 API 服务](#5-启动-api-服务)
6. [对话测试](#6-对话测试)
7. [队列模式（高并发）](#7-队列模式高并发)
8. [评测](#8-评测)
9. [常见问题](#9-常见问题)

---

## 1. 环境准备

### 1.1 安装依赖

```bash
cd /home/xyc/agent/ecommerce_customer_service
pip install -r requirements.txt
```

### 1.2 启动本地 LLM（以 Qwen 为例）

系统通过 OpenAI 兼容接口调用 LLM，任何支持该接口的服务均可。

**选项 A — vLLM（推荐，GPU）**：

```bash
pip install vllm
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-7B-Instruct \
    --port 8000 \
    --dtype auto
```

**选项 B — Ollama（CPU / 本地测试）**：

```bash
ollama pull qwen2.5:7b
ollama serve   # 默认端口 11434
```

此时需将 `LLM_API_BASE` 改为 `http://localhost:11434/v1`。

---

## 2. 基础设施启动

### 2.1 启动 Milvus（向量数据库）

使用项目根目录下的 standalone 脚本（需要 Docker）：

```bash
cd /home/xyc/agent
bash standalone_embed.sh start
```

验证 Milvus 已启动：

```bash
# 默认端口 19530
curl http://localhost:9091/healthz
# 返回 {"status":"healthy"} 即正常
```

> **说明**：`embedEtcd.yaml` 和 `volumes/` 是 Milvus standalone 的配置和数据目录，已在项目根目录。

### 2.2 启动 Redis（消息队列 / 长期记忆）

```bash
docker run -d --name redis \
    -p 6379:6379 \
    redis:7-alpine
```

验证：

```bash
redis-cli ping   # 返回 PONG
```

---

## 3. 配置

所有配置通过环境变量或 `.env` 文件注入，由 `config/settings.py` 读取。

### 3.1 创建 `.env` 文件

在 `ecommerce_customer_service/` 目录下创建：

```dotenv
# LLM
LLM_MODEL_NAME=Qwen/Qwen2.5-7B-Instruct
LLM_API_BASE=http://localhost:8000/v1
LLM_API_KEY=EMPTY
LLM_TEMPERATURE=0.1
LLM_MAX_TOKENS=1024

# Embedding
EMBEDDING_MODEL_NAME=paraphrase-multilingual-MiniLM-L12-v2
EMBEDDING_BATCH_SIZE=64
EMBEDDING_DIM=384

# Chunking
CHUNK_SIZE=512
CHUNK_OVERLAP=100

# Retrieval
RETRIEVAL_TOP_K=10
RERANK_TOP_N=3
RERANKER_MODEL_NAME=cross-encoder/ms-marco-MiniLM-L-6-v2
HYBRID_SEARCH_ALPHA=0.5

# Milvus
MILVUS_HOST=localhost
MILVUS_PORT=19530
MILVUS_COLLECTION_NAME=ecommerce_faq

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=
REDIS_DB=0
REDIS_QUEUE_KEY=customer_service:queue

# Queue 模式（False = 同步直接调用图，True = 异步队列）
USE_QUEUE=False

# API
API_HOST=0.0.0.0
API_PORT=8080

# 日志
LOG_LEVEL=INFO
```

### 3.2 关键参数说明

| 参数 | 作用 | 调优建议 |
|------|------|----------|
| `CHUNK_SIZE` | 每个文本块的字符数 | 中文 FAQ 建议 256–512 |
| `CHUNK_OVERLAP` | 相邻块重叠字符数 | 约为 chunk_size 的 20% |
| `RETRIEVAL_TOP_K` | 向量召回候选数 | 越大召回越全但越慢，建议 10 |
| `RERANK_TOP_N` | Rerank 后保留条数 | 传给 LLM 的上下文数，建议 3 |
| `HYBRID_SEARCH_ALPHA` | 稠密/稀疏融合权重 | 0.5 = 均等；偏关键词匹配可调低 |

---

## 4. 知识库构建

这是整个系统上线前必须完成的步骤，将原始文档写入 Milvus。

### 4.1 准备数据

将知识文档放入 `data/raw/`，支持 `.txt` 和 `.pdf`：

```
data/
└── raw/
    ├── product_faq.txt       # 商品常见问题
    ├── return_policy.txt     # 退换货政策
    ├── shipping_guide.txt    # 物流说明
    └── product_manual.pdf   # 商品说明书
```

**文档格式示例**（`product_faq.txt`）：

```
Q: 商品支持7天无理由退货吗？
A: 支持。收货后7天内，商品未使用、包装完好，均可申请退货。

Q: 如何查询我的订单状态？
A: 您可以在"我的订单"页面查看，或联系客服提供订单号查询。
```

### 4.2 构建 Milvus Collection 并导入数据

在项目根目录执行：

```python
# build_kb.py（在 ecommerce_customer_service/ 目录下运行）
import os, sys
sys.path.insert(0, ".")

from config.settings import settings
from rag.embedder import Embedder
from rag.chunker import SlidingWindowChunker
from rag.knowledge_base import KnowledgeBase
from pathlib import Path

# 初始化组件
embedder = Embedder(
    model_name=settings.EMBEDDING_MODEL_NAME,
    batch_size=settings.EMBEDDING_BATCH_SIZE
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
    collection_name=settings.MILVUS_COLLECTION_NAME
)

# 连接并建立 Collection
kb.connect()
kb.create_collection(settings.MILVUS_COLLECTION_NAME, dim=embedder.embedding_dim)

# 逐文件导入
data_dir = Path("data/raw")
total = 0
for file_path in data_dir.iterdir():
    if file_path.suffix in (".txt", ".pdf"):
        count = kb.load_from_file(file_path)
        print(f"[+] {file_path.name}: 导入 {count} 个 chunks")
        total += count

# 建立向量索引（导入完成后执行一次）
kb.build_index()
print(f"\n知识库构建完成，共 {total} 个 chunks，HNSW 索引已建立。")
```

```bash
cd /home/xyc/agent/ecommerce_customer_service
python build_kb.py
```

预期输出：

```
[+] product_faq.txt: 导入 142 个 chunks
[+] return_policy.txt: 导入 38 个 chunks
[+] shipping_guide.txt: 导入 61 个 chunks
[+] product_manual.pdf: 导入 203 个 chunks

知识库构建完成，共 444 个 chunks，HNSW 索引已建立。
```

### 4.3 构建 BM25 索引（稀疏检索）

Retriever 的 BM25 索引在首次使用时自动构建（需要内存中的语料），或手动预构建：

```python
from rag.retriever import Retriever

# 从 Milvus 拉取所有文本用于构建 BM25
all_texts = [doc["text"] for doc in kb.get_all_texts()]  # 见 KnowledgeBase
retriever = Retriever(
    knowledge_base=kb,
    embedder=embedder,
    reranker_model_name=settings.RERANKER_MODEL_NAME
)
retriever.build_bm25(all_texts)
print("BM25 索引构建完成")
```

### 4.4 验证知识库

```python
# 快速验证检索是否正常
results = retriever.hybrid_search("退货政策", top_k=3)
for r in results:
    print(f"score={r['score']:.4f} | {r['text'][:80]}")
```

---

## 5. 启动 API 服务

确认 Milvus、Redis、LLM 服务均已运行后：

```bash
cd /home/xyc/agent/ecommerce_customer_service
uvicorn api.main:app --host 0.0.0.0 --port 8080 --workers 1
```

启动日志（正常）：

```
INFO: Loading embedding model...
INFO: Connecting to Milvus at localhost:19530
INFO: Knowledge base ready, collection: ecommerce_faq
INFO: Building agent graph...
INFO: Application started. QueueWorker: disabled
INFO: Uvicorn running on http://0.0.0.0:8080
```

健康检查：

```bash
curl http://localhost:8080/health
```

```json
{
  "status": "ok",
  "version": "1.0.0",
  "milvus_connected": true,
  "queue_size": 0
}
```

---

## 6. 对话测试

### 6.1 命令行 curl 测试

**商品 FAQ 咨询**：

```bash
curl -X POST http://localhost:8080/chat \
  -H "Content-Type: application/json" \
  -d '{
    "query": "商品支持退货吗？",
    "user_id": "user_001",
    "session_id": "sess_test_01",
    "history": []
  }'
```

```json
{
  "answer": "支持。收货后7天内，商品未使用、包装完好，均可申请退货。退款将在审核通过后5个工作日内到账。",
  "intent": "faq",
  "latency_ms": 1243.5,
  "request_id": "req-xxxxxxxx",
  "session_id": "sess_test_01"
}
```

**订单查询（Tool Call）**：

```bash
curl -X POST http://localhost:8080/chat \
  -H "Content-Type: application/json" \
  -d '{
    "query": "帮我查一下订单 ORD-20240310-001 的状态",
    "user_id": "user_001",
    "session_id": "sess_test_01",
    "history": []
  }'
```

**物流查询**：

```bash
curl -X POST http://localhost:8080/chat \
  -H "Content-Type: application/json" \
  -d '{
    "query": "我的快递怎么还没到？订单号 ORD-20240308-002",
    "user_id": "user_001",
    "session_id": "sess_test_01",
    "history": []
  }'
```

**退款申请**：

```bash
curl -X POST http://localhost:8080/chat \
  -H "Content-Type: application/json" \
  -d '{
    "query": "我想申请退款，订单号 ORD-20240310-001，商品有质量问题",
    "user_id": "user_001",
    "session_id": "sess_test_01",
    "history": []
  }'
```

**多轮对话（携带历史）**：

```bash
curl -X POST http://localhost:8080/chat \
  -H "Content-Type: application/json" \
  -d '{
    "query": "那我的退款什么时候到账？",
    "user_id": "user_001",
    "session_id": "sess_test_01",
    "history": [
      {"role": "user", "content": "我想申请退款，订单号 ORD-20240310-001"},
      {"role": "assistant", "content": "已为您创建退款单 REFUND-xxx，预计5个工作日内到账。"}
    ]
  }'
```

### 6.2 Python 客户端测试

```python
import requests

BASE_URL = "http://localhost:8080"
session_id = "sess_python_test"
history = []

def chat(query: str, user_id: str = "test_user") -> str:
    resp = requests.post(f"{BASE_URL}/chat", json={
        "query": query,
        "user_id": user_id,
        "session_id": session_id,
        "history": history
    })
    data = resp.json()
    # 更新历史
    history.append({"role": "user", "content": query})
    history.append({"role": "assistant", "content": data["answer"]})
    print(f"[{data['intent']}] {data['answer']} ({data['latency_ms']:.0f}ms)")
    return data["answer"]

# 多轮对话测试
chat("你们支持7天无理由退货吗？")
chat("我的订单 ORD-20240310-001 可以退吗？")
chat("好的，帮我申请退款，原因是质量问题")
chat("退款什么时候能到？")
```

### 6.3 直接调用图（不经过 API）

```python
import sys
sys.path.insert(0, "/home/xyc/agent/ecommerce_customer_service")

from config.settings import settings
from graph.agent_graph import init_graph, run_graph

graph = init_graph(settings)
answer = run_graph(
    query="ORD-20240308-002 的物流到哪了？",
    user_id="dev_user",
    history=[],
    session_id="dev_session",
    compiled_graph=graph
)
print(answer)
```

---

## 7. 队列模式（高并发）

队列模式将请求异步化，ThreadPoolExecutor 并发处理，适合高 QPS 场景。

### 7.1 启用队列

在 `.env` 中设置：

```dotenv
USE_QUEUE=True
THREAD_POOL_MAX_WORKERS=10
```

重启服务后，FastAPI lifespan 自动启动 `QueueWorker`：

```
INFO: QueueWorker started with 10 workers
INFO: Listening on Redis queue: customer_service:queue
```

### 7.2 发送请求

请求接口不变，仍是 `POST /chat`。在队列模式下，接口立即返回 `request_id`，结果异步写入 Redis。

**生产者（推消息）**：

```python
import redis, json, uuid
from datetime import datetime

r = redis.Redis(host="localhost", port=6379, db=0)

message = {
    "request_id": str(uuid.uuid4()),
    "user_id": "user_bulk_001",
    "session_id": str(uuid.uuid4()),
    "query": "商品有质量问题怎么办？",
    "history": [],
    "timestamp": datetime.utcnow().isoformat()
}

r.lpush("customer_service:queue", json.dumps(message, ensure_ascii=False))
print(f"已入队: {message['request_id']}")
```

**手动启动 Worker（脱离 FastAPI）**：

```python
import redis
from queue.message_queue import RedisQueue
from queue.worker import QueueWorker
from graph.agent_graph import init_graph
from config.settings import settings

redis_client = redis.Redis(host=settings.REDIS_HOST, port=settings.REDIS_PORT)
queue = RedisQueue(redis_client, key=settings.REDIS_QUEUE_KEY)
graph = init_graph(settings)

worker = QueueWorker(
    queue=queue,
    agent_graph=graph.invoke,
    max_workers=settings.THREAD_POOL_MAX_WORKERS
)
worker.start()   # 阻塞运行
```

### 7.3 吞吐量预期

| 配置 | 预期 QPS |
|------|----------|
| 1 worker，无 GPU | ~3 |
| 10 workers，无 GPU | ~12 |
| 10 workers，A100 GPU | ~40–50 |

---

## 8. 评测

### 8.1 准备测试集

测试集为 JSONL 格式，每行一条样本，保存在 `data/test_set/test_qa.jsonl`：

```jsonl
{"question": "商品支持7天无理由退货吗？", "intent": "faq", "ground_truth": "支持，收货后7天内可申请。", "expected_tool": null, "context_docs": []}
{"question": "查询订单 ORD-20240310-001", "intent": "order", "ground_truth": "已签收", "expected_tool": "get_order_status", "context_docs": []}
{"question": "我的快递到哪了", "intent": "logistics", "ground_truth": "正在运输中", "expected_tool": "query_logistics", "context_docs": []}
```

### 8.2 运行 RAGAS 评测（FAQ 质量）

RAGAS 评测 RAG 系统生成质量，适用于 FAQ 类问题。

```python
import sys
sys.path.insert(0, "/home/xyc/agent/ecommerce_customer_service")

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from evaluation.ragas_evaluator import RAGASEvaluator
from config.settings import settings

# 初始化 LLM（用于 RAGAS 自动评判）
llm = ChatOpenAI(
    model=settings.LLM_MODEL_NAME,
    base_url=settings.LLM_API_BASE,
    api_key=settings.LLM_API_KEY,
    temperature=0
)
embeddings = OpenAIEmbeddings(
    base_url=settings.LLM_API_BASE,
    api_key=settings.LLM_API_KEY
)

evaluator = RAGASEvaluator(llm=llm, embeddings=embeddings)

# 准备评测数据
questions    = ["商品支持7天无理由退货吗？", "如何查询订单？"]
answers      = ["支持，收货后7天内可申请退货。", "您可以在订单页面查看。"]
contexts     = [["收货后7天内，商品未使用可退货。"], ["登录后在我的订单查看状态。"]]
ground_truths = ["支持7天退货", "在订单页面查询"]

results = evaluator.evaluate_dataset(questions, answers, contexts, ground_truths)
report  = evaluator.generate_report(results)
print(report)
```

预期输出：

```
## RAGAS Evaluation Report

| Metric             | Score  |
|--------------------|--------|
| faithfulness       | 0.81   |
| answer_relevance   | 0.77   |
| context_precision  | 0.76   |
| **composite**      | **0.78** |

Target: 0.79 ✓
```

### 8.3 运行离线批量评测

对全量测试集（5000 条）跑批量推理并统计指标：

```python
import sys
sys.path.insert(0, "/home/xyc/agent/ecommerce_customer_service")

from evaluation.offline_evaluator import OfflineEvaluator
from evaluation.ragas_evaluator import RAGASEvaluator
from graph.agent_graph import init_graph
from config.settings import settings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# 初始化
graph = init_graph(settings)
llm   = ChatOpenAI(model=settings.LLM_MODEL_NAME,
                   base_url=settings.LLM_API_BASE,
                   api_key=settings.LLM_API_KEY)
embeddings = OpenAIEmbeddings(base_url=settings.LLM_API_BASE,
                               api_key=settings.LLM_API_KEY)

ragas_eval = RAGASEvaluator(llm=llm, embeddings=embeddings)
evaluator  = OfflineEvaluator(ragas_evaluator=ragas_eval, max_workers=20)

# 加载测试集
test_set = evaluator.load_test_set("data/test_set/test_qa.jsonl")
print(f"测试集大小: {len(test_set)} 条")

# 运行评测
results = evaluator.run_evaluation(graph.invoke, test_set)

# 保存报告
evaluator.save_report(results, "evaluation_report.json")
print("评测完成，报告已保存至 evaluation_report.json")
```

### 8.4 核心指标解读

| 指标 | 计算方式 | 目标值 |
|------|----------|--------|
| **Intent Accuracy** | 意图分类正确率 | ≥ 93% |
| **Tool Success Rate** | 工具调用成功率 | ≥ 96% |
| **Top-5 Recall** | 知识库 Top-5 命中率 | ≥ 88% |
| **RAGAS Composite** | faithfulness + relevance + precision 均值 | ≥ 0.79 |
| **P95 Latency** | 95% 请求的响应时间 | ≤ 3s |

### 8.5 各组件单独评测

**检索质量（仅 RAG）**：

```python
from rag.retriever import Retriever

# 手工评测 Top-5 Recall
test_queries = [
    ("退货政策", "7天无理由"),
    ("如何查订单", "订单页面"),
]
hits = 0
for query, expected_keyword in test_queries:
    docs = retriever.hybrid_search(query, top_k=5)
    if any(expected_keyword in d["text"] for d in docs):
        hits += 1

recall = hits / len(test_queries)
print(f"Top-5 Recall: {recall:.2%}")
```

**意图分类准确率**：

```python
from agents.router_agent import RouterAgent, IntentType

test_cases = [
    ("商品有质量问题", IntentType.FAQ),
    ("查一下我的订单", IntentType.ORDER),
    ("快递到哪了", IntentType.LOGISTICS),
    ("我要退款", IntentType.REFUND),
]
correct = sum(
    router.classify_intent(q) == expected
    for q, expected in test_cases
)
print(f"Intent Accuracy: {correct}/{len(test_cases)} = {correct/len(test_cases):.2%}")
```

---

## 9. 常见问题

### Q: Milvus 连接失败

```
pymilvus.exceptions.MilvusException: connect failed
```

检查：
```bash
curl http://localhost:9091/healthz   # Milvus 健康检查
docker ps | grep milvus              # 确认容器运行中
```

### Q: Embedding 模型下载慢

首次运行会从 HuggingFace 下载模型，设置镜像加速：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

### Q: LLM 返回乱码或空响应

检查 `LLM_API_BASE` 和 `LLM_MODEL_NAME` 是否与实际服务匹配：

```bash
curl http://localhost:8000/v1/models   # 查看可用模型列表
```

### Q: Redis 长期记忆不生效

确认 Redis 连接正常，且 `user_id` 在多轮请求中保持一致（不能用默认随机 ID）。

### Q: RAGAS 评分极低（< 0.4）

通常是 LLM 输出语言不一致导致，RAGAS 评判 LLM 需要能理解问题语言。建议使用支持中文的模型，或将测试集转为英文评测。

### Q: 重置知识库

```python
kb.drop_collection()          # 删除现有 Collection
kb.create_collection(...)     # 重新创建
# 重新导入数据
```

---

## 附：完整启动 Checklist

```
□ Docker 运行中
□ bash standalone_embed.sh start  → Milvus 启动（端口 19530）
□ docker run redis                → Redis 启动（端口 6379）
□ vLLM / Ollama 启动             → LLM 服务（端口 8000）
□ .env 配置完成
□ python build_kb.py              → 知识库构建完成
□ uvicorn api.main:app ...        → API 启动
□ curl /health → {"status":"ok"}  → 验证通过
```
