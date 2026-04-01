# 技术栈 — 多代理电商客服系统

本文档解释了为什么选择每个技术而不是其替代方案、每个技术的核心概念，以及在实现之前需要理解的最重要的技术决策。

---

## 1. LangGraph（代理框架）

### 为什么选择 LangGraph 而不是原始的 LangChain 代理？

LangChain 的原始 `AgentExecutor` 是一个黑盒 ReAct 循环，很难定制、调试和测试。具体来说：

| AgentExecutor 的问题 | LangGraph 的解决方案 |
|---|---|
| 控制流是隐式的（隐藏在执行器内部） | 显式的 `StateGraph`，带有命名节点和边 |
| 难以在代理之间添加条件路由 | 首先类 `add_conditional_edges()`，带有路由函数 |
| 很难单元测试单个步骤 | 每个节点都是一个普通的 Python 可调用，易于模拟 |
| 没有干净的方式暂停和恢复（人类干预） | 内置的 `interrupt_before` / `interrupt_after` 支持 |
| 状态管理是临时的 | 类型化的 `AgentState` TypedDict 强制执行数据契约 |

### 核心概念

**StateGraph**：中央类。你声明一个状态模式（TypedDict），添加节点（Python 可调用），并用边或条件边连接它们。然后将图编译成一个 `CompiledGraph`，它管理状态转换。

```python
from langgraph.graph import StateGraph, START, END

graph = StateGraph(AgentState)
graph.add_node("router", router_agent.route)
graph.add_conditional_edges("router", routing_fn, {"faq": "faq_agent", ...})
compiled = graph.compile()
```

**节点**：任何签名 `(state: dict) -> dict` 的可调用。返回的 dict 被浅合并（shallow）到共享状态中。节点是无状态的；所有数据都存在于 AgentState 中。

**条件边**：路由函数读取状态并返回一个字符串键。`add_conditional_edges()` 映射将该键转换为下一个节点名称。这就是意图-based 路由的干净实现方式。

**检查点**：将 `MemorySaver`（或 Redis/Postgres 检查点）传递给 `compile()` 以启用持久的多轮会话。每个会话由 `thread_id` 在调用配置中标识。这是推荐的多轮对话状态处理方式。

**流式传输**：使用 `graph.stream(initial_state)` 而不是 `.invoke()` 来实时获取增量节点输出——适用于将令牌流式传输到前端。

### 关键技术决策

将 `AgentState(total=False)` 设置为节点只需要声明它们触及的键。这避免了样板代码，并使以后添加新的状态字段变得容易，而不会破坏现有节点。

---

## 2. Milvus（向量数据库）

### 为什么选择 Milvus 而不是其他向量数据库？

| 替代方案 | 为什么不选择 |
|---|---|
| FAISS | 仅内存，没有持久化，没有内置分布式模式，没有元数据过滤 |
| Chroma | 适合原型，但水平可扩展性有限，没有混合搜索 |
| Weaviate | 部署更复杂；Milvus 在 >10M 向量的基准测试中具有更好的性能 |
| Qdrant | 强大的替代方案，但 Milvus 有更好的原生稀疏+密集混合搜索（Milvus 2.4+） |
| pgvector | Postgres 扩展，适合小数据集；Milvus 是为亿级 ANN 专门构建的 |

Milvus 是**生产就绪的规模**（Bilibili、Shopee 和其他大型电商平台使用它），原生支持混合稀疏+密集搜索，并有成熟的 Python SDK。

### 核心概念

**集合和模式**：基本单位是 `Collection`（类似于 SQL 表）。每个集合有一个模式，定义字段名称和类型：主键、标量元数据字段（VARCHAR、INT64）和 FLOAT_VECTOR 字段用于嵌入。向量字段维度必须匹配嵌入模型输出。

**索引类型**：

- **HNSW**（分层可导航小世界）：基于图的索引。低延迟下最佳召回。参数：`M`（每个节点的边，更高 = 更好召回 + 更多内存）和 `efConstruction`（构建质量）。在查询时使用 `ef` 来权衡速度 vs. 召回。推荐：`M=16, efConstruction=200`。
- **IVF_FLAT**：倒排文件索引。将向量聚类成 `nlist` 个 Voronoi 单元。比 HNSW 构建更快，低内存，召回略低。用于 < 1M 向量的数据集或有限 RAM。
- **IVF_SQ8**：量化的 IVF。vs IVF_FLAT 内存减少 4×，召回下降 ~1%。用于亿级。

**混合搜索（Milvus 2.4+）**：原生支持在单个查询中组合密集向量和稀疏向量（BM25 或 SPLADE）。对密集使用 `AnnSearchRequest`，对稀疏使用 `AnnSearchRequest`，然后用 `RRFRanker` 或 `WeightedRanker` 融合。这比客户端 RRF 更高效，因为它减少了网络往返。

**元数据过滤**：Milvus 支持在 `collection.search()` 的 `expr` 参数中使用标量字段过滤器。示例：`expr="category == 'logistics'"` 在 ANN 之前限制搜索到特定 FAQ 类别。这大大减少延迟并提高精确度。

**分区键**：使用分区键按类别物理分离数据。在集合创建时设置 `partition_key_field="category"`；Milvus 自动将查询路由到正确的分区。

### 关键技术决策

在 `create_index()` 之后调用 `collection.load()` 以将索引加载到内存中。没有此调用，`collection.search()` 将失败。在生产中，使用 `utility.loading_progress()` 来监控加载状态，然后提供流量。

---

## 3. SentenceTransformers（嵌入）

### 为什么选择 SentenceTransformers？

- **开箱即用的多语言**：`paraphrase-multilingual-MiniLM-L12-v2` 在单个模型中处理中文、英文和 50+ 其他语言。没有语言检测或单独模型。
- **简单 API**：`model.encode(texts)` 返回 numpy 数组。没有分词样板。
- **可微调**：域特定微调在查询-段落对上使用 `MultipleNegativesRankingLoss` 可以提高召回 5–15%，只需几百个标记示例。
- **轻量级**：MiniLM-L12-v2 只有 120 MB，并在 CPU 上每秒编码 ~1000 个句子。

### 核心概念

**池化策略**：SentenceTransformers 将 token-level BERT 输出池化成单个句子向量。常见策略：平均池化（默认，最适合语义相似性）、CLS token 池化、最大池化。选择在模型中烘焙——不要在重新训练后更改。

**归一化嵌入**：在 `model.encode()` 中设置 `normalize_embeddings=True` 使余弦相似度等于点积，这更快计算。当使用 Milvus 的 COSINE 度量时始终归一化。

**不对称模型（BGE）**：一些模型如 `BAAI/bge-m3` 是不对称的——它们对查询 vs. 段落使用不同的表示。对于查询，在指令前添加："Represent this sentence for searching relevant passages: "。模型卡指定此。未能添加前缀会降低召回。

**域数据微调**：
1. 收集正对：（客户问题，相关 FAQ 答案）。
2. 使用 `MultipleNegativesRankingLoss` 与批内负样本（无需显式负挖掘）。
3. 用批大小 64–128 训练 1–3 个 epoch。
4. 在保留集上用 Recall@5 评估。
5. 导出并加载微调模型，就像基础模型一样。

### 关键技术决策

明智选择嵌入维度：更大 dim = 更好召回但更多 Milvus 存储和更慢搜索。MiniLM-L12-v2 在 384 dim 是这个用例的甜点。BGE-M3 在 1024 dim 给出 ~5% 更好召回但存储成本翻四倍。

---

## 4. Llama 3 / Qwen（LLM）

### 为什么选择开源 LLM 而不是 GPT-4？

| 考虑因素 | 开源（Qwen/Llama） | GPT-4 API |
|---|---|---|
| 成本 | 硬件成本摊销后近零 | ~$30/1M 令牌 |
| 数据隐私 | 所有数据留在本地 | 客户查询发送到 OpenAI |
| 延迟 | 可控（本地 GPU） | 网络依赖（典型 50–200 ms） |
| 定制 | 完全控制（LoRA，RLHF） | 仅提示工程 |
| 可靠性 | 自管理（无供应商宕机风险） | 依赖 API 可用性 |

对于每天处理数百万客户查询的电商公司，开源模型的成本和隐私优势是决定性的。

### Qwen vs. Llama 3

- **Qwen2.5-7B-Instruct**：中文语言性能更好（Alibaba 在中文数据上训练了它）。推荐用于中文主导的客服。
- **Llama-3-8B-Instruct**：英文性能更好，更大的英文预训练语料库。用于英文或多语言部署。
- **Qwen2.5-72B**：最佳质量，需要 4×A100 GPU。用于最复杂的查询合成。

### 核心概念

**量化**：减少 GPU 内存占用，同时保留大部分质量。
- **GGUF + llama.cpp**：CPU 推理，4-bit 量化。7B 模型适合 ~4 GB RAM。用于开发和低流量部署。
- **AWQ（激活感知权重量化）**：GPU 的 4-bit 量化。质量比 GGUF 更好，同 bit-width。`autoawq` 库。
- **GPTQ**：另一个 4-bit GPU 量化方案。比 AWQ 稍慢但更广泛支持。

**vLLM 服务**：推荐的生产服务框架。关键特性：
- PagedAttention：高效 KV 缓存管理，实现更高的批大小。
- 连续批处理：通过动态批处理传入请求来最大化 GPU 利用率。
- OpenAI 兼容 API：LangChain 的 `ChatOpenAI` 指向 `http://localhost:8000/v1`，无需代码更改。
- 启动命令：`vllm serve Qwen/Qwen2.5-7B-Instruct --host 0.0.0.0 --port 8000 --quantization awq`

**LoRA 微调**：使用低秩适应将基础 LLM 适应电商域。
1. 从历史客户服务日志构建监督微调（SFT）数据集（~50k 示例：客户查询，理想响应）。
2. 使用 `trl` + `peft` 与 LoRA rank=16，alpha=32 微调。
3. 在单个 A100 上训练 ~2 小时，7B 模型。
4. 将 LoRA 权重合并到基础模型中，用于单文件服务，或使用动态适配器加载。

### 关键技术决策

即使对于本地模型，也使用 OpenAI 兼容 API 模式。这使得通过更改 `LLM_API_BASE` 在本地和云 LLM 之间切换变得微不足道——无需代码更改。

---

## 5. RAGAS（评估）

### 为什么选择 RAGAS 而不是手动评估？

手动评估 5 000 个 RAG 响应需要 ~100 小时的人工注释，~2 分钟每个样本。RAGAS 使用 LLM 判断来自动化此过程，将评估时间减少到 ~2 小时，同时保持与人类判断的好相关性。

RAGAS 专门为 RAG 系统设计，并提供一般 NLP 评估工具包中不可用的指标（BLEU、ROUGE、BERTScore）。

### 核心指标解释

**忠实度**（主要指标，用于幻觉检测）：
- 问题："生成的答案是否只包含检索到的上下文支持的声明？"
- 计算：LLM 从答案中提取所有事实声明，然后检查每个声明对上下文。忠实度 = （支持声明） / （总声明）。
- 目标：≥ 0.85。低于 0.7 意味着模型经常编造信息。
- 如何改进：加强系统提示指令，从不超出提供的上下文。降低 LLM 温度。

**答案相关性**（响应质量）：
- 问题："答案是否与用户的查询相关？"
- 计算：RAGAS 从答案生成 N 个改写问题，然后计算每个生成问题与原始问题的余弦相似度。高相似度 = 答案解决了查询。
- 目标：≥ 0.80。低分表示离题或过于冗长的答案。
- 如何改进：在生成提示中添加"保持简洁并直接回答问题"的指令。

**上下文精确度**（检索质量——无需 ground truth）：
- 问题："检索到的块是否实际有助于生成答案？"
- 计算：对于每个检索块，LLM 确定它是否有助于答案。精确度 = （有用块） / （总检索块）。
- 目标：≥ 0.75。低分意味着检索器返回噪声。
- 如何改进：重排序后降低 `top_k`。调整重排序器阈值。添加意图类别元数据过滤。

**上下文召回**（检索完整性——需要 ground truth）：
- 问题："检索到的上下文是否涵盖了正确回答所需的所有事实？"
- 计算：LLM 检查 ground truth 答案中的每个句子对检索上下文。
- 目标：≥ 0.80。

### 关键技术决策

在**自动化回归测试管道**中使用 RAGAS：每次检索配置更改时（新块大小、新嵌入模型、新 top-k），在固定 500 样本评估集上运行它。跟踪时间上的指标趋势，以及早发现回归。

---

## 6. Redis（消息队列 & 长期记忆）

### 为什么为消息队列选择 Redis？

| 替代方案 | 为什么不适合这个用例 |
|---|---|
| Kafka | 更高的运营复杂性，需要 Zookeeper/KRaft；对于 < 1 000 QPS 来说过杀 |
| RabbitMQ | AMQP 协议，更复杂的客户端库，没有原生键值存储用于记忆 |
| SQS（AWS） | 云-only，供应商锁定，~10ms 每个操作的额外延迟 |
| 进程内队列 | 不可持久，崩溃时丢失，无法水平扩展 |

Redis 提供消息队列（通过 LIST 数据结构）和长期用户记忆（通过 HASH 和 STRING 与 TTL）在一个服务中，简化部署。

### 核心概念

**LPUSH / BRPOP 模式**（消息队列）：
- 生产者：`LPUSH queue_key json_message` —— O(1)，原子，线程安全。
- 消费者：`BRPOP queue_key 30` —— 阻塞最多 30 秒等待消息。原子地从尾部弹出。没有轮询循环或睡眠。
- 与天真的 RPOP 循环不同，BRPOP 是基于推送的，当队列为空时使用几乎没有 CPU。

**HSET / HGET**（用户偏好）：
- 在单个 Redis hash 中为每个用户存储多个字段：`HSET user:123:preferences language zh-CN`。
- 无论用户有多少偏好，每个字段访问 O(1)。
- 一个 `HGETALL` 调用获取所有偏好用于提示注入。

**SET 与 TTL**（订单上下文，用户档案）：
- `SET user:123:profile json_blob EX 2592000`（30 天）。
- TTL 防止非活跃用户内存增长——无需清理作业。

**发布/订阅**（可选，用于实时结果交付）：
- 队列工作者处理请求后，发布结果：`PUBLISH result:request_id json_result`。
- API 层订阅以交付答案通过 WebSocket 或长轮询：`SUBSCRIBE result:request_id`。
- 比轮询结果键更高效。

### 关键技术决策

使用 `BRPOP` 与超时（5–30 秒）而不是非阻塞 `RPOP` 在睡眠循环中。BRPOP 在消息到达时立即返回，并在服务器端阻塞等待，否则——这消除了 `sleep()` 间隔的延迟，同时使用没有 CPU。

---

## 7. FastAPI（API 框架）

### 为什么选择 FastAPI 而不是 Flask 或 Django？

| 特性 | FastAPI | Flask | Django |
|---|---|---|---|
| 异步支持 | 原生（asyncio） | 需要扩展 | 有限 |
| 请求验证 | Pydantic（自动） | 手动 | Django REST Framework |
| 自动 API 文档 | OpenAPI / Swagger UI | 需要插件 | DRF Browsable API |
| 性能 | Python 框架中最快的之一 | 更慢（WSGI） | 更慢（WSGI） |
| 类型安全 | 完全（Pydantic 模型） | 无 | 部分 |

对于请求/响应模式复杂的 AI API 服务（嵌套 dict 与可选字段），Pydantic 的自动验证和清晰错误消息是主要生产力提升。

### 核心概念

**依赖注入**（`Depends`）：声明共享资源（编译图，消息队列，Redis 客户端）作为 FastAPI 依赖。FastAPI 一次性实例化它们并注入到每个请求处理器中。这用可测试的注入依赖替换全局单例。

```python
def get_graph() -> CompiledGraph:
    return app.state.graph

@app.post("/chat")
async def chat(req: ChatRequest, graph: CompiledGraph = Depends(get_graph)):
    ...
```

**后台任务**：使用 `BackgroundTasks.add_task()` 用于不需在响应发送前完成的操做（例如，将对话轮保存到记忆，记录到分析）。这保持响应延迟低。

**生命周期上下文管理器**：使用 `@asynccontextmanager` 生命周期（FastAPI 0.95+）而不是已弃用的 `@app.on_event("startup")`。生命周期函数在启动时初始化资源，在关闭时释放它们（Milvus 连接、线程池等的干净拆除）。

**中间件**：添加 `CORSMiddleware` 用于浏览器客户端，`GZipMiddleware` 用于响应压缩，以及自定义中间件用于请求 ID 注入和延迟日志。

### 关键技术决策

当进程内队列工作者线程活跃时，使用 `workers=1` 运行 Uvicorn。Uvicorn 工作者是单独进程，不共享状态；如果你运行多个工作者，每个都会启动自己的队列工作者线程在同一个 Redis 队列上竞争。相反，使用单独的队列工作者进程，并独立扩展 API 和工作者层。