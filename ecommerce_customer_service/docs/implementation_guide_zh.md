# 实施指南 — 多代理电商客服系统

本指南提供了一个分阶段、依赖有序的实施计划。按照阶段顺序：每个阶段建立在上一个阶段的基础上，并且每个阶段内的每个文件都列出了它在下一个之前的原因。

---

## 概述

```
阶段 1 — 基础        (第 1 周)：设置、模拟服务、工具、基本测试
阶段 2 — RAG 管道      (第 2 周)：分块器 → 嵌入器 → 知识库 → 检索器
阶段 3 — 记忆            (第 2 周)：短期 → 长期
阶段 4 — 代理            (第 3 周)：响应代理 → FAQ 代理 → 订单代理 → 路由代理
阶段 5 — 图              (第 3 周)：代理图 → 端到端测试
阶段 6 — API & 队列       (第 4 周)：消息队列 → 工作者 → api/main.py
阶段 7 — 评估            (第 4 周)：RAGAS 评估器 → 离线评估器 → 基线运行
阶段 8 — 优化            (第 5 周)：调整检索，调整路由，提高提示
```

---

## 阶段 1 — 基础（第 1 周）

**目标**：获得一个可运行的环境，具有配置、模拟数据和工具骨架，以便所有后续阶段都有具体的构建对象。

### 步骤 1.1：`config/settings.py`

首先实施，因为每个其他模块都从它导入。实施是微不足道的（pydantic-settings BaseSettings），但必须在任何其他东西可以配置之前完成。

关键函数实施：
- `Settings.__init__` —— 让 pydantic-settings 处理它；验证 `.env` 文件加载工作。
- 测试：`python -c "from config import settings; print(settings.MILVUS_HOST)"` 应该打印 `localhost`。

### 步骤 1.2：`services/mock_order_service.py`

接下来实施，因为工具函数（阶段 1.3）委托给这个服务。让服务先工作，使工具测试独立于工具包装器。

按此顺序实施：
1. `__init__` —— 将种子数据 dict 复制到实例属性。
2. `get_order` —— 简单 dict 查找。
3. `get_order_status` —— 在内部使用 `get_order`。
4. `get_orders_by_user` —— 按 user_id 过滤。
5. `check_refund_eligibility` —— 最逻辑重的函数；仔细测试 7 天窗口。
6. `create_refund` —— 在内部调用 `check_refund_eligibility`。
7. `get_refund_status` —— 简单 dict 查找。

关键路径：`check_refund_eligibility` 被 `create_refund` 调用；正确获取时间比较。

### 步骤 1.3：`services/mock_logistics_service.py`

与 mock_order_service 并行实施（无依赖）。

实施 `get_logistics` 和 `get_tracking` —— 两者都是从种子数据 dict 查找。

### 步骤 1.4：`tools/order_tools.py`、`tools/logistics_tools.py`、`tools/refund_tools.py`

工具包装器是薄的：每个是 `@tool`-装饰的函数，调用相应的模拟服务方法。主要实施任务是验证 `@tool` 装饰器正确应用，函数 docstring 信息丰富（LLM 使用 docstring 作为工具描述）。

每个工具的实施步骤：
1. 导入模块级服务单例。
2. 调用 `service.method(args)` 并返回结果。
3. 用：`from tools.order_tools import get_order_status; print(get_order_status.name)` 验证。

### 步骤 1.5：`tests/test_tools.py`

现在运行工具测试，在接触任何代理代码之前。这些测试快速（无需 LLM），并验证整个模拟服务 + 工具层。

运行：`pytest tests/test_tools.py -v`

阶段 1 结束时，所有测试应该通过。如果 `check_refund_eligibility` 测试由于日期逻辑失败，在继续之前修复服务。

---

## 阶段 2 — RAG 管道（第 2 周）

**目标**：构建文档摄取和检索管道。到这个阶段结束时，你应该能够将 FAQ 文档插入 Milvus 并检索相关块用于查询。

### 步骤 2.1：`rag/chunker.py`

首先实施 `SlidingWindowChunker.chunk()`。它没有外部依赖，易于单元测试。

实施说明：
- 在 `__init__` 中验证 `chunk_size > overlap`。
- 使用 `while` 循环与 `stride = chunk_size - overlap` 而不是 `range()` 以避免 off-by-one 错误。
- `chunk_with_metadata` 是微不足道的包装器；在 `chunk` 之后实施。

立即测试：`pytest tests/test_rag.py::TestSlidingWindowChunker -v`

### 步骤 2.2：`rag/embedder.py`

在分块器之后实施。因为 `KnowledgeBase.load_from_file()` 调用两者。

实施说明：
- 使 `_load_model()` 幂等（仅在 `self.model is None` 时加载）。
- 总是传递 `normalize_embeddings=True` 到 `model.encode()`。
- 返回 `vectors.tolist()`（不是 numpy 数组）用于 JSON 可序列化和 Milvus 兼容性。
- 通过捕获 `RuntimeError` 并重试 `batch_size // 2` 来处理 GPU OOM。

测试：`pytest tests/test_rag.py::TestEmbedder -v`（所有测试使用模拟，无需 GPU）。

### 步骤 2.3：`rag/knowledge_base.py`

在嵌入器之后实施。摄取管道（`load_from_file`）链接分块器 + 嵌入器 + Milvus 插入。

实施顺序：
1. `connect()` —— 一行 `connections.connect()` 调用。
2. `create_collection()` —— 定义模式并处理"集合已存在"情况。
3. `insert()` —— 批插入与 `collection.flush()`。
4. `build_index()` —— 创建 HNSW 索引并调用 `collection.load()`。
5. `load_from_file()` —— 为每个文档链接以上所有。

关键：在所有数据插入后调用 `build_index()`，而不是每个批次后。部分填充集合构建索引浪费时间。

集成测试（手动，需要运行 Milvus）：
```python
from config import settings
from rag import SlidingWindowChunker, Embedder, KnowledgeBase
embedder = Embedder(settings.EMBEDDING_MODEL_NAME)
chunker  = SlidingWindowChunker()
kb = KnowledgeBase(embedder, chunker)
kb.connect()
kb.create_collection("test_col", dim=384)
kb.load_from_file("data/raw/faq.jsonl")
kb.build_index()
```

### 步骤 2.4：`rag/retriever.py`

在 knowledge_base 之后实施。检索器假设集合已连接和加载。

实施顺序：
1. `dense_search()` —— 调用 `collection.search()` 并解析 Hit 对象。
2. `build_bm25()` —— 用 jieba 分词语料，构建 BM25Okapi 索引。
3. `sparse_search()` —— 调用 `bm25_index.get_scores()` 并返回 top-k。
4. `hybrid_search()` —— 实施 RRF 融合结合密集和稀疏结果。
5. `rerank()` —— 懒加载 CrossEncoder，调用 `.predict()`，按分数排序。

测试顺序：首先测试 `dense_search`（最简单），然后 `sparse_search`，然后 `hybrid_search`（取决于两者）。独立测试 `rerank`。

运行：`pytest tests/test_rag.py::TestRetriever -v`

---

## 阶段 3 — 记忆（第 2 周）

**目标**：实施两个记忆层。这些很简单，但必须在代理之前完成，以便代理可以使用历史。

### 步骤 3.1：`memory/short_term.py`

实施所有方法。这是一个薄包装 `collections.deque` —— 应该 < 30 分钟。

手动测试：
```python
from memory import ShortTermMemory
mem = ShortTermMemory(max_turns=3)
mem.add_turn("user", "hello")
mem.add_turn("assistant", "hi")
print(mem.get_history())       # [{"role": "user", "content": "hello"}, ...]
print(mem.to_prompt_string())  # User: hello\nAssistant: hi
```

### 步骤 3.2：`memory/long_term.py`

在 short_term 之后实施。需要运行 Redis 实例（使用 Docker：`docker run -d -p 6379:6379 redis`）。

实施说明：
- 从 Redis 解码字节：`value.decode("utf-8") if isinstance(value, bytes) else value`。
- 对结构化数据（订单上下文，配置文件）使用 `json.dumps` / `json.loads`。
- 每次写入时总是设置 TTL 以防止无界内存增长。

手动测试：
```python
import redis
from memory import LongTermMemory
r = redis.Redis()
mem = LongTermMemory(r, "user_test")
mem.save_preference("language", "zh-CN")
print(mem.get_preference("language"))  # zh-CN
```

---

## 阶段 4 — 代理（第 3 周）

**目标**：实施四个代理类。按照反依赖顺序实施：`ResponseAgent` 首先（最简单，无依赖），然后到 `RouterAgent`（最复杂）。

### 步骤 4.1：`agents/response_agent.py`

最简单的代理 —— 它只是用合成提示调用 LLM。

实施步骤：
1. 存储 `self.llm` 并定义 `self.synthesis_prompt`（ChatPromptTemplate）。
2. 实施 `synthesize()`：
   - 将 tool_results 格式化为 bullet points。
   - 将 rag_results 格式化为编号段落。
   - 检测空结果情况并在没有 LLM 调用时返回后备消息。
   - 调用 `self.llm.invoke(messages)` 并返回 `.content`。
3. 实施 `run()` 作为对 `synthesize()` 的薄包装。

用模拟 LLM 测试以验证状态键更新。

### 步骤 4.2：`agents/faq_agent.py`

在 response_agent 和检索器（阶段 2.4）之后实施。

实施步骤：
1. 存储 `self.llm`、`self.retriever` 并定义 `self.prompt_template`。
2. 实施 `retrieve_context()` —— 调用 `retriever.hybrid_search()` 然后 `retriever.rerank()`。
3. 实施 `generate_answer()` —— 构建带编号上下文的提示，调用 LLM，处理空上下文。
4. 实施 `run()` —— 组合检索 + 生成，更新状态。

关键：生成提示必须明确说"不要包括提供的上下文中不存在的任何信息。"这是 RAGAS 忠实度的主要杠杆。

测试：`pytest tests/test_agents.py::TestFAQAgent -v`

### 步骤 4.3：`agents/order_agent.py`

最复杂的代理由于 ReAct 工具调用循环。

实施步骤：
1. `__init__`：用 `llm.bind_tools(tools)` 绑定工具，构建 `tool_map` dict，存储 `max_iterations`。
2. `select_tool()`：首先实施规则-based 方法（意图 → 默认工具映射），然后可选添加 LLM-based 后备。
3. `execute_tool()`：在 `tool_map` 中查找工具，调用 `.invoke(params)`，将异常包装在错误 dict 中。
4. `run()`：实施 ReAct 循环。这是关键路径：
   - 构建初始消息。
   - 循环直到 `ai_message.tool_calls` 为空或 `max_iterations` 达到。
   - 对于每个工具调用：提取名称和 args，调用 `execute_tool`，追加 `ToolMessage`。
   - 将所有工具结果收集到 `tool_results` 列表中。
   - 返回状态更新。

测试：`pytest tests/test_agents.py::TestOrderAgent -v`

### 步骤 4.4：`agents/router_agent.py`

在代理中最后实施，因为它分派到所有其他。

实施步骤：
1. `__init__`：存储 LLM。可选初始化 `self.classifier = None`。
2. `rewrite_query()`：写 few-shot LLM 提示。测试模糊查询如"那个订单"（代词解析）。
3. `classify_intent()`：从 LLM few-shot 分类开始。提示应该列出所有 IntentType 值，每个一个示例，并指示 LLM 只输出标签。
4. `route()`：组合重写 + 分类，更新状态，返回节点名称字符串。

关键：`route()` 方法必须返回一个**字符串**（下一个节点名称），不是 dict。这是 LangGraph 条件路由约定。

测试：`pytest tests/test_agents.py::TestRouterAgent -v`

---

## 阶段 5 — 图（第 3 周）

**目标**：将所有代理连接成可运行的 LangGraph StateGraph。

### 步骤 5.1：`graph/agent_graph.py`

在所有四个代理实施后实施 `build_graph()` 和 `run_graph()`。

实施步骤：
1. 按照 docstring 中的拓扑实施 `build_graph()`。
2. 实施 `run_graph()` 作为对 `compiled_graph.invoke()` 的薄包装。
3. 实施 `init_graph()` 以连接设置、LLM、RAG 组件和代理。

端到端测试（项目中最重要测试）：
```python
from graph.agent_graph import init_graph, run_graph

# 需要：运行 vLLM 服务器，运行 Milvus 与加载集合，运行 Redis
graph = init_graph()
answer = run_graph("我的订单 ORD-20240310-001 到哪里了？", user_id="user_123")
print(answer)
```

如果这工作，核心系统是功能性的。

---

## 阶段 6 — API & 队列（第 4 周）

### 步骤 6.1：`queue/message_queue.py`

在图之后实施。

实施 `RedisQueue`（推送，弹出，窥视，大小）。BRPOP 模式是唯一棘手的部分 —— 在继续之前手动测试阻塞行为。

跳过 `KafkaQueue` 现在除非你需要它；最后实施 `KafkaQueue.push()` 和 `KafkaQueue.pop()`。

### 步骤 6.2：`queue/worker.py`

在 `RedisQueue` 之后实施。

实施步骤：
1. `__init__`：初始化执行器，停止事件，期货列表。
2. `process_message()`：调用代理图，测量延迟，调用 result_callback。
3. `start()`：主 BRPOP 循环与 ThreadPoolExecutor.submit()。
4. `stop()`：设置停止事件，等待期货，关闭执行器。
5. `start_in_background()`：将 `start()` 包装在守护 Thread 中。

测试：启动工作者，推送到 Redis 的测试消息，验证它在线程中处理。

### 步骤 6.3：`api/main.py`

在图和队列工作后实施。

实施步骤：
1. 实施 `lifespan()` 上下文管理器：`init_graph()`，创建 `RedisQueue`，可选启动 `QueueWorker`。
2. 实施 `get_compiled_graph()` 依赖。
3. 实施 `POST /chat` 端点：调用 `run_graph()`，构建 `ChatResponse`。
4. 实施 `GET /health` 端点。
5. 实施 `update_short_term_memory()` 后台任务。

用 FastAPI TestClient 测试：
```python
from fastapi.testclient import TestClient
from api.main import app
client = TestClient(app)
resp = client.get("/health")
assert resp.status_code == 200
```

---

## 阶段 7 — 评估（第 4 周）

### 步骤 7.1：`evaluation/ragas_evaluator.py`

在 FAQ 代理工作后实施（你需要真实 RAG 输出来评估）。

实施步骤：
1. `__init__`：用判断 LLM 配置 RAGAS 指标。
2. `evaluate_dataset()`：构建 HuggingFace Dataset，调用 `ragas.evaluate()`，解析结果。
3. `generate_report()`：格式化为 markdown 表。

在扩展到 5 000 之前，在 50 个 FAQ 样本上运行快速评估以验证管道。

### 步骤 7.2：`evaluation/offline_evaluator.py`

在 `ragas_evaluator` 和 `run_graph` 之后实施。

实施步骤：
1. `load_test_set()`：加载并验证 JSONL 测试集。
2. `run_evaluation()`：实施并行推理循环与 tqdm 进度条。
3. `compute_metrics()`：计算意图准确性，工具成功率，RAGAS 复合。
4. `save_report()`：将 JSON 写入 `data/test_set/baseline_report.json`。

运行基线评估：
```bash
python -c "
from evaluation.offline_evaluator import OfflineEvaluator
from graph.agent_graph import init_graph

graph = init_graph()
evaluator = OfflineEvaluator(max_workers=10)
test_set = evaluator.load_test_set('data/test_set/test_5000.jsonl')
results = evaluator.run_evaluation(graph, test_set)
evaluator.save_report(results, 'data/test_set/baseline_report.json')
print(results['metrics'])
"
```

优化前的预期基线指标：
- 意图准确性：~82%
- 工具成功率：~86%
- RAGAS 复合：~0.62

---

## 阶段 8 — 优化（第 5 周）

**目标**：使用基线作为参考改进关键指标。每次更改后重新运行离线评估器以测量影响。

### 8.1 检索优化

目标：Top-5 召回 71% → 88%。

步骤（按优先级）：
1. **添加混合搜索**：如果只运行密集搜索，启用 BM25 并 RRF 融合。这通常添加 10+ 召回百分比点。
2. **添加重排序**：在混合搜索后添加 CrossEncoder 重排序器。通常添加 3–5 pp。
3. **调整块大小**：在 chunk_size ∈ {256, 512, 768} 上评估召回。较小块改进精确度；较大块改进多事实问题的召回。
4. **调整重叠**：更高重叠减少边界伪影。
5. **添加元数据过滤**：在 ANN 搜索前按意图类别过滤以减少噪声。

### 8.2 路由准确性优化

目标：意图准确性 82% → 93%。

步骤：
1. **改进 few-shot 示例**：在分类提示中为每个意图添加 2–3 个代表性示例。
2. **微调分类器**：收集每个意图 500+ 个标记示例并微调 `bert-base-chinese` 分类器。用微调模型替换 LLM 分类调用以获得 10× 更低延迟。
3. **添加查询重写在分类前**：验证重写首先改进分类（应该，因为查询变得更明确）。

### 8.3 工具调用优化

目标：工具成功率 86% → 96%。

步骤：
1. **改进工具 docstring**：LLM 使用 docstring 进行工具选择。使它们更具体关于何时使用每个工具。
2. **添加参数提取提示**：对于订单 ID 提取，在工具调用前添加专用提取步骤。
3. **添加错误恢复**：当工具调用失败时，用不同参数重试而不是立即失败。

### 8.4 吞吐量优化

目标：QPS 12 → 46。

步骤：
1. **增加线程池大小**：从 `max_workers=10` 开始，用负载测试测量 QPS，增加直到 LLM 服务器是瓶颈。
2. **LLM 响应缓存**：在 Redis 中缓存相同（重写查询，意图）对 1 小时。常见问题如"退款政策"每次命中相同答案。
3. **批嵌入**：确保嵌入器在使用全批编码（不是逐一）在 RAG 中。
4. **vLLM 连续批处理**：验证 LLM 服务器有动态批处理启用（vLLM 中默认）。增加 `--max-num-seqs` 用于更高批大小。

---

## 快速参考：文件 → 阶段映射

| 文件                                 | 阶段 | 依赖                   |
| ------------------------------------ | ---- | ---------------------- |
| `config/settings.py`                 | 1    | —                      |
| `services/mock_order_service.py`     | 1    | 设置                   |
| `services/mock_logistics_service.py` | 1    | —                      |
| `tools/order_tools.py`               | 1    | mock_order_service     |
| `tools/logistics_tools.py`           | 1    | mock_logistics_service |
| `tools/refund_tools.py`              | 1    | mock_order_service     |
| `tests/test_tools.py`                | 1    | 工具                   |
| `rag/chunker.py`                     | 2    | —                      |
| `rag/embedder.py`                    | 2    | —                      |
| `rag/knowledge_base.py`              | 2    | 分块器，嵌入器         |
| `rag/retriever.py`                   | 2    | knowledge_base，嵌入器 |
| `tests/test_rag.py`                  | 2    | rag/*                  |
| `memory/short_term.py`               | 3    | —                      |
| `memory/long_term.py`                | 3    | redis                  |
| `agents/response_agent.py`           | 4    | LLM                    |
| `agents/faq_agent.py`                | 4    | 检索器，LLM            |
| `agents/order_agent.py`              | 4    | 工具，LLM              |
| `agents/router_agent.py`             | 4    | LLM                    |
| `tests/test_agents.py`               | 4    | 代理                   |
| `graph/agent_graph.py`               | 5    | 所有代理               |
| `queue/message_queue.py`             | 6    | redis                  |
| `queue/worker.py`                    | 6    | message_queue，图      |
| `api/main.py`                        | 6    | 图，队列               |
| `evaluation/ragas_evaluator.py`      | 7    | LLM，嵌入              |
| `evaluation/offline_evaluator.py`    | 7    | ragas_evaluator，图    |

---

## 开发环境设置

```bash
# 1. 创建虚拟环境
python -m venv .venv && source .venv/bin/activate

# 2. 安装依赖
pip install -r requirements.txt

# 3. 启动 Milvus（独立，Docker）
# docker run -d --name milvus \
#   -p 19530:19530 -p 9091:9091 \
#   milvusdb/milvus:v2.4.9 standalone
bash standalone_embed.sh start

# 4. 启动 Redis
docker run -d --name redis -p 6379:6379 redis:7.4

# 5. 启动 vLLM（需要 GPU 和 ~16GB VRAM 用于 7B 模型）
vllm serve Qwen/Qwen2.5-7B-Instruct \
  --host 0.0.0.0 --port 8000 \
  --quantization awq

# 6. 复制并编辑配置
cp .env.example .env
# 编辑 .env 以设置 LLM_API_BASE，MILVUS_HOST，REDIS_HOST 等。

# 7. 运行测试
pytest tests/ -v

# 8. 启动 API
uvicorn api.main:app --host 0.0.0.0 --port 8080 --reload
```