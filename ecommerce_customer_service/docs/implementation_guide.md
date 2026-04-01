# Implementation Guide — Multi-Agent E-commerce Customer Service System

This guide provides a step-by-step, dependency-ordered implementation plan. Follow the phases in order: each phase builds on the previous one, and every file within a phase is listed with the reason it comes before the next.

---

## Overview

```
Phase 1 — Foundation        (Week 1): settings, mock services, tools, basic tests
Phase 2 — RAG Pipeline      (Week 2): chunker → embedder → knowledge_base → retriever
Phase 3 — Memory            (Week 2): short_term → long_term
Phase 4 — Agents            (Week 3): response_agent → faq_agent → order_agent → router_agent
Phase 5 — Graph             (Week 3): agent_graph → end-to-end test
Phase 6 — API & Queue       (Week 4): message_queue → worker → api/main.py
Phase 7 — Evaluation        (Week 4): ragas_evaluator → offline_evaluator → baseline run
Phase 8 — Optimisation      (Week 5): tune retrieval, tune routing, improve prompts
```

---

## Phase 1 — Foundation (Week 1)

**Goal**: Get a runnable environment with configuration, mock data, and tool skeletons so all subsequent phases have something concrete to build against.

### Step 1.1: `config/settings.py`

Implement first because every other module imports from it. The implementation is trivial (pydantic-settings BaseSettings) but must be done before anything else can be configured.

Critical functions to implement:
- `Settings.__init__` — just let pydantic-settings handle it; verify `.env` file loading works.
- Test: `python -c "from config import settings; print(settings.MILVUS_HOST)"` should print `localhost`.

### Step 1.2: `services/mock_order_service.py`

Implement next because the tool functions (Phase 1.3) delegate to this service. Having the service working first makes tool testing independent of tool wrappers.

Implement in this order:
1. `__init__` — copy seed data dicts into instance attributes.
2. `get_order` — simple dict lookup.
3. `get_order_status` — use `get_order` internally.
4. `get_orders_by_user` — filter by user_id.
5. `check_refund_eligibility` — the most logic-heavy method; test the 7-day window carefully.
6. `create_refund` — calls `check_refund_eligibility` internally.
7. `get_refund_status` — simple dict lookup.

Critical path: `check_refund_eligibility` is called by `create_refund`; get the time comparison right.

### Step 1.3: `services/mock_logistics_service.py`

Implement in parallel with mock_order_service (no dependency).

Implement `get_logistics` and `get_tracking` — both are simple dict lookups from seed data.

### Step 1.4: `tools/order_tools.py`, `tools/logistics_tools.py`, `tools/refund_tools.py`

The tool wrappers are thin: each is a `@tool`-decorated function that calls the corresponding mock service method. The main implementation task is verifying the `@tool` decorator is applied correctly and the function docstring is informative (LLMs use docstrings as tool descriptions).

Implementation steps for each tool:
1. Import the module-level service singleton.
2. Call `service.method(args)` and return the result.
3. Verify with: `from tools.order_tools import get_order_status; print(get_order_status.name)`.

### Step 1.5: `tests/test_tools.py`

Write and run the tool tests now, before touching any agent code. These tests are fast (no LLM required) and validate the entire mock service + tool layer.

Run: `pytest tests/test_tools.py -v`

All tests should pass at the end of Phase 1. If `check_refund_eligibility` tests fail due to date logic, fix the service before moving on.

---

## Phase 2 — RAG Pipeline (Week 2)

**Goal**: Build the document ingestion and retrieval pipeline. By the end of this phase you should be able to insert FAQ documents into Milvus and retrieve relevant chunks for a query.

### Step 2.1: `rag/chunker.py`

Implement `SlidingWindowChunker.chunk()` first. It has zero external dependencies and is easy to unit-test.

Implementation notes:
- Validate `chunk_size > overlap` in `__init__`.
- Use a `while` loop with `stride = chunk_size - overlap` rather than `range()` to avoid off-by-one errors.
- `chunk_with_metadata` is a trivial wrapper; implement after `chunk`.

Test immediately: `pytest tests/test_rag.py::TestSlidingWindowChunker -v`

### Step 2.2: `rag/embedder.py`

Implement after chunker because `KnowledgeBase.load_from_file()` calls both.

Implementation notes:
- Make `_load_model()` idempotent (only load if `self.model is None`).
- Always pass `normalize_embeddings=True` to `model.encode()`.
- Return `vectors.tolist()` (not numpy arrays) for JSON serializability and Milvus compatibility.
- Handle GPU OOM by catching `RuntimeError` and retrying with `batch_size // 2`.

Test: `pytest tests/test_rag.py::TestEmbedder -v` (all tests use mocks, no GPU needed).

### Step 2.3: `rag/knowledge_base.py`

Implement after embedder. The ingestion pipeline (`load_from_file`) chains chunker + embedder + Milvus insert.

Implementation order:
1. `connect()` — one-line `connections.connect()` call.
2. `create_collection()` — define schema and handle the "collection already exists" case.
3. `insert()` — batch insert with `collection.flush()`.
4. `build_index()` — create HNSW index and call `collection.load()`.
5. `load_from_file()` — chain all of the above for each document.

Critical: Call `build_index()` after all data is inserted, not after each batch. Building the index on a partially populated collection wastes time.

Integration test (manual, requires running Milvus):
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

### Step 2.4: `rag/retriever.py`

Implement after knowledge_base. The retriever assumes the collection is already connected and loaded.

Implementation order:
1. `dense_search()` — call `collection.search()` and parse Hit objects.
2. `build_bm25()` — tokenise corpus with jieba, build BM25Okapi index.
3. `sparse_search()` — call `bm25_index.get_scores()` and return top-k.
4. `hybrid_search()` — implement RRF fusion combining dense and sparse results.
5. `rerank()` — lazy-load CrossEncoder, call `.predict()`, sort by score.

Testing order: test `dense_search` first (it's the simplest), then `sparse_search`, then `hybrid_search` (which depends on both). Test `rerank` independently.

Run: `pytest tests/test_rag.py::TestRetriever -v`

---

## Phase 3 — Memory (Week 2)

**Goal**: Implement both memory tiers. These are simple but must be done before agents so the agents can use history.

### Step 3.1: `memory/short_term.py`

Implement all methods. This is a thin wrapper over `collections.deque` — should take < 30 minutes.

Test manually:
```python
from memory import ShortTermMemory
mem = ShortTermMemory(max_turns=3)
mem.add_turn("user", "hello")
mem.add_turn("assistant", "hi")
print(mem.get_history())       # [{"role": "user", "content": "hello"}, ...]
print(mem.to_prompt_string())  # User: hello\nAssistant: hi
```

### Step 3.2: `memory/long_term.py`

Implement after short_term. Requires a running Redis instance (use Docker: `docker run -d -p 6379:6379 redis`).

Implementation notes:
- Decode bytes from Redis: `value.decode("utf-8") if isinstance(value, bytes) else value`.
- Use `json.dumps` / `json.loads` for structured data (order context, profile).
- Always set TTL on every write to prevent unbounded memory growth.

Test manually:
```python
import redis
from memory import LongTermMemory
r = redis.Redis()
mem = LongTermMemory(r, "user_test")
mem.save_preference("language", "zh-CN")
print(mem.get_preference("language"))  # zh-CN
```

---

## Phase 4 — Agents (Week 3)

**Goal**: Implement the four agent classes. Implement in reverse dependency order: `ResponseAgent` first (simplest, no dependencies on other agents), then up to `RouterAgent` (most complex).

### Step 4.1: `agents/response_agent.py`

The simplest agent — it just calls the LLM with a synthesis prompt.

Implementation steps:
1. Store `self.llm` and define `self.synthesis_prompt` (ChatPromptTemplate).
2. Implement `synthesize()`:
   - Format tool_results as bullet points.
   - Format rag_results as numbered passages.
   - Detect the empty-results case and return a fallback message without LLM call.
   - Call `self.llm.invoke(messages)` and return `.content`.
3. Implement `run()` as a thin wrapper over `synthesize()`.

Test with a mock LLM to verify state key updates.

### Step 4.2: `agents/faq_agent.py`

Implement after response_agent and after the retriever (Phase 2.4).

Implementation steps:
1. Store `self.llm`, `self.retriever`, and define `self.prompt_template`.
2. Implement `retrieve_context()` — call `retriever.hybrid_search()` then `retriever.rerank()`.
3. Implement `generate_answer()` — build prompt with numbered context, call LLM, handle empty context.
4. Implement `run()` — compose retrieve + generate, update state.

Critical: The generation prompt must explicitly say "Do not include any information not present in the provided context." This is the primary RAGAS faithfulness lever.

Test: `pytest tests/test_agents.py::TestFAQAgent -v`

### Step 4.3: `agents/order_agent.py`

The most complex agent due to the ReAct tool-calling loop.

Implementation steps:
1. `__init__`: bind tools with `llm.bind_tools(tools)`, build `tool_map` dict, store `max_iterations`.
2. `select_tool()`: implement the rule-based approach first (intent → default tool mapping), then optionally add LLM-based fallback later.
3. `execute_tool()`: look up tool in `tool_map`, call `.invoke(params)`, wrap exceptions in error dict.
4. `run()`: implement the ReAct loop. This is the critical path:
   - Build initial messages.
   - Loop until `ai_message.tool_calls` is empty or `max_iterations` reached.
   - For each tool call: extract name and args, call `execute_tool`, append `ToolMessage`.
   - Collect all tool results into `tool_results` list.
   - Return state update.

Test: `pytest tests/test_agents.py::TestOrderAgent -v`

### Step 4.4: `agents/router_agent.py`

Implement last among agents because it dispatches to all others.

Implementation steps:
1. `__init__`: store LLM. Optionally initialise `self.classifier = None`.
2. `rewrite_query()`: write the few-shot LLM prompt. Test with ambiguous queries like "那个订单" (pronoun resolution).
3. `classify_intent()`: start with LLM few-shot classification. The prompt should list all IntentType values with one example each and instruct the LLM to output ONLY the label.
4. `route()`: compose rewrite + classify, update state, return node name string.

Critical: The `route()` method must return a **string** (the next node name), not a dict. This is the LangGraph conditional routing convention.

Test: `pytest tests/test_agents.py::TestRouterAgent -v`

---

## Phase 5 — Graph (Week 3)

**Goal**: Wire all agents together into a runnable LangGraph StateGraph.

### Step 5.1: `graph/agent_graph.py`

Implement `build_graph()` and `run_graph()` after all four agents are implemented.

Implementation steps:
1. Implement `build_graph()` following the topology in the docstring.
2. Implement `run_graph()` as a thin wrapper around `compiled_graph.invoke()`.
3. Implement `init_graph()` to wire together settings, LLM, RAG components, and agents.

End-to-end test (the most important test in the project):
```python
from graph.agent_graph import init_graph, run_graph

# Requires: running vLLM server, running Milvus with loaded collection, running Redis
graph = init_graph()
answer = run_graph("我的订单 ORD-20240310-001 到哪里了？", user_id="user_123")
print(answer)
```

If this works, the core system is functional.

---

## Phase 6 — API & Queue (Week 4)

### Step 6.1: `queue/message_queue.py`

Implement `RedisQueue` (push, pop, peek, size). The BRPOP pattern is the only tricky part — test blocking behaviour manually before proceeding.

Skip `KafkaQueue` for now unless you need it; implement `KafkaQueue.push()` and `KafkaQueue.pop()` last.

### Step 6.2: `queue/worker.py`

Implement after `RedisQueue`.

Implementation steps:
1. `__init__`: initialise executor, stop event, futures list.
2. `process_message()`: call agent graph, measure latency, call result_callback.
3. `start()`: main BRPOP loop with ThreadPoolExecutor.submit().
4. `stop()`: set stop event, wait for futures, shutdown executor.
5. `start_in_background()`: wrap `start()` in a daemon Thread.

Test: spin up a worker, push a test message to Redis, verify it processes in a thread.

### Step 6.3: `api/main.py`

Implement after graph and queue are working.

Implementation steps:
1. Implement `lifespan()` context manager: `init_graph()`, create `RedisQueue`, optionally start `QueueWorker`.
2. Implement `get_compiled_graph()` dependency.
3. Implement `POST /chat` endpoint: call `run_graph()`, build `ChatResponse`.
4. Implement `GET /health` endpoint.
5. Implement `update_short_term_memory()` background task.

Test with FastAPI TestClient:
```python
from fastapi.testclient import TestClient
from api.main import app
client = TestClient(app)
resp = client.get("/health")
assert resp.status_code == 200
```

---

## Phase 7 — Evaluation (Week 4)

### Step 7.1: `evaluation/ragas_evaluator.py`

Implement after the FAQ agent is working (you need real RAG outputs to evaluate).

Implementation steps:
1. `__init__`: configure RAGAS metrics with the LLM judge.
2. `evaluate_dataset()`: build a HuggingFace Dataset, call `ragas.evaluate()`, parse results.
3. `generate_report()`: format as a markdown table.

Run a quick evaluation on 50 FAQ samples to validate the pipeline before scaling to 5 000.

### Step 7.2: `evaluation/offline_evaluator.py`

Implement after `ragas_evaluator` and `run_graph`.

Implementation steps:
1. `load_test_set()`: load and validate the JSONL test set.
2. `run_evaluation()`: implement the parallel inference loop with tqdm progress bar.
3. `compute_metrics()`: calculate intent accuracy, tool success rate, RAGAS composite.
4. `save_report()`: write JSON to `data/test_set/baseline_report.json`.

Run the baseline evaluation:
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

Expected baseline metrics before optimisation:
- Intent accuracy: ~82%
- Tool success rate: ~86%
- RAGAS composite: ~0.62

---

## Phase 8 — Optimisation (Week 5)

**Goal**: Improve key metrics using the baseline as a reference. Re-run the offline evaluator after each change to measure impact.

### 8.1 Retrieval optimisation

Target: Top-5 recall 71% → 88%.

Steps (in priority order):
1. **Add hybrid search**: if only dense search is running, enable BM25 and RRF fusion. This typically adds 10+ recall percentage points.
2. **Add reranking**: add CrossEncoder reranker after hybrid search. Typically adds 3–5 pp.
3. **Tune chunk size**: evaluate recall at chunk_size ∈ {256, 512, 768}. Smaller chunks improve precision; larger chunks improve recall for multi-fact questions.
4. **Tune overlap**: higher overlap reduces boundary artefacts.
5. **Add metadata filtering**: filter by intent category before ANN search to reduce noise.

### 8.2 Routing accuracy optimisation

Target: Intent accuracy 82% → 93%.

Steps:
1. **Improve few-shot examples**: add 2–3 representative examples per intent in the classification prompt.
2. **Fine-tune a classifier**: collect 500+ labelled examples per intent and fine-tune a `bert-base-chinese` classifier. Replace the LLM classification call with the fine-tuned model for 10× lower latency.
3. **Add query rewrite before classification**: verify that rewriting first improves classification (should, since queries become more explicit).

### 8.3 Tool calling optimisation

Target: Tool success rate 86% → 96%.

Steps:
1. **Improve tool docstrings**: LLMs use docstrings for tool selection. Make them more specific about when to use each tool.
2. **Add parameter extraction prompt**: for order ID extraction, add a dedicated extraction step before tool calling.
3. **Add error recovery**: when a tool call fails, retry with different parameters rather than failing immediately.

### 8.4 Throughput optimisation

Target: QPS 12 → 46.

Steps:
1. **Increase thread pool size**: start at `max_workers=10`, measure QPS with a load test, increase until LLM server is the bottleneck.
2. **LLM response caching**: cache identical (rewritten_query, intent) pairs for 1 hour in Redis. Common questions like "退款政策" hit the same answer every time.
3. **Batch embedding**: ensure the embedder is using full-batch encoding (not one-by-one) during RAG.
4. **vLLM continuous batching**: verify the LLM server has dynamic batching enabled (default in vLLM). Increase `--max-num-seqs` for higher batch sizes.

---

## Quick Reference: File → Phase Mapping

| File | Phase | Depends on |
|---|---|---|
| `config/settings.py` | 1 | — |
| `services/mock_order_service.py` | 1 | settings |
| `services/mock_logistics_service.py` | 1 | — |
| `tools/order_tools.py` | 1 | mock_order_service |
| `tools/logistics_tools.py` | 1 | mock_logistics_service |
| `tools/refund_tools.py` | 1 | mock_order_service |
| `tests/test_tools.py` | 1 | tools |
| `rag/chunker.py` | 2 | — |
| `rag/embedder.py` | 2 | — |
| `rag/knowledge_base.py` | 2 | chunker, embedder |
| `rag/retriever.py` | 2 | knowledge_base, embedder |
| `tests/test_rag.py` | 2 | rag/* |
| `memory/short_term.py` | 3 | — |
| `memory/long_term.py` | 3 | redis |
| `agents/response_agent.py` | 4 | LLM |
| `agents/faq_agent.py` | 4 | retriever, LLM |
| `agents/order_agent.py` | 4 | tools, LLM |
| `agents/router_agent.py` | 4 | LLM |
| `tests/test_agents.py` | 4 | agents |
| `graph/agent_graph.py` | 5 | all agents |
| `queue/message_queue.py` | 6 | redis |
| `queue/worker.py` | 6 | message_queue, graph |
| `api/main.py` | 6 | graph, queue |
| `evaluation/ragas_evaluator.py` | 7 | LLM, embeddings |
| `evaluation/offline_evaluator.py` | 7 | ragas_evaluator, graph |

---

## Development Environment Setup

```bash
# 1. Create virtual environment
python -m venv .venv && source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start Milvus (standalone, Docker)
docker run -d --name milvus \
  -p 19530:19530 -p 9091:9091 \
  milvusdb/milvus:v2.4.9 standalone

# 4. Start Redis
docker run -d --name redis -p 6379:6379 redis:7.2

# 5. Start vLLM (requires GPU and ~16GB VRAM for 7B model)
vllm serve Qwen/Qwen2.5-7B-Instruct \
  --host 0.0.0.0 --port 8000 \
  --quantization awq

# 6. Copy and edit config
cp .env.example .env
# Edit .env to set LLM_API_BASE, MILVUS_HOST, REDIS_HOST, etc.

# 7. Run tests
pytest tests/ -v

# 8. Start the API
uvicorn api.main:app --host 0.0.0.0 --port 8080 --reload
```
