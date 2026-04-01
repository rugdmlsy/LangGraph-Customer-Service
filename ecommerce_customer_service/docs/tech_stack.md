# Technology Stack — Multi-Agent E-commerce Customer Service System

This document explains why each technology was chosen over its alternatives, what its core concepts are, and which technical decisions are most important to understand before implementing.

---

## 1. LangGraph (Agent Framework)

### Why LangGraph instead of raw LangChain agents?

LangChain's original `AgentExecutor` is a black-box ReAct loop that is hard to customise, debug, and test. Specifically:

| Problem with AgentExecutor | LangGraph solution |
|---|---|
| Control flow is implicit (hidden inside the executor) | Explicit `StateGraph` with named nodes and edges |
| Difficult to add conditional routing between agents | First-class `add_conditional_edges()` with a routing function |
| Hard to unit-test individual steps | Each node is a plain Python callable, trivially mockable |
| No clean way to pause and resume (human-in-the-loop) | Built-in `interrupt_before` / `interrupt_after` support |
| State management is ad-hoc | Typed `AgentState` TypedDict enforces the data contract |

### Core concepts

**StateGraph**: The central class. You declare a state schema (TypedDict), add nodes (Python callables), and connect them with edges or conditional edges. The graph is then compiled into a `CompiledGraph` that manages state transitions.

```python
from langgraph.graph import StateGraph, START, END

graph = StateGraph(AgentState)
graph.add_node("router", router_agent.route)
graph.add_conditional_edges("router", routing_fn, {"faq": "faq_agent", ...})
compiled = graph.compile()
```

**Nodes**: Any callable with signature `(state: dict) -> dict`. The returned dict is merged (shallow) into the shared state. Nodes are stateless; all data lives in AgentState.

**Conditional edges**: A routing function reads the state and returns a string key. The `add_conditional_edges()` mapping converts that key to the next node name. This is how intent-based routing is implemented cleanly.

**Checkpointer**: Pass a `MemorySaver` (or Redis/Postgres checkpointer) to `compile()` to enable persistent multi-turn sessions. Each session is identified by a `thread_id` in the invocation config. This is the recommended way to handle stateful conversations.

**Streaming**: Use `graph.stream(initial_state)` instead of `.invoke()` to get incremental node outputs in real time — useful for streaming tokens to the frontend.

### Key technical decision

Set `AgentState(total=False)` so nodes only need to declare the keys they touch. This avoids boilerplate and makes it easy to add new state fields later without breaking existing nodes.

---

## 2. Milvus (Vector Database)

### Why Milvus instead of other vector databases?

| Alternative | Why not |
|---|---|
| FAISS | In-memory only, no persistence, no built-in distributed mode, no metadata filtering |
| Chroma | Good for prototyping, but limited horizontal scalability and no hybrid search |
| Weaviate | More complex deployment; Milvus has better performance benchmarks at >10M vectors |
| Qdrant | Strong alternative, but Milvus has better native sparse+dense hybrid search (Milvus 2.4+) |
| pgvector | Postgres extension, good for small datasets; Milvus is purpose-built for billion-scale ANN |

Milvus is **production-ready at scale** (Bilibili, Shopee, and other large e-commerce platforms use it), supports hybrid sparse+dense search natively, and has a mature Python SDK.

### Core concepts

**Collections and schemas**: The basic unit is a `Collection` (analogous to a SQL table). Each collection has a schema defining field names and types: primary key, scalar metadata fields (VARCHAR, INT64), and a FLOAT_VECTOR field for embeddings. The vector field dimension must match the embedding model output.

**Index types**:

- **HNSW** (Hierarchical Navigable Small World): graph-based index. Best recall at low latency. Parameters: `M` (edges per node, higher = better recall + more memory) and `efConstruction` (build quality). Use `ef` at query time to trade off speed vs. recall. Recommended: `M=16, efConstruction=200`.
- **IVF_FLAT**: inverted file index. Clusters vectors into `nlist` Voronoi cells. Faster to build than HNSW, lower memory, slightly worse recall. Use for datasets < 1M vectors or limited RAM.
- **IVF_SQ8**: quantised IVF. 4× memory reduction vs IVF_FLAT with ~1% recall degradation. Good for billion-scale.

**Hybrid search (Milvus 2.4+)**: Native support for combining dense vectors and sparse vectors (BM25 or SPLADE) in a single query. Use `AnnSearchRequest` for dense and `AnnSearchRequest` for sparse, then fuse with `RRFRanker` or `WeightedRanker`. This is more efficient than client-side RRF because it reduces network round-trips.

**Metadata filtering**: Milvus supports scalar field filters in the `expr` parameter of `collection.search()`. Example: `expr="category == 'logistics'"` to restrict search to a specific FAQ category before ANN. This dramatically reduces latency and improves precision for intent-specific retrieval.

**Partition keys**: Use partition keys to physically separate data by category. Set `partition_key_field="category"` at collection creation; Milvus automatically routes queries to the correct partition.

### Key technical decision

Use `collection.load()` after `create_index()` to load the index into memory. Without this call, `collection.search()` will fail. In production, use `utility.loading_progress()` to monitor load status before serving traffic.

---

## 3. SentenceTransformers (Embedding)

### Why SentenceTransformers?

- **Multilingual out of the box**: `paraphrase-multilingual-MiniLM-L12-v2` handles Chinese, English, and 50+ other languages in a single model. No language detection or separate models needed.
- **Simple API**: `model.encode(texts)` returns a numpy array. No tokenisation boilerplate.
- **Fine-tunable**: Domain-specific fine-tuning on query-passage pairs using `MultipleNegativesRankingLoss` can improve recall by 5–15% with just a few hundred labelled examples.
- **Lightweight**: MiniLM-L12-v2 is only 120 MB and encodes ~1000 sentences/second on CPU.

### Core concepts

**Pooling strategies**: SentenceTransformers pools token-level BERT outputs into a single sentence vector. Common strategies: mean pooling (default, best for semantic similarity), CLS token pooling, max pooling. The choice is baked into the model — do not change it without retraining.

**Normalised embeddings**: Setting `normalize_embeddings=True` in `model.encode()` makes cosine similarity equal to dot product, which is faster to compute. Always normalise when using Milvus's COSINE metric.

**Asymmetric models (BGE)**: Some models like `BAAI/bge-m3` are asymmetric — they use a different representation for queries vs. passages. For queries, prepend the instruction: `"Represent this sentence for searching relevant passages: "`. The model card specifies this. Failing to add the prefix reduces recall.

**Fine-tuning on domain data**:
1. Collect positive pairs: (customer question, relevant FAQ answer).
2. Use `MultipleNegativesRankingLoss` with in-batch negatives (no explicit negative mining needed).
3. Train for 1–3 epochs with a batch size of 64–128.
4. Evaluate with Recall@5 on a held-out set.
5. Export and load the fine-tuned model exactly like a base model.

### Key technical decision

Choose embedding dimension wisely: larger dim = better recall but more Milvus storage and slower search. MiniLM-L12-v2 at 384 dim is the sweet spot for this use case. BGE-M3 at 1024 dim gives ~5% better recall but quadruples storage cost.

---

## 4. Llama 3 / Qwen (LLM)

### Why open-source LLMs instead of GPT-4?

| Consideration | Open-source (Qwen/Llama) | GPT-4 API |
|---|---|---|
| Cost | Near-zero after hardware (inference cost amortised) | ~$30/1M tokens |
| Data privacy | All data stays on-premise | Customer queries sent to OpenAI |
| Latency | Controllable (local GPU) | Network-dependent (50–200 ms typical) |
| Customisation | Full control (LoRA, RLHF) | Prompt engineering only |
| Reliability | Self-managed (no vendor downtime risk) | Dependent on API availability |

For an e-commerce company handling millions of customer queries per day, the cost and privacy advantages of open-source models are decisive.

### Qwen vs. Llama 3

- **Qwen2.5-7B-Instruct**: Better Chinese language performance (Alibaba trained it heavily on Chinese data). Recommended for Chinese-dominant customer service.
- **Llama-3-8B-Instruct**: Better English performance, larger English pre-training corpus. Use for English or multilingual deployments.
- **Qwen2.5-72B**: Best quality, requires 4×A100 GPUs. Use for the most complex query synthesis.

### Core concepts

**Quantisation**: Reduce GPU memory footprint while preserving most quality.
- **GGUF + llama.cpp**: CPU inference with 4-bit quantisation. A 7B model fits in ~4 GB RAM. Use for development and low-traffic deployments.
- **AWQ (Activation-aware Weight Quantisation)**: 4-bit quantisation for GPU. Better quality than GGUF at same bit-width. `autoawq` library.
- **GPTQ**: Another 4-bit GPU quantisation scheme. Slightly slower than AWQ but more widely supported.

**vLLM serving**: The recommended production serving framework for LLMs. Key features:
- PagedAttention: efficient KV cache management, enables much higher batch sizes.
- Continuous batching: maximises GPU utilisation by dynamically batching incoming requests.
- OpenAI-compatible API: LangChain's `ChatOpenAI` points at `http://localhost:8000/v1` with no code changes.
- Start command: `vllm serve Qwen/Qwen2.5-7B-Instruct --host 0.0.0.0 --port 8000 --quantization awq`

**LoRA fine-tuning**: Adapt the base LLM to the e-commerce domain using Low-Rank Adaptation.
1. Construct a supervised fine-tuning (SFT) dataset of (customer query, ideal response) pairs from historical customer service logs (~50k examples).
2. Fine-tune using `trl` + `peft` with LoRA rank=16, alpha=32.
3. Training takes ~2 hours on a single A100 for a 7B model.
4. Merge LoRA weights into base model for single-file serving, or use dynamic adapter loading.

### Key technical decision

Use the OpenAI-compatible API pattern even for local models (vLLM, Ollama). This makes it trivial to swap between local and cloud LLMs by changing `LLM_API_BASE` in settings — no code changes required.

---

## 5. RAGAS (Evaluation)

### Why RAGAS instead of manual evaluation?

Manual evaluation of 5 000 RAG responses would require ~100 hours of human annotation at ~2 minutes per sample. RAGAS uses an LLM judge to automate this, reducing evaluation time to ~2 hours while maintaining good correlation with human judgements.

RAGAS is specifically designed for RAG systems and provides metrics that are not available in general NLP evaluation toolkits (BLEU, ROUGE, BERTScore).

### Core metrics explained

**Faithfulness** (primary metric for hallucination detection):
- Question: "Does the generated answer contain only claims that are supported by the retrieved context?"
- Computation: LLM extracts all factual claims from the answer, then checks each claim against the context. Faithfulness = (supported claims) / (total claims).
- Target: ≥ 0.85. Below 0.7 means the model is frequently fabricating information.
- How to improve: Strengthen the system prompt instruction to never go beyond the provided context. Reduce the LLM temperature.

**Answer Relevance** (response quality):
- Question: "Is the answer relevant to the user's question?"
- Computation: RAGAS generates N paraphrased questions from the answer, then computes cosine similarity between each generated question and the original. High similarity = the answer addresses the question.
- Target: ≥ 0.80. Low scores indicate off-topic or overly verbose answers.
- How to improve: Add a "be concise and directly answer the question" instruction to the generation prompt.

**Context Precision** (retrieval quality — no ground truth needed):
- Question: "Are the retrieved chunks actually useful for generating the answer?"
- Computation: For each retrieved chunk, LLM determines if it contributed to the answer. Precision = (useful chunks) / (total retrieved chunks).
- Target: ≥ 0.75. Low scores mean the retriever is returning noise.
- How to improve: Reduce `top_k` after reranking. Tune the reranker threshold. Add metadata filtering by intent category.

**Context Recall** (retrieval completeness — requires ground truth):
- Question: "Did the retrieved context cover all facts needed to answer correctly?"
- Computation: LLM checks each sentence in the ground truth answer against the retrieved context.
- Target: ≥ 0.80.

### Key technical decision

Use RAGAS in an **automated regression testing pipeline**: run it on a fixed 500-sample evaluation set every time the retrieval configuration changes (new chunk size, new embedding model, new top-k). Track metric trends over time to catch regressions early.

---

## 6. Redis (Message Queue & Long-Term Memory)

### Why Redis for message queue?

| Alternative | Why not for this use case |
|---|---|
| Kafka | Higher operational complexity, requires Zookeeper/KRaft; overkill for < 1 000 QPS |
| RabbitMQ | AMQP protocol, more complex client library, no native key-value store for memory |
| SQS (AWS) | Cloud-only, vendor lock-in, ~10ms additional latency per operation |
| In-process queue | Not durable, lost on crash, can't scale horizontally |

Redis provides both the message queue (via LIST data structure) and long-term user memory (via HASH and STRING with TTL) in a single service, simplifying the deployment.

### Core concepts

**LPUSH / BRPOP pattern** (message queue):
- Producer: `LPUSH queue_key json_message` — O(1), atomic, thread-safe.
- Consumer: `BRPOP queue_key 30` — blocks for up to 30 seconds waiting for a message. Atomically pops from the tail. No polling loop or sleep needed.
- Unlike a naive RPOP loop, BRPOP is push-based and uses almost no CPU when the queue is empty.

**HSET / HGET** (user preferences):
- Store multiple fields per user in a single Redis hash: `HSET user:123:preferences language zh-CN`.
- O(1) per field access regardless of how many preferences the user has.
- One `HGETALL` call fetches all preferences for prompt injection.

**SET with TTL** (order context, user profiles):
- `SET user:123:profile json_blob EX 2592000` (30 days).
- TTL prevents memory growth from inactive users — no cleanup job needed.

**Pub/Sub** (optional, for real-time result delivery):
- After the queue worker processes a request, publish the result: `PUBLISH result:request_id json_result`.
- The API layer subscribes with `SUBSCRIBE result:request_id` to deliver the answer via WebSocket or long-polling.
- More efficient than polling a result key.

### Key technical decision

Use `BRPOP` with a timeout (5–30 seconds) rather than a non-blocking `RPOP` in a sleep loop. BRPOP returns immediately when a message arrives and uses a server-side blocking wait otherwise — this eliminates latency from the `sleep()` interval while using no CPU.

---

## 7. FastAPI (API Framework)

### Why FastAPI instead of Flask or Django?

| Feature | FastAPI | Flask | Django |
|---|---|---|---|
| Async support | Native (asyncio) | Requires extensions | Limited |
| Request validation | Pydantic (automatic) | Manual | Django REST Framework |
| Auto API docs | OpenAPI / Swagger UI | Plugin required | DRF Browsable API |
| Performance | Among the fastest Python frameworks | Slower (WSGI) | Slower (WSGI) |
| Type safety | Full (Pydantic models) | None | Partial |

For an AI API service where request/response schemas are complex (nested dicts with optional fields), Pydantic's automatic validation and clear error messages are a major productivity win.

### Core concepts

**Dependency injection** (`Depends`): Declare shared resources (compiled graph, message queue, Redis client) as FastAPI dependencies. FastAPI instantiates them once and injects them into each request handler. This replaces global singletons with testable, injected dependencies.

```python
def get_graph() -> CompiledGraph:
    return app.state.graph

@app.post("/chat")
async def chat(req: ChatRequest, graph: CompiledGraph = Depends(get_graph)):
    ...
```

**Background tasks**: Use `BackgroundTasks.add_task()` for operations that don't need to complete before the response is sent (e.g. saving conversation turns to memory, logging to analytics). This keeps response latency low.

**Lifespan context manager**: Use `@asynccontextmanager` lifespan (FastAPI 0.95+) instead of deprecated `@app.on_event("startup")`. The lifespan function initialises resources on startup and releases them on shutdown (clean teardown of Milvus connections, thread pools, etc.).

**Middleware**: Add `CORSMiddleware` for browser clients, `GZipMiddleware` for response compression, and custom middleware for request ID injection and latency logging.

### Key technical decision

Run Uvicorn with `workers=1` when the in-process queue worker thread is active. Uvicorn workers are separate processes that do not share state; if you run multiple workers, each would start its own queue worker thread competing on the same Redis queue. Instead, use a separate queue worker process and scale the API and worker layers independently.
