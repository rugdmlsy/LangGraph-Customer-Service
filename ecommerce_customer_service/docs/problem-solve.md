1. 一开始选用bge-large作为embedding模型，但是由于corpus大多是短文本，语义密度低，不需要高维，故改成BAAI/bge-small-zh-v1.5
2. 通过gRPC向milvus请求插入数据的时候没有做批处理（batching），导致 gRPC payload 超 64MB，后来改成batch size为512
3. insert() 里每批都调一次 collection.flush()。collection.flush() 是同步磁盘 sync，一次约 10s，之前每 1000 条 chunk 就触发一次，完全掩盖了 GPU 速度。现在改为所有批次写完后只 flush 一次。之前把flush写在 `KnowledgeBase.insert()` 里，后来改成在 `KnowledgeBase.load_from_file()` 中统一 flush。
4. RecursiveCharacterTextSplitter 偶尔会产生略超过 CHUNK_SIZE 的 chunk（4102 > 4096），而 Milvus schema 的 `max_length=4096` 卡住了。两处修：schema 调大 + insert 前截断兜底。`RecursiveCharacterTextSplitter` 优先按 separator 分割，超长的 fallback 保留
5. 重构了代码，用 milvus 原生 sparse 检索代替客户端 rank_bm25  
  1. insert() 中用 BM25 per-batch 的分数不是真正的稀疏向量格式
  2. sparse_search() 和 hybrid_search() 用客户端 rank_bm25，没有用 Milvus 原生 sparse 检索 
  3. dense_search() 有 results[0] 多余索引 bug 
  重构方案：
  - Embedder 支持 BGE-M3 产出 dense+sparse（embed_hybrid/embed_query_sparse） 
  - insert() 接受 sparse_vectors 参数
  - load_from_file() 改用 embed_hybrid()
  - Retriever 用 Milvus AnnSearchRequest + hybrid_search() + RRFRanker   
6. 修复了 order_agent.py：工具调用后必须追加 ToolMessage(tool_call_id=...) 而非 plain tuple，否则 LangChain 会报错。
7. 从整个项目来看，我认为技术含量最高的部分和 debug 中印象最深的分别是：

---

**最有挑战的部分：BGE-M3 混合检索架构**

RAG 管道里用 BGE-M3 同时生成 dense + sparse 双向量，再通过 Milvus 的 `AnnSearchRequest` + `RRFRanker` 做混合检索，最后接 reranker 二次排序——这条链路的每一环都要对齐（向量维度、字段名、评分归一化），任何一处脱节都是静默失败。纯稠密检索很常见，但把稀疏（BM25-like）和稠密融合起来，再加 rerank，是工程上真正有深度的设计。

---

**debug 中印象最深：LangGraph 节点返回值的语义**

这个 bug 最隐蔽。`RouterAgent.route()` 返回了一个路由字符串（`"faq"` / `"order"`），看起来完全合理——毕竟函数名就叫 route。但 LangGraph 的 `StateGraph` 要求每个节点返回**状态更新字典**，路由逻辑是通过 `add_conditional_edges` 里独立传入的路由函数来表达的，两者职责完全分离。

错误信息是 `"Expected dict, got order_agent"`，指向的是后续节点，完全不暴露真正的根因。这类"概念边界"类 bug——不是语法错、不是 API 用法错，而是对框架设计意图的误解——是最消耗排查时间的一类。

---

**另一个值得一提的：pydantic-settings 的 `env_file` 路径陷阱**

`env_file=".env"` 相对于进程的 CWD 解析，不是相对于 `settings.py` 所在位置。FastAPI 从项目根目录之外启动时，`.env` 静默加载失败，所有配置回落默认值，下游 LLM 调用直接 401。这类"配置没加载但没有任何报错"的问题，在分布式/多服务环境里会非常难定位。


**项目名称：电商智能客服多智能体系统**

---

**【Situation】**
电商场景下客服问题覆盖面广：既有大量重复性产品咨询（FAQ），又有需要实时访问后端数据的订单/物流/退款查询，单一模型无法同时兼顾检索质量与工具调用能力。

**【Task】**
设计并实现一套生产可用的多智能体客服系统，支持意图路由、知识库问答与事务性操作，并建立可量化的 RAG 质量评估体系。

**【Action】**

- 基于 **LangGraph StateGraph** 构建多智能体编排框架，实现 Router → FAQ/Order → Response 的条件路由流水线，各节点通过状态字典解耦，支持并行扩展
- 设计 **BGE-M3 混合检索管道**：dense + sparse 双向量并行召回，经 Milvus `RRFRanker` 融合排序后接 reranker 二次精排，显著提升长尾问题的检索覆盖率
- 实现基于 **LangChain ReAct 循环**的 OrderAgent，通过 `bind_tools` + `ToolMessage` 协议支持多轮工具调用（订单查询、物流追踪、退款申请），最大迭代次数可配置
- 使用 **FastAPI lifespan** 管理图编译与向量库连接的单例生命周期，避免重复初始化；通过 pydantic-settings 统一管理多环境配置
- 基于 **RAGAS 0.4.x** 搭建自动化评估脚本，覆盖 Faithfulness / AnswerRelevancy / ContextPrecision 三项指标，支持批量采样与 JSON 报告导出

**【Result】**

- 在 20 条扫地机器人领域测试集上：**Faithfulness 0.929**（阈值 0.85 ✓）、**Context Precision 0.760**（阈值 0.75 ✓），系统综合评分 0.753
- 订单/退款工具调用链路端到端联调通过，覆盖 pending / shipped / delivered / refund 全状态流转
- 评估脚本可一键复现，支持 `--n` 采样数与 `--test-file` 路径参数，具备持续集成接入能力