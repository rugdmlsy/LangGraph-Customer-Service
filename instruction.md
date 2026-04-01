仔细阅读下面的文件，请为我完成这个项目架构的搭建，包括：
1. 为我生成项目的骨干目录结构，但先不要生成具体代码
2. 为每个文件写明注释，说明这个文件是干什么的
3. 写出每个文件中核心函数和类的声明，用TODO取代具体实现，并在注释中写出其作用即推荐实现方法
4. 对于项目中用到的每个技术栈，说明为何选用它和其核心技术点，保存在一个单独的文件中
5. 对于如何实现该项目，如实现文件、方法的顺序等，为我生成一个实现指南，保存在单独的文件中


**工业界简历项目的几个核心要求**：

1. **有真实业务场景（背景）**
2. **有完整 pipeline（数据 → 模型 → 系统 → 部署）**
3. **技术点要深入（不是只写用了什么）**
4. **有可量化指标（提升多少）**
5. **有工程架构（agent / memory / queue / 并发）**
6. **有评测体系（RAGAS、离线评测等）**

同时结合现在 **LLM岗位趋势**（Agent + RAG + 多模态 + 工程能力），我最推荐你做一个 **“多 Agent 电商客服自动化系统”**。这个项目既 **贴近真实业务**，又能体现 **Agent / RAG / 工程能力 / 数据构建 / 评估体系**。

---

# 推荐项目

## 多 Agent 电商客服自动化系统（Agent + RAG + Tool Use）

### 1 项目背景（Business Scenario）

电商平台客服每天需要处理：

* 商品咨询
* 订单查询
* 售后退款
* 物流问题

传统客服系统：

* FAQ命中率低
* 人工成本高
* 查询多个系统效率低

因此设计 **多 Agent 客服系统**：

* 自动理解用户问题
* 调用内部系统工具
* 使用知识库回答问题
* 自动完成简单售后流程

---

# 项目架构

### Multi-Agent Architecture

```
                User Query
                     │
                     ▼
            Intent Router Agent
         (意图识别 / query rewrite)
              │           │
              │           │
      FAQ Agent       Order Agent
        (RAG)         (Tool Call)
              │           │
              ▼           ▼
       Knowledge DB     Order API
        (Milvus)        (Mock Service)

              │
              ▼
         Response Agent
        (answer synthesis)

              │
              ▼
           Memory
    (conversation history)
```

---

# 技术栈

Agent Framework

* LangGraph

向量数据库

* Milvus

Embedding

* SentenceTransformers

LLM

* Llama 3
  或
* Qwen

评测

* RAGAS

---

# 核心技术点（简历最关键）

## 1 Query Rewrite + Intent Routing

解决问题：

用户问题表达不规范

例：

> “快递怎么还没到？”

改写为

```
物流状态查询 + 订单号
```

方法：

* LLM Query Rewrite
* 分类模型做 Intent Detection

效果指标：

intent accuracy
从 **82% → 93%**

---

# 2 RAG系统优化

知识库构建：

数据来源

* 商品 FAQ
* 客服历史对话
* 商品说明

数据处理：

* 文本清洗
* chunking

分片策略：

```
sliding window chunking
chunk size = 512
overlap = 100
```

向量存储：

Milvus

检索策略：

* hybrid search
* reranker

指标：

Top-5 Recall

```
baseline: 71%
optimized: 88%
```

---

# 3 Tool Calling Agent

Agent调用系统 API：

例如：

```
get_order_status(order_id)
create_refund(order_id)
query_logistics(order_id)
```

Agent流程：

```
User → intent → tool call → result → answer
```

成功率指标：

Tool execution success rate

```
86% → 96%
```

---

# 4 Memory系统

实现：

```
short term memory
long term memory
```

存储内容：

* 用户历史订单
* 用户偏好
* 上下文对话

提升：

multi-turn success rate

```
68% → 87%
```

---

# 5 评估体系

自动评测

使用：

* RAGAS

指标：

```
faithfulness
answer relevance
context precision
```

Offline Evaluation：

测试集：

```
5000客服真实问题
```

结果：

```
RAGAS score: 0.62 → 0.79
```

---

# 6 工程架构（很多人简历没有）

并发处理：

```
ThreadPoolExecutor
```

消息队列：

```
Redis Queue / Kafka
```

解决问题：

* agent超时
* tool调用阻塞

吞吐量提升：

```
QPS: 12 → 46
```

---

# 简历写法示例（工业风）

**Multi-Agent E-commerce Customer Service System**

* Designed a **multi-agent LLM system** for e-commerce customer service using LangGraph, supporting FAQ QA, order queries, and refund processing.
* Built a **RAG pipeline** with Milvus, implementing query rewriting, hybrid retrieval, and reranking, improving Top-5 retrieval recall from **71% to 88%**.
* Constructed a **domain-specific dataset (50k QA pairs)** from customer service logs and product manuals for SFT fine-tuning.
* Implemented **tool-calling agents** integrating order and logistics APIs, increasing tool execution success rate from **86% to 96%**.
* Developed a **conversation memory module** enabling multi-turn context tracking, improving task completion rate from **68% to 87%**.
* Built an automated evaluation pipeline using RAGAS, increasing RAG score from **0.62 to 0.79**.
* Optimized system throughput using message queues and thread pools, improving QPS from **12 to 46**.

---

# 为什么这个项目特别适合写简历

它同时覆盖了评论里提到的所有点：

| 评论要求      | 这个项目               |
| --------- | ------------------ |
| 有业务背景     | 电商客服               |
| 数据构建      | 客服对话数据             |
| RAG       | 有                  |
| Agent     | 有                  |
| memory    | 有                  |
| tool call | 有                  |
| 评估        | RAGAS              |
| 工程能力      | queue / threadpool |
| 可量化指标     | recall / qps       |

**面试基本问不倒。**

---

# 如果想做得更强（更容易拿面试）

可以再加两个升级：

### 1 多模态客服

用户上传商品图片：

Agent调用 **视觉模型识别商品问题**

（破损 / 款式）

---

### 2 自动客服流程Agent

例如：

```
退货流程
```

Agent自动执行

```
1 判断是否符合退款条件
2 创建退款单
3 通知仓库
```

