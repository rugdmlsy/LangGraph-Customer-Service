- [ ] 对比语义分块和token分块性能差异
- [ ] 对比重排方法性能差异（RRF	RankLLM	Cross-Encoder	ColBERT）
- [ ] 压缩以存储长上下文而不是简单的设置ttl
- [ ] 用标注的错误回答改进rag

✅ 用法 A：负样本（Negative Sampling）

用于优化检索器：

构造 (query, 正确文档, 错误文档) 三元组
用于：
reranker 训练（cross-encoder）
embedding 对比学习（contrastive learning）

👉 作用：

让模型学会“这类内容虽然相似，但不该被选”

✅ 用法 B：Guardrail / 拒答机制

把错误问答存入一个**“黑名单知识库”**：

在生成前做一层检测：
如果 query 与某个已知错误 case 高相似
→ 触发：
拒答
或提示“不可靠问题类型”

👉 这本质是一个 failure pattern detector

✅ 用法 C：训练修复 Agent（你现在做的方向）

你之前提到你在做 multi-agent crash recovery，这里可以复用：

错误问答 → 构造成：

{
  "question": "...",
  "wrong_answer": "...",
  "error_type": "...",
  "correct_answer": "...",
  "analysis": "..."
}

用于：

训练一个 diagnostic agent
或作为 self-reflection / critique prompt 的 few-shot

👉 作用：

提升模型“识别错误 + 修复”的能力，而不是传播错误

3️⃣ 如果你“必须”放进 RAG（有些场景确实需要）

比如你要做：

FAQ纠错系统
hallucination detection
安全审计系统

那可以放，但必须加强约束：

🔒 策略 1：显式标签
[INCORRECT EXAMPLE]
Question: ...
Answer: ...
Reason: ...
🔒 策略 2：检索隔离（关键）
正确知识库（primary KB）
错误样本库（error KB）

检索流程变成：

query
 ├─ retrieve(correct KB)
 └─ retrieve(error KB) → 仅用于校验，不参与生成
🔒 策略 3：Rerank / Filtering
错误样本永远不进入 top-k context
只用于：
scoring penalty
或置信度计算
4️⃣ 一个工程上更成熟的架构（推荐）

结合你现在在做的 multi-agent，可以这样设计：

           ┌──────────────┐
           │   Retriever   │
           └──────┬───────┘
                  ↓
        ┌──────────────────┐
        │  Candidate Docs   │
        └──────┬───────────┘
               ↓
     ┌──────────────────────┐
     │  Verifier Agent       │  ← 使用错误问答作为负例知识
     └──────┬──────────────┘
            ↓
     ┌──────────────────────┐
     │  Generator Agent      │
     └──────────────────────┘

👉 错误问答只进入 Verifier / Critic 层，不污染 Retriever。

- [ ] 换用更小的模型进行route、query rewrite等操作
- [ ] 用子图循环优化text2sql