#!/usr/bin/env python3
"""
scripts/evaluate_rag.py

RAGAS evaluation of the RAG pipeline using test questions from data/test/.

Steps:
    1. Parse Q&A pairs from the test file.
    2. Run retrieval + generation for each sampled question.
    3. Evaluate with RAGAS (Faithfulness, AnswerRelevancy, ContextPrecision).
    4. Print a markdown report and save results to evaluation/ragas_report.json.

Usage:
    cd ecommerce_customer_service
    python scripts/evaluate_rag.py [--n 20] [--output evaluation/ragas_report.json]
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import sys
import os
import time
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("evaluate_rag")

# ─────────────────────────────────────────────────────────────────────────────
# Test-file parser
# ─────────────────────────────────────────────────────────────────────────────

def parse_qa_file(path: Path) -> list[dict]:
    """
    Parse Q&A pairs from the structured text file.

    Expected format (one block per question):
        N. **question text**
        - answer line 1
        - answer line 2  (optional continuation)
    """
    text = path.read_text(encoding="utf-8")
    qa_pairs: list[dict] = []

    # Split on numbered items: "N. **..."
    blocks = re.split(r"\n(?=\d+\.\s+\*\*)", text)
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        # Extract question
        q_match = re.match(r"\d+\.\s+\*\*(.+?)\*\*", block)
        if not q_match:
            continue
        question = q_match.group(1).strip()
        # Extract answer lines (lines starting with "- ")
        answer_lines = re.findall(r"^-\s+(.+)", block, re.MULTILINE)
        if not answer_lines:
            continue
        ground_truth = " ".join(answer_lines).strip()
        qa_pairs.append({"question": question, "ground_truth": ground_truth})

    logger.info("Parsed %d Q&A pairs from %s", len(qa_pairs), path.name)
    return qa_pairs


# ─────────────────────────────────────────────────────────────────────────────
# RAG runner (retrieval + generation)
# ─────────────────────────────────────────────────────────────────────────────

def run_rag(question: str, faq_agent, retriever, settings) -> dict:
    """Run retrieval + generation for a single question."""
    try:
        docs = retriever.hybrid_search(question, top_k=settings.RETRIEVAL_TOP_K)
        reranked = retriever.rerank(question, docs, top_n=settings.RERANK_TOP_N)
        answer = faq_agent.generate_answer(question, reranked, history=[])
        context_texts = [d["text"] for d in reranked]
        return {"answer": answer, "contexts": context_texts, "error": None}
    except Exception as e:
        logger.warning("RAG failed for '%s': %s", question[:40], e)
        return {"answer": "", "contexts": [], "error": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="RAGAS evaluation for the RAG pipeline")
    parser.add_argument("--n",      type=int,  default=20,
                        help="Number of questions to evaluate (default: 20)")
    parser.add_argument("--seed",   type=int,  default=42,
                        help="Random seed for sampling questions")
    parser.add_argument("--output", type=str,  default="evaluation/ragas_report.json",
                        help="Path to save the JSON report")
    parser.add_argument("--test-file", type=str,
                        default="data/test/扫地机器人100问2.txt",
                        help="Path to test Q&A file")
    args = parser.parse_args()

    # ── 1. Load settings & init components ──────────────────────────────────
    logger.info("Loading settings and initialising RAG components...")
    from config import settings
    from rag import Embedder, KnowledgeBase, SlidingWindowChunker, Retriever
    from agents.faq_agent import FAQAgent
    from langchain_openai import ChatOpenAI
    from pymilvus import Collection

    llm = ChatOpenAI(
        base_url=settings.LLM_API_BASE,
        api_key=settings.LLM_API_KEY,
        model=settings.LLM_MODEL_NAME,
        temperature=settings.LLM_TEMPERATURE,
        max_completion_tokens=settings.LLM_MAX_TOKENS,
        extra_body={"enable_thinking": False},
        timeout=600,  # 10分钟超时
        # max_retries=3,
    )
    embedder  = Embedder(settings.EMBEDDING_MODEL_NAME)
    chunker   = SlidingWindowChunker(settings.CHUNK_SIZE, settings.CHUNK_OVERLAP)
    kb        = KnowledgeBase(
        embedder, chunker,
        settings.MILVUS_HOST, settings.MILVUS_PORT,
        collection_name=settings.MILVUS_COLLECTION_NAME,
    )
    kb.connect()
    kb.create_collection(settings.MILVUS_COLLECTION_NAME, settings.EMBEDDING_DIM)
    retriever = Retriever(kb, embedder, settings.RERANKER_MODEL_NAME)
    faq_agent = FAQAgent(llm, retriever)

    # ── 2. Parse & sample test questions ────────────────────────────────────
    test_path = Path(args.test_file)
    qa_pairs  = parse_qa_file(test_path)
    if not qa_pairs:
        logger.error("No Q&A pairs parsed from %s", test_path)
        sys.exit(1)

    random.seed(args.seed)
    n = min(args.n, len(qa_pairs))
    sampled = random.sample(qa_pairs, n)
    logger.info("Sampled %d / %d questions for evaluation", n, len(qa_pairs))

    # ── 3. Run RAG on each question ──────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"  Running RAG inference on {n} questions...")
    print(f"{'─'*60}")

    questions, answers, contexts, ground_truths = [], [], [], []
    failed = 0
    for i, item in enumerate(sampled, 1):
        q = item["question"]
        gt = item["ground_truth"]
        t0 = time.monotonic()
        result = run_rag(q, faq_agent, retriever, settings)
        elapsed = (time.monotonic() - t0) * 1000
        status = "✗" if result["error"] else "✓"
        print(f"  [{i:>2}/{n}] {status}  {q[:50]:<50}  {elapsed:>6.0f}ms")
        if result["error"]:
            failed += 1
            continue
        questions.append(q)
        answers.append(result["answer"])
        contexts.append(result["contexts"])
        ground_truths.append(gt)

    if not questions:
        logger.error("All RAG calls failed — nothing to evaluate.")
        sys.exit(1)

    logger.info("RAG inference done: %d succeeded, %d failed", len(questions), failed)

    # ── 4. Build RAGAS metrics with our LLM ─────────────────────────────────
    logger.info("Setting up RAGAS evaluator...")
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.metrics._faithfulness import Faithfulness
    from ragas.metrics._answer_relevance import AnswerRelevancy
    from ragas.metrics._context_precision import ContextPrecision
    from ragas import evaluate
    from datasets import Dataset
    from langchain_community.embeddings import HuggingFaceEmbeddings

    ragas_llm = LangchainLLMWrapper(llm)
    # Use a lightweight multilingual model for semantic similarity in AnswerRelevancy
    hf_emb = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={"device": "cuda"},
    )
    ragas_emb = LangchainEmbeddingsWrapper(hf_emb)

    faithfulness = Faithfulness()
    faithfulness.llm = ragas_llm

    answer_relevancy = AnswerRelevancy(embeddings=ragas_emb)
    answer_relevancy.llm = ragas_llm

    context_precision = ContextPrecision()
    context_precision.llm = ragas_llm

    metrics = [faithfulness, answer_relevancy, context_precision]

    # ── 5. Evaluate ─────────────────────────────────────────────────────────
    dataset = Dataset.from_dict({
        "question":     questions,
        "answer":       answers,
        "contexts":     contexts,
        "ground_truth": ground_truths,
    })

    print(f"\n{'─'*60}")
    print(f"  Running RAGAS evaluation on {len(questions)} samples...")
    print(f"{'─'*60}\n")

    t0 = time.monotonic()
    result = evaluate(
        dataset,
        metrics=metrics,
        raise_exceptions=False,
        show_progress=True,
    )
    eval_elapsed = time.monotonic() - t0

    df = result.to_pandas()

    # ── 6. Compute summary scores ────────────────────────────────────────────
    _base_cols = {"question", "answer", "contexts", "ground_truth"}
    metric_cols = [c for c in df.select_dtypes(include="number").columns
                   if c not in _base_cols]
    scores = {col: float(df[col].mean(skipna=True)) for col in metric_cols}
    scores["composite_score"] = sum(scores.values()) / max(len(scores), 1)

    thresholds = {
        "faithfulness":      0.85,
        "answer_relevancy":  0.80,
        "context_precision": 0.75,
    }

    # ── 7. Print report ──────────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print(f"  RAGAS Evaluation Report  ({datetime.now().strftime('%Y-%m-%d %H:%M')})")
    print(f"{'═'*60}")
    print(f"  Test file : {test_path.name}")
    print(f"  Samples   : {len(questions)}  (failed: {failed})")
    print(f"  Eval time : {eval_elapsed:.0f}s")
    print(f"{'─'*60}")
    print(f"  {'Metric':<25} {'Score':>7}  {'Threshold':>9}  {'Status'}")
    print(f"  {'─'*25} {'─'*7}  {'─'*9}  {'─'*6}")
    for metric, score in scores.items():
        if metric == "composite_score":
            continue
        threshold = thresholds.get(metric, 0.75)
        status = "PASS ✓" if score >= threshold else "FAIL ✗"
        print(f"  {metric:<25} {score:>7.3f}  {threshold:>9.2f}  {status}")
    print(f"{'─'*60}")
    print(f"  {'composite_score':<25} {scores['composite_score']:>7.3f}")
    print(f"{'═'*60}\n")

    # Per-sample breakdown (df may not include input cols in ragas 0.4.x old-style)
    print("  Per-sample detail:")
    print(f"  {'#':>3}  {'Question':<45}  " +
          "  ".join(f"{c[:8]:>8}" for c in metric_cols))
    print(f"  {'─'*3}  {'─'*45}  " + "  ".join(["─"*8]*len(metric_cols)))
    for i, (_, row) in enumerate(df.iterrows()):
        q_short = questions[i][:44] if i < len(questions) else ""
        vals = "  ".join(
            f"{float(row[c]):>8.3f}" if not __import__("math").isnan(float(row[c])) else f"{'nan':>8}"
            for c in metric_cols
        )
        print(f"  {i+1:>3}. {q_short:<45}  {vals}")
    print()

    # ── 8. Save JSON report ──────────────────────────────────────────────────
    import math
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    report = {
        "timestamp":     datetime.now().isoformat(),
        "test_file":     str(test_path),
        "n_samples":     len(questions),
        "n_failed":      failed,
        "eval_time_sec": round(eval_elapsed, 1),
        "scores":        {k: round(v, 4) for k, v in scores.items()},
        "per_sample":    [
            {
                "question": questions[i] if i < len(questions) else "",
                "answer":   answers[i]   if i < len(answers)   else "",
                **{c: (round(float(row[c]), 4) if not math.isnan(float(row[c])) else None)
                   for c in metric_cols},
            }
            for i, (_, row) in enumerate(df.iterrows())
        ],
    }
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    logger.info("Report saved to %s", output_path)
    print(f"  Report saved → {output_path}\n")


if __name__ == "__main__":
    main()
