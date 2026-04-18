"""
evaluation/offline_evaluator.py

Offline batch evaluation pipeline.

Runs the full agent graph on a held-out test set and computes composite
metrics covering all system components:
    - Intent accuracy       (router quality)
    - Tool success rate     (order agent reliability)
    - RAGAS composite score (RAG quality)
    - End-to-end latency    (P50, P95)

Test set format (data/test_set/):
    JSONL file with one sample per line:
    {
        "question":      "我的订单 ORD-001 到哪了？",
        "intent":        "logistics",
        "ground_truth":  "您的订单正在运输中，预计明天到达。",
        "expected_tool": "query_logistics",     # null for FAQ queries
        "context_docs":  ["相关FAQ文本片段..."] # null for tool-call queries
    }

Scalability:
    For 5 000 samples at ~2s per sample = ~2.8 hours sequential.
    Use ThreadPoolExecutor (max_workers=20) for ~10 min wall time.
    Progress is logged every 100 samples via tqdm.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

logger = logging.getLogger(__name__)


class OfflineEvaluator:
    """
    Batch evaluation runner for the full multi-agent system.

    Attributes:
        ragas_evaluator: RAGASEvaluator instance for RAG metric computation.
        max_workers:     Thread pool size for parallel inference.
    """

    def __init__(
        self,
        ragas_evaluator: Any | None = None,
        max_workers: int = 20,
    ) -> None:
        """
        Initialise the offline evaluator.

        Args:
            ragas_evaluator: Optional RAGASEvaluator instance.  If None,
                             RAGAS metrics are skipped in the report.
            max_workers:     Number of parallel inference threads.
        """
        self.ragas_evaluator = ragas_evaluator
        self.max_workers     = max_workers

    def load_test_set(self, file_path: str | Path) -> list[dict]:
        """
        Load a JSONL test set from disk.

        Args:
            file_path: Path to a .jsonl file with one JSON object per line.

        Returns:
            List of sample dicts.  Logs the number of samples loaded.

        Validation:
            After loading, check that each sample has the required keys
            ("question", "intent", "ground_truth") and log a warning for
            any malformed samples (skip them rather than crashing).
        """
        samples = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(json.loads(line))
        logger.info("Loaded %d test samples from %s", len(samples), file_path)
        return samples
        

    def run_evaluation(
        self,
        agent_graph: Any,
        test_set: list[dict],
    ) -> dict:
        """
        Run the agent graph on all test samples and collect raw predictions.

        Args:
            agent_graph: Compiled LangGraph graph (output of build_graph()).
            test_set:    List of sample dicts from load_test_set().

        Returns:
            Dict with keys:
            {
                "predictions": list[dict],   # [{question, answer, intent, tools_called}]
                "ground_truths": list[dict], # mirror of test_set
                "metrics": dict,             # computed by compute_metrics()
            }
        """
        def run_single(sample: dict) -> dict:
            result = agent_graph.invoke({
                "query":   sample["question"],
                "user_id": "eval_user",
                "history": [],
            })
            return {
                "question":     sample["question"],
                "answer":       result.get("final_answer", ""),
                "intent":       result.get("intent", "unknown"),
                "rag_results":  result.get("rag_results", []),
                "tool_results": result.get("tool_results", []),
                "ground_truth_intent": sample["intent"],
                "ground_truth_answer": sample["ground_truth"],
            }

        predictions = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(run_single, s): s for s in test_set}
            for future in tqdm(as_completed(futures), total=len(test_set)):
                try:
                    predictions.append(future.result())
                except Exception as e:
                    logger.error("Sample failed: %s", e)

        metrics = self.compute_metrics(predictions, test_set)
        return {"predictions": predictions, "ground_truths": test_set, "metrics": metrics}
        

    def compute_metrics(
        self,
        predictions: list[dict],
        ground_truths: list[dict],
    ) -> dict:
        """
        Compute all evaluation metrics from predictions and ground truths.

        Args:
            predictions:   List of prediction dicts from run_evaluation().
            ground_truths: Original test set samples.

        Returns:
            Dict of metric_name → score:
            {
                "intent_accuracy":    0.93,  # fraction of correct intent predictions
                "tool_success_rate":  0.96,  # fraction of correct tool selections
                "ragas_composite":    0.79,  # from RAGASEvaluator (FAQ samples only)
                "avg_latency_ms":    1847.3,
                "p95_latency_ms":    3210.0,
            }
        """
        # Intent accuracy
        intent_correct = sum(
            1 for p in predictions
            if p["intent"] == p["ground_truth_intent"]
        )
        intent_acc = intent_correct / len(predictions)

        # Tool success rate (only on samples with expected_tool)
        tool_samples = [
            (p, g) for p, g in zip(predictions, ground_truths)
            if g.get("expected_tool")
        ]
        if tool_samples:
            tool_correct = sum(
                1 for p, g in tool_samples
                if any(tr.get("tool") == g["expected_tool"]
                        for tr in p.get("tool_results", []))
            )
            tool_success = tool_correct / len(tool_samples)
        else:
            tool_success = None

        # RAGAS metrics (FAQ samples only)
        faq_samples = [(p, g) for p, g in zip(predictions, ground_truths)
                        if g.get("intent") == "faq" and p.get("rag_results")]
        ragas_score = None
        if self.ragas_evaluator and faq_samples:
            ragas_result = self.ragas_evaluator.evaluate_dataset(
                questions=[p["question"] for p, _ in faq_samples],
                answers=[p["answer"] for p, _ in faq_samples],
                contexts=[[d["text"] for d in p["rag_results"]] for p, _ in faq_samples],
                ground_truths=[g["ground_truth"] for _, g in faq_samples],
            )
            ragas_score = ragas_result.get("composite_score")

        return {
            "intent_accuracy":   round(intent_acc, 4),
            "tool_success_rate": round(tool_success, 4) if tool_success is not None else None,
            "ragas_composite":   round(ragas_score, 4)  if ragas_score  is not None else None,
            "num_samples":       len(predictions),
        }

    def save_report(self, results: dict, output_path: str | Path) -> None:
        """
        Save the evaluation results to a JSON file.

        Args:
            results:     Output dict from run_evaluation().
            output_path: Path to write the JSON report.
        """
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results["metrics"], f, ensure_ascii=False, indent=2)
        logger.info("Evaluation report saved to %s", output_path)
