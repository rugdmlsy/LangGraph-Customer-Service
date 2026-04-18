"""
evaluation/ragas_evaluator.py

RAGAS-based evaluation pipeline for the RAG subsystem.

RAGAS (Retrieval Augmented Generation Assessment) provides reference-free
evaluation metrics that do not require human-labelled ground truth answers
for every test question.  An LLM judge is used internally by RAGAS to assess
quality — this means evaluation can scale to thousands of questions cheaply.

Metrics computed:
    faithfulness        – Does the answer contain only information from the context?
                         High faithfulness = low hallucination rate.
                         Formula: (# claims supported by context) / (# total claims)

    answer_relevance    – Is the answer relevant to the question?
                         RAGAS generates N question variants from the answer and
                         measures cosine similarity to the original question.

    context_precision   – Are the retrieved chunks actually useful for the answer?
                         Measures what fraction of retrieved chunks are relevant.
                         Penalises over-retrieval of noise documents.

    context_recall      – (requires ground truth) Are all facts in the ground truth
                         answer covered by the retrieved context?

Target scores (from project benchmarks):
    Baseline:  RAGAS composite ≈ 0.62
    Optimised: RAGAS composite ≈ 0.79  (after hybrid search + reranker)

Dependencies:
    pip install ragas datasets langchain-openai
"""

from __future__ import annotations

import logging
from typing import Any
from ragas.metrics.collections import AnswerRelevancy, ContextPrecision, Faithfulness
from datasets import Dataset
from ragas import evaluate
from ragas.evaluation import EvaluationResult  # type: ignore[import-not-found]

logger = logging.getLogger(__name__)


class RAGASEvaluator:
    """
    Wrapper around the RAGAS evaluation framework.

    Attributes:
        llm:        LangChain LLM used by RAGAS as the judge model.
        embeddings: LangChain Embeddings instance used for semantic similarity.
        metrics:    List of RAGAS metric objects to compute.

    Example:
        evaluator = RAGASEvaluator(llm, embeddings)
        results = evaluator.evaluate_dataset(
            questions=["退款需要多久？"],
            answers=["退款通常需要3-5个工作日。"],
            contexts=[["退款政策：申请退款后3-7个工作日内退回原支付方式。"]],
            ground_truths=["3到7个工作日"],
        )
        print(results)  # {"faithfulness": 0.92, "answer_relevance": 0.87, ...}
    """

    def __init__(self, llm: Any, embeddings: Any) -> None:
        """
        Initialise the RAGAS evaluator.

        Args:
            llm:        LangChain chat model (used by RAGAS internally as judge).
                        Should be a capable model (GPT-4 or equivalent) for
                        accurate faithfulness scoring.
            embeddings: LangChain embeddings (used for answer_relevance metric).
        """
        self.llm        = llm
        self.embeddings = embeddings
        self.metrics    = [Faithfulness(llm=llm), AnswerRelevancy(llm=llm, embeddings=embeddings), ContextPrecision(llm=llm)]

        # Configure each metric with the judge LLM and embeddings
        # faithfulness.llm           = llm
        # answer_relevance.llm       = llm
        # answer_relevance.embeddings = embeddings
        # context_precision.llm      = llm

    def evaluate_dataset(
        self,
        questions: list[str],
        answers: list[str],
        contexts: list[list[str]],
        ground_truths: list[str] | None = None,
    ) -> dict:
        """
        Run RAGAS evaluation on a batch of QA samples.

        Args:
            questions:    List of user questions.
            answers:      List of model-generated answers (parallel to questions).
            contexts:     List of context lists; each inner list contains the
                          retrieved document texts for that question.
            ground_truths: Optional list of reference answers.  Required only
                          for context_recall metric; omit for reference-free eval.

        Returns:
            Dict of metric_name → score (float, range 0–1):
            {
                "faithfulness":     0.91,
                "answer_relevance": 0.84,
                "context_precision":0.78,
                "composite_score":  0.84,  # mean of all metrics
            }

        Tip: For large datasets (>1000 samples), sample a representative
        subset of 200–500 questions to keep evaluation time under 1 hour.
        """
        data = {
            "question":  questions,
            "answer":    answers,
            "contexts":  contexts,
        }
        if ground_truths:
            data["ground_truth"] = ground_truths

        dataset = Dataset.from_dict(data)
        result= evaluate(dataset, metrics=self.metrics)  
        df      = result.to_pandas() # type: ignore[union-attr]

        scores = {col: float(df[col].mean()) for col in df.columns
                    if col not in ("question", "answer", "contexts", "ground_truth")}
        scores["composite_score"] = sum(scores.values()) / len(scores)
        logger.info("RAGAS evaluation completed: %s", scores)
        return scores

    def generate_report(self, results: dict) -> str:
        """
        Format the RAGAS results dict into a human-readable markdown report.

        Args:
            results: Dict from evaluate_dataset().

        Returns:
            Markdown string with a table of metrics and pass/fail thresholds.
        """
        lines = [
            "# RAGAS Evaluation Report",
            "",
            "| Metric | Score | Threshold | Status |",
            "|--------|-------|-----------|--------|",
        ]
        thresholds = {
            "faithfulness":      0.85,
            "answer_relevance":  0.80,
            "context_precision": 0.75,
        }
        for metric, score in results.items():
            if metric == "composite_score":
                continue
            threshold = thresholds.get(metric, 0.75)
            status = "PASS" if score >= threshold else "FAIL"
            lines.append(f"| {metric} | {score:.3f} | {threshold:.2f} | {status} |")
        lines.append(f"\n**Composite score: {results.get('composite_score', 0):.3f}**")
        return "\n".join(lines)

    def evaluate_single(
        self,
        question: str,
        answer: str,
        context: list[str],
        ground_truth: str | None = None,
    ) -> dict:
        """
        Evaluate a single QA sample.  Convenience wrapper for online eval.

        Args:
            question:     Single user question.
            answer:       Single model answer.
            context:      List of retrieved context strings.
            ground_truth: Optional reference answer.

        Returns:
            Dict of metric scores for this sample.
        """
        return self.evaluate_dataset(
            [question], [answer], [context],
            [ground_truth] if ground_truth else None,
        )
