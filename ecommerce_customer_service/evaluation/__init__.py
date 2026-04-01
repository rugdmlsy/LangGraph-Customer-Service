"""
evaluation package

Automated evaluation infrastructure for the multi-agent system.

Components:
    RAGASEvaluator   – online/offline RAG quality metrics (faithfulness,
                       answer relevance, context precision) using RAGAS framework.
    OfflineEvaluator – batch evaluation on the 5 000-question test set with
                       intent accuracy, tool success rate, and composite scores.
"""

from evaluation.ragas_evaluator import RAGASEvaluator
from evaluation.offline_evaluator import OfflineEvaluator

__all__ = ["RAGASEvaluator", "OfflineEvaluator"]
