"""
agents/faq_agent.py

FAQ Agent — answers product and policy questions using Retrieval-Augmented Generation.

Responsibilities:
    1. Retrieve relevant documents from the Milvus knowledge base using hybrid
       search (dense ANN + sparse BM25) followed by cross-encoder reranking.
    2. Generate a grounded, concise answer by passing the retrieved context and
       conversation history to the LLM.
    3. Expose a `run` method compatible with LangGraph node conventions.

Design rationale:
    - Decoupling retrieval from generation lets each step be unit-tested and
      swapped independently (e.g. upgrade reranker without touching generation).
    - The agent reads and writes only the keys it owns in AgentState, keeping
      the graph data-flow explicit.
    - Generation uses a structured prompt that instructs the LLM to answer
      ONLY from the provided context, which is the primary RAGAS faithfulness
      control knob.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class FAQAgent:
    """
    RAG-based FAQ agent.

    Attributes:
        llm:       LangChain-compatible chat model for answer generation.
        retriever: rag.retriever.Retriever instance that performs hybrid search
                   and reranking.

    LangGraph integration:
        graph.add_node("faq_agent", faq_agent.run)
        graph.add_edge("faq_agent", "response_agent")
    """

    def __init__(self, llm: Any, retriever: Any) -> None:
        """
        Initialise the FAQ agent.

        Args:
            llm:       Chat model instance (same LLM shared across agents
                       to avoid loading multiple model replicas).
            retriever: rag.retriever.Retriever instance.  Injected so that
                       tests can substitute a mock retriever.

        TODO:
            - Assign self.llm = llm and self.retriever = retriever.
            - Set up a LangChain ChatPromptTemplate for the generation step.
              Store it as self.prompt_template for reuse across calls.
        """
        # TODO: implement
        pass

    # ---------------------------------------------------------------------- #
    # Step 1 – Retrieval                                                      #
    # ---------------------------------------------------------------------- #

    def retrieve_context(self, query: str, top_k: int = 3) -> list[dict]:
        """
        Fetch the most relevant documents for `query` from the knowledge base.

        Args:
            query: Rewritten user query (output of RouterAgent.rewrite_query).
            top_k: Number of documents to return after reranking.
                   Defaults to settings.RERANK_TOP_N if not specified.

        Returns:
            List of document dicts, each containing at minimum:
            {
                "text":     str,   # chunk text
                "score":    float, # reranker relevance score
                "metadata": dict,  # source file, chunk_id, etc.
            }

        How to implement:
            1. Call self.retriever.hybrid_search(query, top_k=settings.RETRIEVAL_TOP_K)
               to get first-stage candidates.
            2. Pass candidates to self.retriever.rerank(query, docs, top_n=top_k)
               for score refinement.
            3. Return the reranked list.
            4. On retriever failure, return [] and log the exception — the
               generation step should gracefully handle an empty context by
               admitting it does not have enough information.
        """
        # TODO: implement
        pass

    # ---------------------------------------------------------------------- #
    # Step 2 – Generation                                                     #
    # ---------------------------------------------------------------------- #

    def generate_answer(
        self,
        query: str,
        context: list[dict],
        history: list[dict],
    ) -> str:
        """
        Generate a grounded answer using the retrieved context and LLM.

        Args:
            query:   Rewritten user query.
            context: List of document dicts from retrieve_context().
                     May be empty if retrieval found nothing relevant.
            history: Conversation history list [{"role": ..., "content": ...}].

        Returns:
            Model-generated answer string.

        How to implement:
            1. Concatenate context[i]["text"] into a single context block,
               delimited by "---" separators and numbered for traceability.
            2. Build the prompt using self.prompt_template:
               - System: instruct the LLM to answer strictly from context,
                 cite which chunk supports each claim (optional), and reply
                 in the same language as the user query (Chinese / English).
               - Context: the numbered chunks.
               - History: last N turns (default N=5 to bound context length).
               - Human: the rewritten query.
            3. Call self.llm.invoke(messages) and extract .content.
            4. If context is empty, return a templated "I don't have enough
               information" message rather than calling the LLM, to avoid
               hallucinations.
            5. Log query + answer length at DEBUG level for latency tracking.

        RAGAS faithfulness tip:
            The prompt must explicitly say "Do not include information not
            present in the provided context." This is the single biggest lever
            for faithfulness scores.
        """
        # TODO: implement
        pass

    # ---------------------------------------------------------------------- #
    # LangGraph node                                                          #
    # ---------------------------------------------------------------------- #

    def run(self, state: dict) -> dict:
        """
        LangGraph node entry point for the FAQ agent.

        Args:
            state: AgentState dict.  Reads: "rewritten_query", "history".
                   Writes: "rag_results" (list of retrieved docs),
                           "final_answer" (draft answer — may be overwritten
                           by ResponseAgent if other agents also ran).

        Returns:
            Partial state dict with updated keys.

        How to implement:
            1. query   = state.get("rewritten_query") or state["query"]
            2. history = state.get("history", [])
            3. docs    = self.retrieve_context(query)
            4. answer  = self.generate_answer(query, docs, history)
            5. Return {"rag_results": docs, "final_answer": answer}
            6. Wrap everything in try/except; on failure set final_answer to
               a friendly error message and log the traceback.
        """
        # TODO: implement
        pass
