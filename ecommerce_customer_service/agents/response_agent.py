"""
agents/response_agent.py

Response Agent — final answer synthesis.

This agent is the last node before the graph terminates.  It receives all
intermediate results (RAG context, tool call outputs, conversation history)
and produces a single coherent, customer-friendly answer.

Responsibilities:
    1. Merge partial answers from FAQAgent and/or OrderAgent.
    2. Apply tone and style guidelines (polite, concise, in the user's language).
    3. Enrich the answer with follow-up suggestions when appropriate.
    4. Handle the edge case where all upstream agents returned empty results.

Why a separate ResponseAgent?
    - In complex queries (e.g. "Can I return this item and what's the policy?")
      both FAQAgent (policy info) and OrderAgent (refund eligibility) may run.
      A dedicated synthesis step avoids awkward concatenation artefacts.
    - Centralising style / tone enforcement in one agent makes A/B testing
      different response styles trivial.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class ResponseAgent:
    """
    Answer synthesis agent — terminal node of the LangGraph workflow.

    Attributes:
        llm: Chat model for synthesis.  Can be the same model instance shared
             across all agents or a smaller, faster model (e.g. Qwen1.5-1.8B)
             if latency is critical, since synthesis is a simpler task.

    LangGraph integration:
        graph.add_node("response_agent", response_agent.run)
        graph.set_finish_point("response_agent")
    """

    def __init__(self, llm: Any) -> None:
        """
        Initialise the response agent.

        Args:
            llm: Chat model instance.

        TODO:
            - self.llm = llm
            - Define self.synthesis_prompt: a ChatPromptTemplate with a system
              message that instructs the LLM to:
              * Merge information from multiple sources without repetition.
              * Answer in the same language as the user.
              * Be concise (≤ 3 paragraphs for most queries).
              * Suggest a relevant follow-up action when applicable.
              * Never fabricate information not present in the provided inputs.
        """
        # TODO: implement
        pass

    # ---------------------------------------------------------------------- #
    # Core synthesis                                                          #
    # ---------------------------------------------------------------------- #

    def synthesize(
        self,
        query: str,
        tool_results: list[dict],
        rag_results: list[dict],
        history: list[dict],
    ) -> str:
        """
        Merge tool call outputs and RAG snippets into a final customer response.

        Args:
            query:        Rewritten user query (for grounding the response).
            tool_results: List of tool call records from OrderAgent, each dict:
                          {"tool": str, "params": dict, "result": dict}.
                          May be empty if routing went to FAQAgent only.
            rag_results:  List of retrieved document dicts from FAQAgent.
                          May be empty if routing went to OrderAgent only.
            history:      Conversation history for coreference context.

        Returns:
            Final answer string suitable for returning directly to the user.

        How to implement:
            1. Determine what information is available:
               has_tool = len(tool_results) > 0
               has_rag  = len(rag_results) > 0
            2. If neither — return a graceful "I'm sorry, I couldn't find
               relevant information" message without calling the LLM.
            3. Build a prompt that includes:
               - Section A: Structured API results (formatted as bullet points).
               - Section B: Relevant FAQ passages (numbered citations).
               - User query.
               - Instruction: synthesise a single coherent answer from A and B.
            4. Call self.llm.invoke(messages) and return .content.
            5. Post-process: strip extra whitespace, ensure the response ends
               with a period / question mark for readability.

        Edge cases:
            - Tool call returned an error → acknowledge the failure politely
              and suggest the user contact human support.
            - RAG returned irrelevant docs (low scores) → omit them from the
              synthesis prompt to avoid confusing the LLM.
        """
        # TODO: implement
        pass

    # ---------------------------------------------------------------------- #
    # LangGraph node                                                          #
    # ---------------------------------------------------------------------- #

    def run(self, state: dict) -> dict:
        """
        LangGraph node entry point for the response agent.

        Args:
            state: AgentState dict.
                   Reads:  "rewritten_query" or "query", "tool_results",
                           "rag_results", "history".
                   Writes: "final_answer" (overrides any draft set by earlier agents).

        Returns:
            Partial state dict: {"final_answer": str}.

        How to implement:
            1. query        = state.get("rewritten_query") or state["query"]
            2. tool_results = state.get("tool_results", [])
            3. rag_results  = state.get("rag_results", [])
            4. history      = state.get("history", [])
            5. If the graph only ran one agent and it already set a good
               final_answer, optionally skip the LLM call and return as-is
               (check a "needs_synthesis" flag in state to control this).
            6. answer = self.synthesize(query, tool_results, rag_results, history)
            7. Return {"final_answer": answer}.
        """
        # TODO: implement
        pass
