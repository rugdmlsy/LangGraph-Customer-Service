"""
agents/router_agent.py

Intent Router Agent — the entry point of every user request.

Responsibilities:
    1. Query Rewrite: normalise informal / ambiguous queries using an LLM prompt
       so that downstream agents receive a well-formed, explicit query.
    2. Intent Classification: decide which specialist agent should handle the
       rewritten query (FAQ, ORDER, LOGISTICS, REFUND, or UNKNOWN).
    3. LangGraph Routing: expose a `route` method that acts as a LangGraph node
       and returns the name of the next node to transition to.

Design rationale:
    - Separating rewrite from classification keeps each step testable.
    - A few-shot LLM prompt handles most classification; a fine-tuned
      lightweight classifier (e.g. BERT on ~1 k labelled examples) can be
      swapped in for lower latency and higher accuracy.
    - The router is intentionally stateless; all state is stored in AgentState
      (graph/agent_graph.py) and threaded through the graph.
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Intent taxonomy                                                              #
# --------------------------------------------------------------------------- #


class IntentType(str, Enum):
    """
    Exhaustive set of intents the router can classify.

    Values are plain strings so they serialise cleanly to JSON / Redis.

    Implementation note:
        Extend this enum if the business adds new service categories
        (e.g. MULTIMODAL for image-based complaints).
    """

    FAQ = "faq"
    ORDER = "order"
    LOGISTICS = "logistics"
    REFUND = "refund"
    UNKNOWN = "unknown"


# --------------------------------------------------------------------------- #
# Router Agent                                                                 #
# --------------------------------------------------------------------------- #


class RouterAgent:
    """
    Stateless LangGraph node that rewrites the user query and routes it to
    the appropriate specialist agent.

    Attributes:
        llm: An LLM client instance (e.g. langchain_openai.ChatOpenAI pointed
             at a local vLLM server, or a HuggingFace pipeline wrapper).
             The same LLM is reused for both rewrite and classification.
        intent_labels: Ordered list of IntentType values used in the
                       classification prompt to enumerate valid choices.

    Typical LangGraph usage:
        graph.add_node("router", router_agent.route)
        graph.add_conditional_edges("router", router_agent.route, {
            "faq":       "faq_agent",
            "order":     "order_agent",
            "logistics": "order_agent",   # order_agent handles logistics too
            "refund":    "order_agent",
            "unknown":   "response_agent",
        })
    """

    def __init__(self, llm: Any) -> None:
        """
        Initialise the router with an LLM instance.

        Args:
            llm: Any LangChain-compatible chat model (supports .invoke() /
                 .ainvoke()).  Recommended: ChatOpenAI with model pointed at
                 vLLM serving Qwen2.5-7B-Instruct.

        TODO:
            - Store llm as self.llm.
            - Optionally load a fine-tuned intent classifier from disk
              (transformers AutoModelForSequenceClassification) and store as
              self.classifier for sub-20 ms classification without an LLM call.
        """
        self.llm = llm
        self.routing_map = {
            IntentType.FAQ: "faq_agent",
            IntentType.ORDER: "order_agent",
            IntentType.LOGISTICS: "order_agent",
            IntentType.REFUND: "order_agent",
            IntentType.UNKNOWN: "response_agent",
        }
        self.rewrite_prompt = [
            ("system", 
            """You are a query rewriting assistant for an e-commerce customer service chatbot.

            Your task is to rewrite the user's latest query into a canonical, self-contained form that improves retrieval and intent classification accuracy.

            Instructions:
            1. Resolve pronouns and references using the conversation history. For example:
            - "它到了吗？" (Has it arrived?) → identify the specific product/order from history and substitute.
            - Replace vague pronouns with concrete entities (product names, order numbers, etc.).

            2. Expand abbreviations, slang, and informal language into standard terms. For example:
            - "快递" (express delivery) → "物流" (logistics status) if context suggests tracking.
            - "退货" → "退款" (refund) depending on intent.

            3. Add implicit context that clarifies the query. For example:
            - If the user mentions an order number from earlier in the conversation, include it explicitly.
            - If discussing a previous purchase, mention the product or order ID.

            4. Output ONLY the rewritten query in Mandarin Chinese. Do not include explanations, JSON, or any other text.

            Rewritten query:""")
        ]
        self.classify_prompt = [
            ("system", 
            """You are an intent classification assistant for an e-commerce customer service chatbot.

            Your task is to classify the user's rewritten query into one of the following categories:

            - FAQ: Questions about products, services, or policies.
            - ORDER: Queries related to order status, modifications, or cancellations.
            - LOGISTICS: Questions about shipping, delivery, or tracking information.
            - REFUND: Requests for refunds or exchanges.
            - UNKNOWN: Queries that do not fit into any of the above categories.

            Instructions:
            1. Read the rewritten query carefully.
            2. Determine the most appropriate category based on the content and intent.
            3. Output ONLY the category name in all uppercase letters.

            Category:""")
        ]

    # ---------------------------------------------------------------------- #
    # Step 1 – Query Rewrite                                                  #
    # ---------------------------------------------------------------------- #

    def rewrite_query(self, query: str, history: list[dict]) -> str:
        """
        Use the LLM to rewrite an informal user query into a canonical form
        that improves retrieval and tool-selection accuracy.

        Args:
            query:   Raw user input, e.g. "快递怎么还没到？"
            history: List of previous turns [{"role": "user"|"assistant",
                     "content": "..."}] for coreference resolution.
                     An empty list means no prior context.

        Returns:
            Rewritten query string, e.g.
            "查询订单 ORD-20240310-001 的物流状态"
            "查询用户 USER-12345 最近一次订单的物流状态"

        How to implement:
            1. Build a system prompt that instructs the LLM to:
               - Resolve pronouns using history ("它" → the specific product).
               - Expand abbreviations / slang.
               - Add implicit context (order number if mentioned earlier).
               - Output ONLY the rewritten query, no explanation.
            2. Format `history` into a conversational prompt block.
            3. Call self.llm.invoke([SystemMessage(...), *history, HumanMessage(query)]).
            4. Strip leading/trailing whitespace from the response.
            5. If the LLM call fails, fall back to returning `query` unchanged
               and log a warning — never crash the pipeline on rewrite failure.

        Example prompt snippet:
            "You are a query rewriting assistant for an e-commerce customer
             service chatbot. Given the conversation history and the user's
             latest message, output a single rewritten query in Mandarin
             Chinese that is self-contained and specific. Output ONLY the
             rewritten query."
        """
        history_messages = [(turn["role"], turn["content"]) for turn in history]
        prompt = self.rewrite_prompt + history_messages + [("human", query)]
        try:
            response = self.llm.invoke(prompt)
            rewritten_query = response.content.strip()
            return rewritten_query
        except Exception as e:
            logger.warning(f"LLM rewrite failed: {e}. Returning original query.")
            return query
        

    # ---------------------------------------------------------------------- #
    # Step 2 – Intent Classification                                          #
    # ---------------------------------------------------------------------- #

    def classify_intent(self, query: str) -> IntentType:
        """
        Classify the (rewritten) query into one of the IntentType categories.

        Args:
            query: Rewritten, normalised user query.

        Returns:
            IntentType enum member representing the predicted intent.

        How to implement (choose one approach based on latency budget):

        Approach A — LLM few-shot classification (simplest, ~200–500 ms):
            1. Build a prompt listing all IntentType values with one example each.
            2. Ask the LLM to respond with ONLY the intent label (one word).
            3. Parse the response; on parse failure return IntentType.UNKNOWN.

        Approach B — Fine-tuned classifier (recommended for production, ~10 ms):
            1. Collect ~500–2 000 labelled query examples per intent.
            2. Fine-tune bert-base-chinese or a smaller BERT variant using
               Hugging Face Trainer on the intent classification task.
            3. At inference: tokenize → forward pass → argmax over logits.
            4. Map predicted class index to IntentType.

        Approach C — Keyword heuristics (fast baseline):
            1. Maintain keyword → IntentType mapping dicts.
            2. Return the first matching intent; default to UNKNOWN.
            3. Use as fallback when LLM is unavailable.

        Key metric: intent accuracy target ≥ 93%.
        """
        prompt = self.classify_prompt + [("human", query)]
        try:
            response = self.llm.invoke(prompt)
            intent_str = response.content.strip().lower()
            intent = IntentType(intent_str) if intent_str in IntentType.__members__.values() else IntentType.UNKNOWN
            return intent
        except Exception as e:
            logger.warning(f"LLM classification failed: {e}. Returning UNKNOWN intent.")
            return IntentType.UNKNOWN

    # ---------------------------------------------------------------------- #
    # Step 3 – LangGraph node                                                 #
    # ---------------------------------------------------------------------- #

    def route(self, state: dict) -> dict:
        """
        LangGraph node function. Rewrites the query, classifies intent, and
        returns a partial state dict. Routing is handled by conditional edges
        in agent_graph.py which read state["intent"].

        Args:
            state: AgentState TypedDict (see graph/agent_graph.py).
                   Relevant keys: "query", "history".

        Returns:
            Partial state dict with "rewritten_query" and "intent" keys.
        """
        query = state["query"]
        history = state.get("history", [])
        rewritten = self.rewrite_query(query, history)
        intent = self.classify_intent(rewritten)
        logger.info("Routing query '%s' with intent '%s'", rewritten, intent.value)
        return {"rewritten_query": rewritten, "intent": intent.value}
