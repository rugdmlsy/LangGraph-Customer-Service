"""
graph/agent_graph.py

LangGraph StateGraph definition for the multi-agent customer service pipeline.

Graph topology:
    START → router_agent → (conditional) → faq_agent ──┐
                                        → order_agent ──┤
                                        → response_agent ← (always)
                                                       └→ END

Node functions:
    router_agent:   RouterAgent.route()   — rewrites query, classifies intent
    faq_agent:      FAQAgent.run()        — RAG retrieval + generation
    order_agent:    OrderAgent.run()      — tool-calling ReAct loop
    response_agent: ResponseAgent.run()   — final answer synthesis

State flow:
    Each node reads from AgentState and returns a partial state dict.
    LangGraph merges the partial dict into the global state automatically.
    All nodes share the same state; keys written by one node are readable
    by all subsequent nodes.

Why LangGraph over vanilla LangChain?
    - Explicit state graph makes the control flow auditable and debuggable.
    - Conditional edges enable clean intent-based routing without if/else chains
      in agent code.
    - Built-in support for human-in-the-loop (interrupt_before / interrupt_after).
    - StateGraph supports streaming output (stream() method) for real-time UX.
"""

from __future__ import annotations

import logging
import uuid, time
from typing import Annotated, Any

from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# State schema                                                                 #
# --------------------------------------------------------------------------- #


class AgentState(TypedDict, total=False):
    """
    Shared state TypedDict flowing through all graph nodes.

    All fields are optional (total=False) so each node only needs to declare
    the fields it reads/writes.  LangGraph performs shallow merge of partial
    state dicts returned by each node.

    Fields:
        query            (str):  Original user query (immutable).
        rewritten_query  (str):  Query after RouterAgent rewrite.
        intent           (str):  Classified intent (IntentType.value string).
        rag_results      (list): Retrieved documents from FAQAgent.
        tool_results     (list): Tool call records from OrderAgent.
        final_answer     (str):  Final answer string (written last by ResponseAgent).
        history          (list): Conversation history [{"role", "content"}].
        user_id          (str):  Unique user identifier.
        session_id       (str):  Session identifier (for short-term memory).
        error            (str):  Error message if any node fails (optional).
    """

    query: str
    rewritten_query: str
    intent: str
    rag_results: list[dict]
    tool_results: list[dict]
    final_answer: str
    history: list[dict]
    user_id: str
    session_id: str
    error: str


# --------------------------------------------------------------------------- #
# Graph factory                                                                #
# --------------------------------------------------------------------------- #


def build_graph(
    router_agent: Any,
    faq_agent: Any,
    order_agent: Any,
    response_agent: Any,
) -> Any:
    """
    Assemble and compile the multi-agent LangGraph StateGraph.

    Args:
        router_agent:   RouterAgent instance.
        faq_agent:      FAQAgent instance.
        order_agent:    OrderAgent instance.
        response_agent: ResponseAgent instance.

    Returns:
        A compiled LangGraph CompiledGraph ready to invoke with .invoke()
        or stream with .stream().

    How to implement:
        from langgraph.graph import StateGraph, START, END

        # 1. Create the graph with AgentState as the state schema
        graph = StateGraph(AgentState)

        # 2. Add nodes — each node is a callable (state: dict) → dict
        graph.add_node("router_agent",   router_agent.route)
        graph.add_node("faq_agent",      faq_agent.run)
        graph.add_node("order_agent",    order_agent.run)
        graph.add_node("response_agent", response_agent.run)

        # 3. Entry point
        graph.set_entry_point("router_agent")

        # 4. Conditional routing from router to specialist agents
        def routing_fn(state: AgentState) -> str:
            intent = state.get("intent", "unknown")
            if intent == "faq":
                return "faq_agent"
            elif intent in ("order", "logistics", "refund"):
                return "order_agent"
            else:
                return "response_agent"   # unknown → go directly to synthesis

        graph.add_conditional_edges(
            "router_agent",
            routing_fn,
            {
                "faq_agent":      "faq_agent",
                "order_agent":    "order_agent",
                "response_agent": "response_agent",
            }
        )

        # 5. Both specialist agents always flow to the response agent
        graph.add_edge("faq_agent",   "response_agent")
        graph.add_edge("order_agent", "response_agent")

        # 6. Response agent ends the graph
        graph.add_edge("response_agent", END)

        # 7. Compile (optionally add checkpointer for stateful multi-turn support)
        return graph.compile()

    Advanced: for multi-turn memory support, pass a MemorySaver checkpointer:
        from langgraph.checkpoint.memory import MemorySaver
        return graph.compile(checkpointer=MemorySaver())
    Then invoke with config={"configurable": {"thread_id": session_id}}.
    """
    graph = StateGraph(AgentState)
    graph.add_node("router_agent",   router_agent.route)
    graph.add_node("faq_agent",      faq_agent.run)
    graph.add_node("order_agent",    order_agent.run)
    graph.add_node("response_agent", response_agent.run)
    graph.set_entry_point("router_agent")   
    
    def routing_fn(state: AgentState) -> str:
        intent = state.get("intent", "unknown")
        if intent == "faq":
            return "faq_agent"
        elif intent in ("order", "logistics", "refund"):
            return "order_agent"
        else:
            return "response_agent"
        
    graph.add_conditional_edges(
        "router_agent",
        routing_fn,
        {
            "faq_agent":      "faq_agent",
            "order_agent":    "order_agent",
            "response_agent": "response_agent",
        }
    )
    graph.add_edge("faq_agent",   "response_agent")
    graph.add_edge("order_agent", "response_agent")
    graph.add_edge("response_agent", END)
    return graph.compile()


# --------------------------------------------------------------------------- #
# High-level runner                                                            #
# --------------------------------------------------------------------------- #


def run_graph(
    query: str,
    user_id: str,
    history: list[dict] | None = None,
    session_id: str | None = None,
    compiled_graph: Any = None,
) -> str:
    """
    Run the compiled agent graph on a single user query.

    This is the primary interface used by:
        - api/main.py (direct synchronous mode)
        - queue/worker.py (async mode via thread pool)

    Args:
        query:          Raw user query string.
        user_id:        User identifier for long-term memory lookup.
        history:        Conversation history.  Pass None to start fresh.
        session_id:     Session identifier.  If None, a UUID is generated.
        compiled_graph: Pre-compiled graph object.  Pass None to use the
                        module-level singleton (built at startup).

    Returns:
        Final answer string from ResponseAgent.

    How to implement:
        import uuid, time
        if history is None:
            history = []
        if session_id is None:
            session_id = str(uuid.uuid4())

        initial_state: AgentState = {
            "query":      query,
            "history":    history,
            "user_id":    user_id,
            "session_id": session_id,
        }

        graph = compiled_graph or _compiled_graph_singleton  # lazy init
        start = time.monotonic()
        result = graph.invoke(initial_state)
        latency_ms = (time.monotonic() - start) * 1000

        logger.info(
            "Graph completed | user=%s | intent=%s | latency=%.1f ms",
            user_id, result.get("intent"), latency_ms
        )
        return result.get("final_answer", "抱歉，系统暂时无法处理您的请求。")

    For streaming output:
        for event in graph.stream(initial_state):
            # event is a dict of {node_name: partial_state}
            yield event
    """
    if history is None:
        history = []
    if session_id is None:
        session_id = str(uuid.uuid4())

    initial_state: AgentState = {
        "query":      query,
        "history":    history,
        "user_id":    user_id,
        "session_id": session_id,
    }

    graph = compiled_graph or _compiled_graph_singleton  # lazy init
    start = time.monotonic()
    result = graph.invoke(initial_state)
    latency_ms = (time.monotonic() - start) * 1000

    logger.info(
        "Graph completed | user=%s | intent=%s | latency=%.1f ms",
        user_id, result.get("intent"), latency_ms
    )
    return result.get("final_answer", "抱歉，系统暂时无法处理您的请求。")


# --------------------------------------------------------------------------- #
# Module-level singleton (initialised at startup)                             #
# --------------------------------------------------------------------------- #

# The compiled graph singleton is built once at application startup
# (e.g. in FastAPI lifespan) and reused across all requests.
# This avoids re-instantiating agents and re-loading models on every request.
_compiled_graph_singleton: Any = None


def init_graph(settings: Any = None) -> Any:
    """
    Build and cache the compiled graph singleton.

    Call this once during application startup before handling any requests.

    Args:
        settings: Settings instance.  If None, imports from config.

    Returns:
        The compiled graph (also stored as module-level singleton).

    How to implement:
        global _compiled_graph_singleton

        from config import settings as cfg
        if settings is None:
            settings = cfg

        # Initialise LLM
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(
            base_url=settings.LLM_API_BASE,
            api_key=settings.LLM_API_KEY,
            model=settings.LLM_MODEL_NAME,
            temperature=settings.LLM_TEMPERATURE,
            max_tokens=settings.LLM_MAX_TOKENS,
        )

        # Initialise RAG components
        from rag import Embedder, KnowledgeBase, SlidingWindowChunker, Retriever
        embedder = Embedder(settings.EMBEDDING_MODEL_NAME)
        chunker  = SlidingWindowChunker(settings.CHUNK_SIZE, settings.CHUNK_OVERLAP)
        kb       = KnowledgeBase(embedder, chunker, settings.MILVUS_HOST, settings.MILVUS_PORT)
        kb.connect()
        retriever = Retriever(kb, embedder, settings.RERANKER_MODEL_NAME)

        # Initialise agents
        from agents import RouterAgent, FAQAgent, OrderAgent, ResponseAgent
        from tools import ALL_TOOLS
        router   = RouterAgent(llm)
        faq      = FAQAgent(llm, retriever)
        order    = OrderAgent(llm, ALL_TOOLS)
        response = ResponseAgent(llm)

        _compiled_graph_singleton = build_graph(router, faq, order, response)
        return _compiled_graph_singleton
    """
    global _compiled_graph_singleton

    from config import settings as cfg
    if settings is None:
        settings = cfg

    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(
        base_url=settings.LLM_API_BASE,
        api_key=settings.LLM_API_KEY,
        model=settings.LLM_MODEL_NAME,
        temperature=settings.LLM_TEMPERATURE,
        max_completion_tokens=settings.LLM_MAX_TOKENS,
    )
    
    from rag import Embedder, KnowledgeBase, SlidingWindowChunker, Retriever
    embedder = Embedder(settings.EMBEDDING_MODEL_NAME)
    chunker  = SlidingWindowChunker(settings.CHUNK_SIZE, settings.CHUNK_OVERLAP)
    kb       = KnowledgeBase(embedder, chunker, settings.MILVUS_HOST, settings.MILVUS_PORT)
    kb.connect()
    retriever = Retriever(kb, embedder, settings.RERANKER_MODEL_NAME)   
    
    from agents import RouterAgent, FAQAgent, OrderAgent, ResponseAgent
    from tools import ALL_TOOLS
    router   = RouterAgent(llm)
    faq      = FAQAgent(llm, retriever)
    order    = OrderAgent(llm, ALL_TOOLS)
    response = ResponseAgent(llm)
    
    _compiled_graph_singleton = build_graph(router, faq, order, response)
    return _compiled_graph_singleton