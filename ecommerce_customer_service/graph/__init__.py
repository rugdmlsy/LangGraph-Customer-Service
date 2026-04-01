"""
graph package

LangGraph workflow definition for the multi-agent pipeline.

Exports:
    AgentState  – TypedDict defining the state schema shared across all nodes.
    build_graph – Factory function that assembles and compiles the StateGraph.
    run_graph   – High-level function to run the compiled graph on a query.
"""

from graph.agent_graph import AgentState, build_graph, run_graph

__all__ = ["AgentState", "build_graph", "run_graph"]
