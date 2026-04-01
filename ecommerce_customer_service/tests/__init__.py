"""
tests package

Unit and integration tests for the multi-agent system.

Test structure:
    test_agents.py  — RouterAgent, FAQAgent, OrderAgent unit tests
    test_rag.py     — chunker, embedder, retriever unit tests
    test_tools.py   — order/logistics/refund tool unit tests

Running tests:
    pytest tests/ -v                  # all tests
    pytest tests/test_rag.py -v       # RAG tests only
    pytest tests/ -k "test_router"    # filter by name
    pytest tests/ --cov=. --cov-report=html  # with coverage

Test philosophy:
    - Mock all external dependencies (Milvus, Redis, LLM API) in unit tests.
    - Use real MockOrderService / MockLogisticsService (they're already mocks).
    - Integration tests (not included here) would spin up real services
      via Docker Compose.
"""
