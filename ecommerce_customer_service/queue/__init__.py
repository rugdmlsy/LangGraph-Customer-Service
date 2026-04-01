"""
queue package

Asynchronous message queue infrastructure for decoupling HTTP request
acceptance from agent processing.

Components:
    MessageQueue   – abstract base class defining the queue interface.
    RedisQueue     – Redis LIST-backed implementation (primary).
    KafkaQueue     – Kafka-backed implementation (optional, for higher throughput).
    QueueWorker    – Worker that pops messages and dispatches to ThreadPoolExecutor.

Architecture pattern:
    FastAPI endpoint → LPUSH → Redis queue → QueueWorker BRPOP → ThreadPool → AgentGraph
    This decoupling allows:
        - HTTP responses to return immediately (202 Accepted) while processing continues.
        - Graceful backpressure: the queue absorbs spikes without dropping requests.
        - Easy horizontal scaling: add more QueueWorker processes as load increases.
"""

from queue.message_queue import MessageQueue, RedisQueue
from queue.worker import QueueWorker

__all__ = ["MessageQueue", "RedisQueue", "QueueWorker"]
