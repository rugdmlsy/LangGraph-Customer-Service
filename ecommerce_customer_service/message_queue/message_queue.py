"""
queue/message_queue.py

Message queue abstraction layer.

Defines a common interface (MessageQueue) so the worker code is decoupled
from the specific queue backend.  Swap Redis for Kafka by changing a single
config flag without touching worker.py.

Message format (JSON):
    {
        "request_id": str,   # UUID for deduplication and tracking
        "user_id":    str,
        "session_id": str,
        "query":      str,
        "history":    list,  # last N turns
        "timestamp":  str,   # ISO-8601
    }

Redis implementation:
    Uses LPUSH / BRPOP (blocking pop) pattern.
    LPUSH: O(1) push to the head of the list.
    BRPOP: blocks until a message is available, then atomically pops.
    This avoids busy-waiting and reduces Redis CPU usage vs. non-blocking RPOP.

Kafka implementation (optional):
    Producer: send() to the configured topic.
    Consumer: poll() in a loop with auto-commit disabled for at-least-once delivery.
"""

from __future__ import annotations

import abc
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Abstract base                                                                #
# --------------------------------------------------------------------------- #


class MessageQueue(abc.ABC):
    """
    Abstract message queue interface.

    All concrete implementations must provide push(), pop(), and peek().
    """

    @abc.abstractmethod
    def push(self, message: dict) -> str:
        """
        Enqueue a message.

        Args:
            message: Dict conforming to the message schema above.
                     A "request_id" field will be added if missing.

        Returns:
            The request_id assigned to the message.
        """

    @abc.abstractmethod
    def pop(self, timeout: int = 30) -> dict | None:
        """
        Dequeue and return the next message.

        Args:
            timeout: Seconds to block waiting for a message (0 = non-blocking).

        Returns:
            Message dict, or None if the queue is empty / timeout expired.
        """

    @abc.abstractmethod
    def peek(self) -> dict | None:
        """
        Return the next message without removing it.

        Returns:
            Message dict, or None if the queue is empty.
        """

    @abc.abstractmethod
    def size(self) -> int:
        """Return the number of messages currently in the queue."""


# --------------------------------------------------------------------------- #
# Redis implementation                                                         #
# --------------------------------------------------------------------------- #


class RedisQueue(MessageQueue):
    """
    Redis LIST-backed message queue using the LPUSH / BRPOP pattern.

    Attributes:
        redis_client: Initialised redis.Redis client.
        key:          Redis list key (e.g. "customer_service:queue").

    Example:
        import redis
        r = redis.Redis(host="localhost", port=6379)
        q = RedisQueue(r, key="cs:queue")
        q.push({"user_id": "u1", "query": "退款问题"})
        msg = q.pop(timeout=10)
    """

    def __init__(self, redis_client: Any, key: str = "customer_service:queue") -> None:
        """
        Initialise the Redis queue.

        Args:
            redis_client: Connected redis.Redis instance.
            key:          Redis list key for the queue.
        """
        self.redis = redis_client
        self.key = key


    def push(self, message: dict) -> str:
        """
        Serialize and push a message to the head of the Redis list.
        """
        if "request_id" not in message:
            message["request_id"] = str(uuid.uuid4())
        if "timestamp" not in message:
            message["timestamp"] = datetime.now(timezone.utc).isoformat()
        self.redis.lpush(self.key, json.dumps(message, ensure_ascii=False))
        logger.debug("Pushed request_id=%s to queue", message["request_id"])
        return message["request_id"]

    def pop(self, timeout: int = 30) -> dict | None:
        """
        Blocking pop from the tail of the Redis list.
        """
        result = self.redis.brpop(self.key, timeout=timeout)
        if result is None:
            return None  # timeout expired
        _, raw = result
        return json.loads(raw)

    def peek(self) -> dict | None:
        """
        Non-destructive peek at the last item (next to be popped).
        """
        raw = self.redis.lindex(self.key, -1)
        return json.loads(raw) if raw else None

    def size(self) -> int:
        """
        Return the queue length.
        """
        return self.redis.llen(self.key)


# --------------------------------------------------------------------------- #
# Kafka implementation (optional)                                              #
# --------------------------------------------------------------------------- #


class KafkaQueue(MessageQueue):
    """
    Kafka-backed message queue for higher-throughput deployments.

    Use when:
        - Message volume exceeds ~1 000 messages/second (Redis can handle ~10 k/s
          but Kafka is better for durability and replay).
        - You need message replay / audit trail (Kafka retains messages by default).
        - Multiple consumer groups need to read the same messages independently.

    Attributes:
        producer: kafka.KafkaProducer instance.
        consumer: kafka.KafkaConsumer instance.
        topic:    Kafka topic name.
    """

    def __init__(
        self,
        bootstrap_servers: str = "localhost:9092",
        topic: str = "customer_service_requests",
        group_id: str = "agent_workers",
    ) -> None:
        """
        Initialise Kafka producer and consumer.

        TODO:
            from kafka import KafkaProducer, KafkaConsumer
            self.topic = topic
            self.producer = KafkaProducer(
                bootstrap_servers=bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            )
            self.consumer = KafkaConsumer(
                topic,
                bootstrap_servers=bootstrap_servers,
                group_id=group_id,
                auto_offset_reset="earliest",
                enable_auto_commit=False,
                value_deserializer=lambda v: json.loads(v.decode("utf-8")),
            )
        """
        # TODO: implement
        pass

    def push(self, message: dict) -> str:
        """
        Produce a message to the Kafka topic.

        How to implement:
            if "request_id" not in message:
                message["request_id"] = str(uuid.uuid4())
            self.producer.send(self.topic, value=message)
            self.producer.flush()
            return message["request_id"]
        """
        # TODO: implement
        pass

    def pop(self, timeout: int = 30) -> dict | None:
        """
        Poll one message from Kafka.

        How to implement:
            records = self.consumer.poll(timeout_ms=timeout * 1000, max_records=1)
            for tp, msgs in records.items():
                if msgs:
                    self.consumer.commit()
                    return msgs[0].value
            return None
        """
        # TODO: implement
        pass

    def peek(self) -> dict | None:
        """Kafka does not support non-destructive peek; raise NotImplementedError."""
        # TODO: implement — raise NotImplementedError or use a workaround
        pass

    def size(self) -> int:
        """
        Estimate the number of unconsumed messages.

        How to implement:
            Use consumer.end_offsets() - consumer.position() per partition.
            Sum across all partitions.
        """
        # TODO: implement
        pass
