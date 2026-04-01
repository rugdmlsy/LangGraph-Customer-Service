"""
memory/long_term.py

Long-term user memory backed by Redis.

Stores cross-session information that persists beyond a single conversation:
    - User preferences (preferred language, notification settings)
    - Past order context (frequently referenced order IDs)
    - Customer loyalty tier (affects refund policy checks)
    - Previous complaint categories (for agent personalisation)

Redis data model:
    All keys are namespaced under "user:{user_id}:" for isolation.
    Preferences: HSET user:{user_id}:preferences  key value
    Order context: SET user:{user_id}:order:{order_id}  JSON  EX ttl
    Profile summary: SET user:{user_id}:profile  JSON  EX ttl

Why Redis for long-term memory?
    - Sub-millisecond read latency is critical in the agent loop.
    - HSET/HGET operations are O(1) per field.
    - Built-in TTL avoids memory bloat from stale user data.
    - Persistence (AOF/RDB) survives restarts without data loss.
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


class LongTermMemory:
    """
    Redis-backed persistent user memory.

    Attributes:
        redis_client: A redis.Redis (or redis.asyncio.Redis) client instance.
        user_id:      Unique user identifier used as part of all Redis keys.
        ttl:          Default TTL in seconds for new keys.

    Example:
        import redis
        r = redis.Redis(host="localhost", port=6379)
        mem = LongTermMemory(redis_client=r, user_id="user_123")
        mem.save_preference("language", "zh-CN")
        lang = mem.get_preference("language")  # → "zh-CN"
    """

    def __init__(
        self,
        redis_client: Any,
        user_id: str,
        ttl: int = 86400 * 30,
    ) -> None:
        """
        Initialise long-term memory for a specific user.

        Args:
            redis_client: Initialised redis.Redis client.  Inject from the
                          application startup context so connection pooling
                          is shared across all memory instances.
            user_id:      Unique, stable user identifier (e.g. platform UID).
            ttl:          TTL in seconds for preference and profile keys.
                          Default: 30 days.

        TODO:
            - self.redis = redis_client
            - self.user_id = user_id
            - self.ttl = ttl
            - self._pref_key  = f"user:{user_id}:preferences"
            - self._profile_key = f"user:{user_id}:profile"
        """
        # TODO: implement
        pass

    # ---------------------------------------------------------------------- #
    # Preferences                                                             #
    # ---------------------------------------------------------------------- #

    def save_preference(self, key: str, value: str) -> None:
        """
        Store a user preference as a string field in a Redis hash.

        Args:
            key:   Preference name (e.g. "language", "contact_channel").
            value: Preference value as a string.

        How to implement:
            self.redis.hset(self._pref_key, key, value)
            self.redis.expire(self._pref_key, self.ttl)
        """
        # TODO: implement
        pass

    def get_preference(self, key: str, default: str = "") -> str:
        """
        Retrieve a stored user preference.

        Args:
            key:     Preference key.
            default: Value to return if the key does not exist.

        Returns:
            Stored preference string, or `default` if not found.

        How to implement:
            value = self.redis.hget(self._pref_key, key)
            if value is None:
                return default
            return value.decode("utf-8") if isinstance(value, bytes) else value
        """
        # TODO: implement
        pass

    def get_all_preferences(self) -> dict[str, str]:
        """
        Return all preferences for the user as a dict.

        How to implement:
            raw = self.redis.hgetall(self._pref_key)
            return {k.decode(): v.decode() for k, v in raw.items()}
        """
        # TODO: implement
        pass

    # ---------------------------------------------------------------------- #
    # Order context                                                           #
    # ---------------------------------------------------------------------- #

    def save_order_context(self, order_id: str, context: dict) -> None:
        """
        Persist a structured order context dict for future reference.

        Useful for coreference resolution: when the user says "that order"
        the agent can look up the most recently discussed order.

        Args:
            order_id: Order identifier string.
            context:  Dict with relevant order info
                      (status, amount, creation_date, etc.).

        How to implement:
            key = f"user:{self.user_id}:order:{order_id}"
            self.redis.set(key, json.dumps(context), ex=self.ttl)
        """
        # TODO: implement
        pass

    def get_order_context(self, order_id: str) -> dict:
        """
        Retrieve a previously stored order context.

        Args:
            order_id: Order identifier.

        Returns:
            Context dict, or empty dict if not found.

        How to implement:
            key = f"user:{self.user_id}:order:{order_id}"
            raw = self.redis.get(key)
            return json.loads(raw) if raw else {}
        """
        # TODO: implement
        pass

    # ---------------------------------------------------------------------- #
    # User profile                                                            #
    # ---------------------------------------------------------------------- #

    def save_user_profile(self, profile: dict) -> None:
        """
        Persist a user profile summary (e.g. loyalty tier, total spend).

        Args:
            profile: Dict with profile fields.

        How to implement:
            self.redis.set(self._profile_key, json.dumps(profile), ex=self.ttl)
        """
        # TODO: implement
        pass

    def get_user_profile(self) -> dict:
        """
        Retrieve the stored user profile.

        Returns:
            Profile dict, or empty dict if not set.

        How to implement:
            raw = self.redis.get(self._profile_key)
            return json.loads(raw) if raw else {}
        """
        # TODO: implement
        pass
