"""
config package

Exposes the Settings singleton so the rest of the codebase can do:
    from config import settings
"""

from config.settings import Settings

settings = Settings()

__all__ = ["settings", "Settings"]
