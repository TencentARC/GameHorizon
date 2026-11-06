"""Shared implementation for the unified annotation stages."""

from .profiles import GameProfile, load_profile, supported_games
from .schema import SCHEMA_VERSION

__all__ = ["GameProfile", "SCHEMA_VERSION", "load_profile", "supported_games"]

