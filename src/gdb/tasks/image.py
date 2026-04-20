"""Backward-compatible shim for image-domain task imports.

Text removal (`image-6`) is implemented in `gdb.tasks.typography`.
"""

from gdb.tasks.typography import TextRemoval

__all__ = ["TextRemoval"]
