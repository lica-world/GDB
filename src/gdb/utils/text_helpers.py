"""Shared text processing helpers for model outputs."""

from __future__ import annotations

import difflib
import json
import re
from typing import Any, Optional


def strip_thinking(text: str) -> str:
    """Remove ``<think>...</think>`` blocks and extract content from code fences.

    Handles Qwen-style thinking tags and markdown fenced blocks that models
    commonly wrap around their answers.
    """
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    m = re.search(r"```(?:\w*)\s*\n?(.*?)```", text, re.DOTALL)
    if m:
        text = m.group(1).strip()
    return text


def strip_code_fence(text: str) -> str:
    """Remove leading/trailing markdown code fences (e.g. ```json ... ```)."""
    text = re.sub(r"^```(?:json)?\s*\n?", "", text)
    return re.sub(r"\n?```\s*$", "", text).strip()


def extract_json_obj(text: str) -> Any:
    """Find and parse the first JSON object or array in *text*.

    Strips code fences first, then tries a full parse followed by scanning
    for the first ``{`` or ``[``.  Returns ``None`` on failure.
    """
    text = strip_code_fence(text)
    try:
        return json.loads(text)
    except Exception:
        pass
    for i, ch in enumerate(text):
        if ch in ("{", "["):
            try:
                return json.loads(text[i:])
            except Exception:
                continue
    return None


def parse_json_from_text(text: str) -> Optional[Any]:
    """Robust JSON extraction from model output.

    Strips thinking blocks and code fences, then attempts balanced-brace
    parsing as a fallback.  Used by template tasks that may receive deeply
    nested JSON.
    """
    text = strip_thinking(text)
    fence_match = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1).strip()
    brace_start = text.find("{")
    bracket_start = text.find("[")
    if brace_start == -1 and bracket_start == -1:
        return None
    start = (
        bracket_start
        if bracket_start != -1 and (brace_start == -1 or bracket_start < brace_start)
        else brace_start
    )
    try:
        return json.loads(text[start:])
    except json.JSONDecodeError:
        depth = 0
        open_char = text[start]
        close_char = "]" if open_char == "[" else "}"
        for i in range(start, len(text)):
            if text[i] == open_char:
                depth += 1
            elif text[i] == close_char:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        return None
    return None


def normalized_edit_distance(a: str, b: str) -> float:
    """Normalized dissimilarity: ``1.0 - SequenceMatcher.ratio()``."""
    return 1.0 - difflib.SequenceMatcher(None, a, b).ratio()
