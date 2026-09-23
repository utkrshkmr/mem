"""Strict single tool-call parser."""

from __future__ import annotations

import json
import re
from typing import Any

CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.S)

FULL_TOOLS = frozenset({
    "memory_write", "memory_update", "memory_delete", "memory_read",
    "memory_search", "memory_list", "next_chunk", "memory_store_span",
})
MINIMAL_TOOLS = frozenset({
    "memory_write", "memory_delete", "memory_read", "memory_list", "next_chunk",
})


def parse_call(text: str) -> tuple[dict[str, Any] | None, str | None]:
    if type(text) is not str:
        return None, "PARSE_ERROR"
    matches = list(CALL_RE.finditer(text))
    if len(matches) != 1:
        return None, "PARSE_ERROR"
    trailing = text[matches[0].end():].strip()
    if trailing:
        return None, "TRAILING_CONTENT"
    try:
        obj = json.loads(matches[0].group(1))
    except json.JSONDecodeError:
        return None, "BAD_JSON"
    if not isinstance(obj, dict) or set(obj) != {"name", "arguments"}:
        return None, "BAD_SCHEMA"
    if not isinstance(obj["name"], str) or not isinstance(obj["arguments"], dict):
        return None, "BAD_SCHEMA"
    return obj, None


def _ints(value: object, name: str) -> tuple[int, ...] | str:
    if type(value) is not list or not value:
        return "BAD_ARGUMENT"
    items = []
    for item in value:
        if type(item) is not int:
            return "BAD_ARGUMENT"
        if item < 1:
            return "BAD_ARGUMENT"
        items.append(item)
    if len(set(items)) != len(items):
        return "BAD_ARGUMENT"
    return tuple(items)


def validate_arguments(name: str, arguments: dict[str, Any], *, tools: str) -> str | None:
    allowed = FULL_TOOLS if tools == "full" else MINIMAL_TOOLS
    if tools not in {"full", "minimal"}:
        raise ValueError("tools must be full or minimal")
    if name not in allowed:
        return "UNKNOWN_TOOL"
    if name == "memory_write":
        if set(arguments) != {"key", "value"}:
            return "BAD_ARGUMENT"
        if type(arguments["key"]) is not str or type(arguments["value"]) is not str:
            return "BAD_ARGUMENT"
        return None
    if name == "memory_update":
        if "slot_id" not in arguments or type(arguments["slot_id"]) is not int:
            return "BAD_ARGUMENT"
        extra = set(arguments) - {"slot_id", "key", "value"}
        if extra or set(arguments) == {"slot_id"}:
            return "BAD_ARGUMENT"
        if "key" in arguments and type(arguments["key"]) is not str:
            return "BAD_ARGUMENT"
        if "value" in arguments and type(arguments["value"]) is not str:
            return "BAD_ARGUMENT"
        return None
    if name in {"memory_delete", "memory_read"}:
        if set(arguments) != {"slot_ids"}:
            return "BAD_ARGUMENT"
        parsed = _ints(arguments["slot_ids"], "slot_ids")
        if type(parsed) is str:
            return parsed
        return None
    if name == "memory_search":
        if set(arguments) != {"query", "top_k"}:
            return "BAD_ARGUMENT"
        if type(arguments["query"]) is not str or type(arguments["top_k"]) is not int:
            return "BAD_ARGUMENT"
        if arguments["top_k"] < 1:
            return "BAD_ARGUMENT"
        return None
    if name == "memory_list":
        if set(arguments) != {"page"} or type(arguments["page"]) is not int:
            return "BAD_ARGUMENT"
        if arguments["page"] != 0:
            return "INDEX_OVERFLOW"
        return None
    if name == "next_chunk":
        if arguments:
            return "BAD_ARGUMENT"
        return None
    if name == "memory_store_span":
        needed = {"start_char", "end_char", "key"}
        if set(arguments) != needed:
            return "BAD_ARGUMENT"
        if type(arguments["start_char"]) is not int or type(arguments["end_char"]) is not int:
            return "BAD_ARGUMENT"
        if type(arguments["key"]) is not str:
            return "BAD_ARGUMENT"
        if arguments["start_char"] < 0 or arguments["end_char"] <= arguments["start_char"]:
            return "BAD_ARGUMENT"
        return None
    return "UNKNOWN_TOOL"
