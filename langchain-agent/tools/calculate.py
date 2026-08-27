"""Evaluate a mathematical expression and return the result."""
from __future__ import annotations

import json
import math as _math
from typing import Any

from langchain_core.tools import tool

_ALLOWED_NAMES: dict[str, Any] = {
    name: getattr(_math, name)
    for name in dir(_math)
    if not name.startswith("_")
}
_ALLOWED_NAMES["abs"] = abs
_ALLOWED_NAMES["round"] = round
_ALLOWED_NAMES["min"] = min
_ALLOWED_NAMES["max"] = max


@tool(parse_docstring=True)
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression and return the numeric result.
    Supports standard arithmetic operators (+, -, *, /, //, %, **),
    parentheses, and Python's math module functions (e.g. sqrt, log, sin,
    cos, pi, e). Examples: '2 + 3 * 4', 'sqrt(144)', 'log(1000, 10)'.

    Args:
        expression: The mathematical expression to evaluate.
    """
    if not isinstance(expression, str) or not expression.strip():
        return json.dumps({"error": "expression must be a non-empty string"})
    try:
        result = eval(expression, {"__builtins__": {}}, _ALLOWED_NAMES)  # noqa: S307
    except Exception as exc:
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"})
    return json.dumps({"expression": expression, "result": result}, default=str)
