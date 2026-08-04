"""Small deterministic calculator for ordinary Agent computations.

This is deliberately an AST allow-list rather than ``eval``.  The tool does
not access the network or workspace and returns the expression/result pair so
RWKV can explain the calculation without doing the arithmetic itself.
"""

from __future__ import annotations

import ast
import json
import math
from typing import Any

from tools.registry import ToolRegistry


_BINARY_OPS = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.FloorDiv: lambda a, b: a // b,
    ast.Mod: lambda a, b: a % b,
    ast.Pow: lambda a, b: a ** b,
}
_UNARY_OPS = {ast.UAdd: lambda value: +value, ast.USub: lambda value: -value}
_FUNCTIONS = {"abs": abs, "ceil": math.ceil, "floor": math.floor, "round": round, "sqrt": math.sqrt}


def _evaluate(node: ast.AST) -> int | float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        if not math.isfinite(float(node.value)):
            raise ValueError("non-finite numbers are not allowed")
        return node.value
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_evaluate(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPS:
        left = _evaluate(node.left)
        right = _evaluate(node.right)
        result = _BINARY_OPS[type(node.op)](left, right)
        if isinstance(result, (int, float)) and not math.isfinite(float(result)):
            raise ValueError("result is not finite")
        return result
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCTIONS:
        if node.keywords or len(node.args) not in (1, 2):
            raise ValueError("calculator function arguments are invalid")
        return _FUNCTIONS[node.func.id](*[_evaluate(arg) for arg in node.args])
    raise ValueError("expression contains an unsupported operation")


@ToolRegistry.register(
    name="calculator",
    phase="ALL",
    model_visible=True,
    category="computation",
    description="Safely evaluate a complete numeric expression; use it for arithmetic after all operands are known, not for web facts.",
    signature="""[Tool] calculator
- Function: deterministically evaluate a small arithmetic expression.
- Parameters: expression (numbers, parentheses, + - * / // % **, abs/ceil/floor/round/sqrt).
- It never searches, reads files, executes code, or invents missing operands.""",
)
def calculator(expression: str, **_: Any) -> str:
    expression = str(expression or "").strip()
    if not expression or len(expression) > 500:
        return json.dumps(
            {"status": "error", "tool": "calculator", "error_class": "invalid_expression", "message": "expression is empty or too long"},
            ensure_ascii=False,
        )
    try:
        tree = ast.parse(expression, mode="eval")
        result = _evaluate(tree.body)
    except (SyntaxError, TypeError, ValueError, ZeroDivisionError, OverflowError) as exc:
        return json.dumps(
            {"status": "error", "tool": "calculator", "expression": expression, "error_class": "invalid_expression", "message": str(exc)[:300]},
            ensure_ascii=False,
        )
    return json.dumps(
        {
            "status": "ok",
            "tool": "calculator",
            "expression": expression,
            "result": result,
            "formatted_result": format(result, ".15g") if isinstance(result, float) else str(result),
            "deterministic": True,
            "source_refs": [],
        },
        ensure_ascii=False,
    )


__all__ = ["calculator"]
