"""
工具系统 — 工具定义、注册中心与执行器（Anthropic 协议格式）。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass
class Tool:
    """一个可由 Agent 调用的工具。"""

    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema properties
    fn: Callable[..., str]

    def to_param(self) -> dict[str, Any]:
        """转为 Anthropic tool 定义格式。"""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": self.parameters,
                "required": list(self.parameters.keys()),
            },
        }

    def execute(self, **kwargs: Any) -> str:
        """执行工具并返回字符串结果。"""
        try:
            return self.fn(**kwargs)
        except Exception as e:
            return f"Tool execution error: {e}"


class ToolRegistry:
    """工具注册中心 — 管理工具集合并分发执行。"""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def to_params(self) -> list[dict[str, Any]]:
        return [t.to_param() for t in self._tools.values()]

    def execute(self, name: str, input_data: dict[str, Any]) -> str:
        tool = self._tools.get(name)
        if tool is None:
            return f"Unknown tool: {name}"
        return tool.execute(**input_data)


# ── 内置示例工具 ──────────────────────────────────────────────


def _calculator(expression: str) -> str:
    """安全的数学表达式计算器。"""
    allowed = set("0123456789+-*/().%^ eEpiPI")
    sanitized = "".join(c for c in expression if c in allowed)
    if not sanitized:
        return "Error: expression is empty or contains only invalid characters."
    try:
        result = eval(sanitized, {"__builtins__": {}}, {})
        return f"Result: {result}"
    except Exception as e:
        return f"Error evaluating expression: {e}"


def _get_current_time(format: str = "%Y-%m-%d %H:%M:%S") -> str:
    """返回当前时间。"""
    import datetime
    return datetime.datetime.now().strftime(format)


def create_default_registry() -> ToolRegistry:
    """创建预置了常用工具的注册中心。"""
    registry = ToolRegistry()
    registry.register(Tool(
        name="calculator",
        description="Evaluate a mathematical expression. Supports +, -, *, /, (), **, %.",
        parameters={
            "expression": {
                "type": "string",
                "description": "The mathematical expression to evaluate, e.g. '2 + 3 * 4'",
            }
        },
        fn=_calculator,
    ))
    registry.register(Tool(
        name="get_current_time",
        description="Get the current date and time.",
        parameters={
            "format": {
                "type": "string",
                "description": "strftime format string, default is '%Y-%m-%d %H:%M:%S'",
            }
        },
        fn=_get_current_time,
    ))
    return registry
