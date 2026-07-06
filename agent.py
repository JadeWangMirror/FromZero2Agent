"""
核心 Agent — ReAct 循环 + 流式输出 + 对话历史。
协议：Anthropic Messages API（DeepSeek 兼容端点）。
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from anthropic.types import MessageParam

from llm import LLMClient, StreamEvent
from tools import ToolRegistry

# ── System Prompt ──────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a helpful AI assistant with access to tools. Follow these rules:

1. When you need information that a tool can provide, use the tool immediately — do not guess.
2. After receiving a tool result, reason about it and decide your next step.
3. When you have enough information to answer the user, respond directly in natural language.
4. If a tool returns an error, explain the problem to the user and suggest an alternative.
5. Always respond in the same language the user used.
"""

# ── 回调类型 ───────────────────────────────────────────────

StepCallback = Callable[[str, dict[str, Any]], None]


# ── Agent ──────────────────────────────────────────────────


class Agent:
    """基于 ReAct 模式的 LLM Agent，支持流式 thinking + 对话历史。"""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = "https://api.deepseek.com/anthropic",
        model: str = "deepseek-v4-pro",
        max_tokens: int = 4096,
        max_turns: int = 10,
        tools: ToolRegistry | None = None,
        max_history: int = 50,
    ):
        self.llm = LLMClient(
            api_key=api_key,
            base_url=base_url,
            model=model,
            max_tokens=max_tokens,
        )
        self.tools = tools
        self.max_turns = max_turns
        self._history: list[MessageParam] = []
        self._max_history = max_history

    def reset_history(self) -> None:
        """清空对话历史。"""
        self._history.clear()

    def run(self, task: str, callback: StepCallback | None = None) -> str:
        """执行 Agent 对话（流式 thinking，自动维护上下文）。

        callback 事件:
          "thinking"     → {"text": str}      # 思考增量（流式逐 token）
          "tool_call"    → {"name": str, "args": dict}
          "tool_result"  → {"name": str, "result": str}
          "text"         → {"text": str}      # 最终回复
        """
        # 从历史 + 当前用户消息开始
        messages: list[MessageParam] = list(self._history)
        messages.append({"role": "user", "content": task})

        tool_params = self.tools.to_params() if self.tools else None

        for _turn in range(self.max_turns):
            # ── 流式调用 + 实时回调 ──
            def on_stream(ev: StreamEvent) -> None:
                if ev.type == "thinking_delta" and callback:
                    callback("thinking", {"text": ev.text})

            response = self.llm.send_stream(
                messages=messages,
                tools=tool_params,
                system=SYSTEM_PROMPT,
                on_event=on_stream,
            )

            # ── 解析完整响应 ──
            text_parts: list[str] = []
            tool_uses: list[dict] = []

            for block in response.content:
                t = block.type
                if t == "text":
                    text_parts.append(block.text)
                elif t == "tool_use":
                    tool_uses.append({
                        "id": block.id,
                        "name": block.name,
                        "input": block.input,
                    })

            # 构建 assistant 消息
            assistant_content: list[dict] = []
            for block in response.content:
                t = block.type
                if t == "text":
                    assistant_content.append({"type": "text", "text": block.text})
                elif t == "tool_use":
                    assistant_content.append({
                        "type": "tool_use",
                        "id": block.id,
                        "name": block.name,
                        "input": block.input,
                    })

            messages.append({"role": "assistant", "content": assistant_content})

            # 无工具调用 → 最终回复，保存历史
            if not tool_uses:
                final_text = "\n".join(text_parts)
                if callback:
                    callback("text", {"text": final_text})
                # 保存本轮到历史
                self._save_history(messages)
                return final_text

            # 执行工具
            tool_results: list[dict] = []
            for tu in tool_uses:
                if callback:
                    callback("tool_call", {"name": tu["name"], "args": tu["input"]})

                if self.tools is None:
                    result = "No tools available."
                else:
                    result = self.tools.execute(tu["name"], tu["input"])

                if callback:
                    callback("tool_result", {"name": tu["name"], "result": result})

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tu["id"],
                    "content": result,
                })

            messages.append({"role": "user", "content": tool_results})

        return "Agent reached maximum turns without a final response."

    def _save_history(self, messages: list[MessageParam]) -> None:
        """将本轮完整消息链保存到历史，并裁剪超长历史。"""
        # messages 已包含历史前缀 + 本轮所有消息
        # 直接替换历史为完整消息链
        self._history = list(messages)
        # 裁剪：保留最近 N 条
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]


# ── 便捷函数 ───────────────────────────────────────────────

def create_agent(api_key: str | None = None, **kwargs) -> Agent:
    from tools import create_default_registry
    registry = create_default_registry()
    return Agent(api_key=api_key, tools=registry, **kwargs)
