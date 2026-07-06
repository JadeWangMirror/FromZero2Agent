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
You are MIRROR, a SELF-EVOLVING AI agent. You solve tasks with tools, DELEGATE to \
sub-agents, and — your defining ability — EXTEND YOURSELF by building new tools \
when justified. Crucially, you know WHEN to build and when NOT to.

CORE RULES
1. Use a tool for anything a tool can do — never guess facts you can look up or compute.
2. After each tool result, reason about it, then decide the next step.
3. When you have the answer, respond directly in the user's language.
4. On tool error, diagnose the cause and try an alternative.

TOOL SELECTION
- Pick the most specific tool (e.g. read_file, not run_python, for reading).
- If unsure what exists, call list_tools or find_similar_tools("<capability>").

═══════════════════════════════════════════════════════════
SELF-EVOLUTION — the disciplined way to decide about new tools
═══════════════════════════════════════════════════════════
Never jump straight to create_tool. Follow this decision protocol:

STEP 1 — ASSESS (mandatory before building):
  Call propose_tool(capability, reuse_signal) OR self_evolve(goal).
  It runs a deterministic check and returns a VERDICT: BUILD or SKIP, with reason.
  - propose_tool already checks for duplicate existing tools (find_similar).
  - reuse_signal: "once" (one-off) | "few" | "recurring" (default).

STEP 2 — RESPECT THE VERDICT:
  • SKIP because an existing tool covers it → USE that tool. Do not rebuild.
  • SKIP because reuse_signal="once" → use run_python. One-offs must not become tools.
  • BUILD → proceed to STEP 3.

STEP 3 — BUILD (only when verdict is BUILD):
  create_tool(name, description, parameters, code, test_code):
  - description must state WHEN to use it (it drives future selection).
  - ALWAYS include test_code — a tool without a test is unverified.
  - Define `execute(**kwargs) -> str`; wrap risky logic in try/except.
  - Compose with existing tools via `use("tool_name", **kwargs)` instead of \
reimplementing what exists.
  - Keep it minimal — one clear responsibility.
  After create_tool succeeds, the tool is LIVE and immediately callable — just call \
it by name like any tool to try it. Do NOT try to import its .py file via run_python.

STEP 4 — VERIFY & ITERATE:
  After creating: call review_tool(name). Fix every [!] WARN via update_tool, \
then re-review. Loop until clean.
  On a runtime failure: improve_tool(name, failure=<error>) → update_tool.

STEP 5 — MAINTAIN (occasionally):
  Call tool_stats(detail="unused") to find dead tools, then delete_tool them.

WHEN UNSURE whether building is worth it, delegate to spawn_agent("tool_designer") \
— it runs the full assess→build→verify flow in isolation and reports.

═══════════════════════════════════════════════════════════
DELEGATION — spawn_agent(role, task) for complex, self-contained work
═══════════════════════════════════════════════════════════
- researcher: look up multiple things online (independent context).
- planner: decompose a big task before executing.
- coder: implement + verify code in isolation.
- critic: review a plan or code for problems.
- tool_designer: assess + build + verify a tool (the self-evolution specialist).
- general: anything self-contained.
Delegate when a sub-task would clutter the main conversation; keep simple work inline.

Always verify code with a tool before claiming it works. Prefer concrete results \
over assumptions.
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
        temperature: float = 1.0,
        system_prompt: str | None = None,
        toolforge=None,
    ):
        self.llm = LLMClient(
            api_key=api_key,
            base_url=base_url,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        self.tools = tools
        self.toolforge = toolforge
        self.max_turns = max_turns
        self._history: list[MessageParam] = []
        self._max_history = max_history
        # 自定义 system prompt，空则用内置默认
        self.system_prompt = system_prompt.strip() if system_prompt else None
        # 层次化 agent
        self._depth = 0
        self._max_depth = 3
        self._current_callback: StepCallback | None = None
        # 上下文压缩
        self._compress_threshold = 20_000   # 字符数阈值
        self._keep_recent = 6               # 压缩时保留最近消息数
        # 单个工具结果回传给 LLM 的字符上限（超出截断，保护上下文）
        self._max_tool_result_chars = 30_000

    def _build_system_prompt(self) -> str:
        """构建 system prompt：基础指导（或用户自定义）+ 动态工具清单。"""
        base = self.system_prompt if self.system_prompt else SYSTEM_PROMPT
        if not self.tools:
            return base
        lines = ["", "AVAILABLE TOOLS:"]
        for name, t in self.tools._tools.items():
            desc = t.description.split("\n")[0].strip()
            if len(desc) > 90:
                desc = desc[:87] + "..."
            lines.append(f"  - {name}: {desc}")
        return base + "\n" + "\n".join(lines)

    # ── 层次化:派生子 agent ───────────────────────────────

    def spawn_as_tool(self, role: str, task: str, max_turns: int = 8) -> str:
        """作为工具被调用:派生一个专职子 agent 执行子任务。

        子 agent 独立上下文、共享工具集、专用 system prompt。
        中间事件通过 'sub:*' 转发给当前 callback。
        """
        from subagents import get_prompt, role_names

        if self._depth >= self._max_depth:
            return (f"Error: max agent nesting depth ({self._max_depth}) reached. "
                    f"Complete the work at this level instead of spawning more.")
        role = role.lower().strip()
        prompt = get_prompt(role)
        if prompt is None:
            return (f"Error: unknown role '{role}'. Available: {', '.join(role_names())}")

        sub = Agent(
            api_key=self.llm.api_key,
            base_url=self.llm.base_url,
            model=self.llm.model,
            max_tokens=self.llm.max_tokens,
            temperature=self.llm.temperature,
            max_turns=max_turns,
            tools=self.tools,
            toolforge=self.toolforge,
            system_prompt=prompt,
        )
        sub._depth = self._depth + 1
        sub_cb = self._make_sub_callback(role)
        try:
            result = sub.run(task, callback=sub_cb)
        except Exception as e:
            return f"Sub-agent '{role}' failed: {e}"
        return f"[{role}] {result}"

    def _make_sub_callback(self, role: str) -> StepCallback:
        """构造子 agent 回调,把事件加 role 前缀转发给主 callback。"""
        def cb(ev: str, data: dict) -> None:
            if self._current_callback:
                self._current_callback(f"sub:{ev}", {"role": role, **data})
        return cb

    def reset_history(self) -> None:
        """清空对话历史。"""
        self._history.clear()

    def compact(self) -> tuple[int, int]:
        """手动触发上下文压缩。返回 (压缩前消息数, 压缩后消息数)。"""
        before = len(self._history)
        self._maybe_compress(force=True)
        return before, len(self._history)

    def context_info(self) -> dict:
        """返回当前上下文规模信息（供状态条/命令使用）。"""
        msgs = self._history
        chars = sum(self._msg_size(m) for m in msgs)
        return {
            "messages": len(msgs),
            "chars": chars,
            "max_history": self._max_history,
            "threshold": self._compress_threshold,
        }

    def estimate_context_tokens(self) -> int:
        """估算即将发送的上下文 token 数：system(含工具表)+历史，~4 字符/token。"""
        sys_chars = len(self._build_system_prompt())
        hist_chars = sum(self._msg_size(m) for m in self._history)
        return (sys_chars + hist_chars) // 4

    def context_tokens(self) -> int:
        """当前上下文占用（token）：取「最近一次实际用量」与「实时估算」的较大值，
        保证进度条在请求前后、历史增长时都持续反映真实占用。"""
        actual = getattr(self.llm, "last", {}).get("input", 0) if self.llm else 0
        return max(actual or 0, self.estimate_context_tokens())

    def export_history(self) -> list[dict]:
        """导出对话历史为可序列化列表（用于持久化）。"""
        return [dict(m) for m in self._history]

    def import_history(self, data: list[dict]) -> None:
        """从列表导入对话历史（覆盖现有历史）。"""
        self._history = [dict(m) for m in data]  # type: ignore[arg-type]

    def run(self, task: str, callback: StepCallback | None = None) -> str:
        """执行 Agent 对话（流式 thinking，自动维护上下文）。

        callback 事件:
          "thinking"     → {"text": str}      # 思考增量（流式逐 token）
          "text_delta"   → {"text": str}      # 最终回复增量（流式逐 token）
          "tool_call"    → {"name": str, "args": dict}
          "tool_result"  → {"name": str, "result": str}
          "text"         → {"text": str}      # 最终回复完整文本
          "sub:*"        → 子 agent 事件,{"role":..., ...}
        """
        self._current_callback = callback
        # 上下文过大则先压缩历史
        self._maybe_compress()
        # 从历史 + 当前用户消息开始
        messages: list[MessageParam] = list(self._history)
        messages.append({"role": "user", "content": task})

        tool_params = self.tools.to_params() if self.tools else None

        for _turn in range(self.max_turns):
            if callback:
                callback("turn", {"turn": _turn + 1, "max": self.max_turns})
            # ── 流式调用 + 实时回调 ──
            def on_stream(ev: StreamEvent) -> None:
                if ev.type == "thinking_delta" and callback:
                    callback("thinking", {"text": ev.text})
                elif ev.type == "text_delta" and callback:
                    callback("text_delta", {"text": ev.text})

            response = self.llm.send_stream(
                messages=messages,
                tools=tool_params,
                system=self._build_system_prompt(),
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
                    "content": self._truncate_result(result),
                })

            messages.append({"role": "user", "content": tool_results})

        # 达到最大轮次仍未结束：强制一次无工具收尾，让模型总结已完成的成果
        wrap = self.llm.send_stream(
            messages=messages,
            tools=None,
            system=self._build_system_prompt()
                + "\n\nYou have reached the tool-call turn limit. Stop calling tools and "
                  "now give the user a concise final answer: summarize what you accomplished "
                  "and, if incomplete, state precisely what remains.",
            on_event=lambda ev: (
                callback("text_delta", {"text": ev.text})
                if callback and ev.type == "text_delta" else None
            ),
        )
        final_text = "".join(
            getattr(b, "text", "") for b in wrap.content
            if getattr(b, "type", "") == "text"
        )
        if callback:
            callback("text", {"text": final_text})
        self._save_history(messages + [{"role": "assistant", "content": [
            {"type": "text", "text": final_text}]}])
        return final_text or "Agent reached maximum turns; see tool output above."

    def _save_history(self, messages: list[MessageParam]) -> None:
        """将本轮完整消息链保存到历史，并裁剪超长历史。"""
        # messages 已包含历史前缀 + 本轮所有消息
        # 直接替换历史为完整消息链
        self._history = list(messages)
        # 裁剪：保留最近 N 条
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]

    # ── 上下文自动压缩 ─────────────────────────────────────

    def _maybe_compress(self, force: bool = False) -> None:
        """历史过大时，把旧消息摘要化，保留最近几轮原文。"""
        if len(self._history) <= self._keep_recent + 2:
            return
        total = sum(self._msg_size(m) for m in self._history)
        if not force and total < self._compress_threshold:
            return
        old = self._history[:-self._keep_recent]
        recent = self._history[-self._keep_recent:]
        try:
            summary = self._summarize(old)
        except Exception:
            return  # 压缩失败不影响主流程
        summary_msg: MessageParam = {
            "role": "user",
            "content": f"[Earlier conversation summary]\n{summary}",
        }
        self._history = [summary_msg] + list(recent)

    def _truncate_result(self, result: str) -> str:
        """工具结果过长则截断，避免单条结果撑爆上下文。"""
        if not isinstance(result, str):
            result = str(result)
        cap = self._max_tool_result_chars
        if len(result) <= cap:
            return result
        return result[:cap] + f"\n\n... (truncated, {len(result) - cap} more chars omitted)"

    @staticmethod
    def _msg_size(m) -> int:
        c = m.get("content") if isinstance(m, dict) else None
        if isinstance(c, str):
            return len(c)
        if isinstance(c, list):
            return sum(len(str(b)) for b in c)
        return 0

    def _messages_to_text(self, messages) -> str:
        out = []
        for m in messages:
            role = m.get("role", "?") if isinstance(m, dict) else "?"
            c = m.get("content") if isinstance(m, dict) else ""
            if isinstance(c, str):
                out.append(f"[{role}] {c}")
            elif isinstance(c, list):
                parts = []
                for b in c:
                    if not isinstance(b, dict):
                        parts.append(str(b))
                        continue
                    t = b.get("type", "")
                    if t == "text":
                        parts.append(b.get("text", ""))
                    elif t == "tool_use":
                        parts.append(f"(tool_use {b.get('name')} {b.get('input')})")
                    elif t == "tool_result":
                        parts.append(f"(tool_result {b.get('content', '')})")
                out.append(f"[{role}] " + " ".join(parts))
        return "\n".join(out)

    def _summarize(self, old_messages) -> str:
        text = self._messages_to_text(old_messages)
        if len(text) > 12000:
            text = text[:12000] + "\n... (truncated for summarization)"
        prompt = [{
            "role": "user",
            "content": (
                "Summarize the conversation below. Preserve: key facts, user "
                "preferences, decisions made, code/files touched, important tool "
                "results, and any reusable context. Be concise but lossless on "
                "technical details.\n\n" + text
            ),
        }]
        resp = self.llm.send(
            prompt,
            system="You compress conversation history. Concise, no preamble.",
        )
        return "".join(
            getattr(b, "text", "") for b in resp.content
            if getattr(b, "type", "") == "text"
        )


# ── 便捷函数 ───────────────────────────────────────────────

def create_agent(api_key: str | None = None, config=None, **kwargs) -> Agent:
    """创建带完整工具集 + 自我完善能力的 Agent。

    注册顺序: 基础工具 → 文件工具 → 网络工具 → 自造工具(load) → 元工具。
    """
    from filetools import create_file_tools
    from toolforge import ToolForge
    from tools import ToolRegistry, create_default_registry
    from webtools import create_web_tools

    registry = create_default_registry()
    for tool in create_file_tools():
        registry.register(tool)
    for tool in create_web_tools():
        registry.register(tool)

    # 自我完善系统:加载已有自造工具 + 注册元工具
    forge = ToolForge(registry)
    forge.load_existing()
    # 接通使用统计：每次工具调用都记一笔
    registry._stats_sink = forge.record_usage
    for tool in forge.get_meta_tools():
        registry.register(tool)

    # 从 config 提取字段作为默认值
    if config is not None:
        cfg_fields = {
            "model", "base_url", "max_tokens", "max_turns",
            "max_history", "temperature", "system_prompt",
        }
        for f in cfg_fields:
            kwargs.setdefault(f, getattr(config, f))

    agent = Agent(api_key=api_key, tools=registry, toolforge=forge, **kwargs)

    # 层次化:注册 spawn_agent 工具(绑定到该 agent 实例)
    from tools import Tool
    registry.register(Tool(
        "spawn_agent",
        "Spawn a specialized sub-agent for a sub-task. Use for: research (web), "
        "task decomposition, coding+verification, critique, or deciding+building a tool. "
        "Sub-agent runs with its own context and shared tools. "
        "roles: researcher | planner | coder | critic | tool_designer | general. "
        "Prefer this over doing everything inline for complex multi-step work.",
        {"role": {"type": "string", "description": "researcher|planner|coder|critic|tool_designer|general"},
         "task": {"type": "string", "description": "clear, self-contained sub-task description"},
         "max_turns": {"type": "integer", "description": "default 8"}},
        agent.spawn_as_tool,
        required=["role", "task"],
    ))

    return agent
