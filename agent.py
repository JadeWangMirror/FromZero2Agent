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
SELF-EVOLUTION — you decide what to build, and you build it
═══════════════════════════════════════════════════════════
PROACTIVITY — you are not passive. This is what makes you MIRROR, not a chatbot:
- A ⚡ capability gap in MEMORY is YOUR to-do, not background info. The turn it
  appears, call self_evolve() (no args) to assess it; if VERDICT is BUILD, build
  the tool that same turn. Deferring a real recurring gap is a failure mode — it
  means you keep hand-rolling the same workaround forever.
- Memory folds ITSELF when your struggle accumulates (you don't have to call
  consolidate_memory unless asked). When a gap self-emerges from that, ACT on it.
- Record durable facts the instant you confirm them: corrections, preferences,
  environment/project conventions. Call remember() right then — never defer or
  batch memory writes; late or missing memory is as bad as no memory.
Silence in the face of a surfaced gap is the one thing you must not do.

Self-evolve AUTONOMOUSLY — do not wait for the user to describe a tool. Two
self-detected signals trigger you, and you run the full assess→build→verify loop
on your own authority:
 (a) SELF-REPETITION — a ⚠ SELF-EVOLUTION SIGNAL appears when you repeat an
     operation (near-identical run_python twice, or one tool 3+ times).
 (b) CAPABILITY GAPS — ⚡ intents in MEMORY: recurring needs no current tool covers.
On either signal, call self_evolve() (no args → it assesses memory's gaps). It
returns a deterministic BUILD/SKIP verdict. If BUILD, write create_tool yourself
(name, schema, code, test_code), then review_tool until clean. You own this
decision; do not ask the user to specify the tool.
Never jump straight to create_tool WITHOUT running the verdict first — the
protocol IS the judgment. Then follow it:

STEP 1 — ASSESS (mandatory before building):
  Call propose_tool(capability, reuse_signal) OR self_evolve(goal).
  It runs a deterministic check and returns a VERDICT: BUILD or SKIP, with reason.
  - propose_tool already checks for duplicate existing tools (find_similar).
  - reuse_signal: "once" (one-off) | "few" | "recurring" (default).

STEP 2 — RESPECT THE VERDICT:
  • SKIP because an existing tool covers it → USE that tool. Do not rebuild.
  • SKIP because reuse_signal="once" → use run_python. One-offs must not become tools.
  • BUILD → proceed to STEP 2.5.

STEP 2.5 — REUSE FIRST (mandatory before writing any tool code):
  Do NOT reinvent the wheel. Before create_tool, ACTIVELY look for an existing solution:
  (a) web_search "<capability> python library" (and "<capability> pypi"); skim top results.
  (b) web_fetch the candidate's README/docs if unclear.
  (c) If a mature, maintained package exists, PREFER WRAPPING IT — create_tool whose
      execute() imports and calls that package — over reimplementing from scratch.
      Battle-tested open-source beats hand-rolled code on correctness, edge cases, and
      maintenance. Record the dependency in the tool description.
  (d) Only write from scratch if no good existing solution exists (and say why you searched
      and found nothing).
  This step is non-negotiable: building a tool without first searching for an existing
  solution is a process failure.

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
MEMORY — emergent, NOT retrieved
═══════════════════════════════════════════════════════════
Your memory is a concept graph (hippocampal events -> neocortical concepts ->
prefrontal intents), read by graph topology each turn. You do NOT query it.
- remember(content, tags): commit ONE event — corrections, durable facts,
  preferences, environment notes, project conventions. Record it the INSTANT you
  confirm it; never defer or batch. This is raw material.
- consolidate_memory(): FOLD the graph. Clusters of recurring events abstract
  into concepts; strongly-supported concepts CRYSTALLIZE into intents that
  surface on their own. Run it after adding several events, or when the MEMORY
  block shows unconsolidated events piling up. Understanding emerges here —
  without it, events stay raw forever.
- The MEMORY block you see each turn is the graph's emergent state (PageRank +
  recency), NOT a search result. Intents listed there were never asked for;
  they crystallized because the topology accumulated the conditions for them.
  Honor surfaced intents: act on pending ones, then update_intent(id, status).
- update_intent(intent_id, status): mark an intent in_progress/done/skipped.
- Intents tagged ⚡ (capability gap) are recurring operations no current tool
  covers — prime self_evolve/propose_tool candidates. Building the tool
  (create_tool) auto-satisfies the gap: memory is the demand sensor, ToolForge
  the supply. This is how you notice your own repetitive work and close the loop.
Do not try to read/search memory — it is already in front of you each turn.

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
        """构建 system prompt：基础指导（或用户自定义）+ 持久记忆 + 动态工具清单。"""
        base = self.system_prompt if self.system_prompt else SYSTEM_PROMPT
        # 自动注入涌现式记忆（读图当前状态，无 query —— 见 memory.select_context）
        try:
            from memory import context_block
            mem = context_block()
            if mem:
                base = base + "\n" + mem
        except Exception:
            pass
        # 自主自进化信号:agent 自己发现的重复操作(廉价预筛,零 LLM)
        try:
            sig = self._detect_repetition()
            if sig:
                base = base + "\n\n⚠ SELF-EVOLUTION SIGNAL (self-detected): " + sig
        except Exception:
            pass
        # 待反思信号攒够 → 提醒 consolidate_memory(否则边界永远不浮现)
        try:
            from memory import pending_signals
            n = pending_signals()
            if n >= 2:
                base += (f"\n\n⚠ {n} struggle signals (failures / manual workarounds) "
                         f"queued. Memory auto-folds these into capability gaps; the turn "
                         f"a ⚡ gap surfaces above, run self_evolve() to assess building a "
                         f"tool for it — do not defer.")
        except Exception:
            pass
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

        异常安全:无论正常结束还是中途抛错(API 400 / 超时 / 工具异常…),
        都把已积累的对话落盘 —— 不正常结束的对话也能恢复,下一轮不会失忆,
        也不会因「孤儿 tool_use」触发 400。
        """
        self._current_callback = callback
        self._maybe_compress()
        self._maybe_auto_consolidate()
        messages: list[MessageParam] = list(self._history)
        messages.append({"role": "user", "content": task})
        try:
            return self._run_loop(messages, callback)
        finally:
            # 兜底落盘:即便 _run_loop 抛错,也把已积累的 messages 持久化。
            # _finalize_interrupted 先把半截对话补成合法形态(孤儿 tool_use 补结果、
            # 末尾补 assistant 保轮替),否则下次请求会因 tool_use 无对应 result /
            # 角色不轮替而 400 —— 这正是「断开后下一轮失忆 + 反复 tool_use 400」的根因。
            try:
                self._finalize_interrupted(messages)
                self._save_history(messages)
            except Exception:
                pass

    def _run_loop(self, messages: list[MessageParam],
                  callback: StepCallback | None) -> str:
        """ReAct 主循环 + max_turns 收尾。历史持久化统一由 run() 的 finally 兜底。"""
        for _turn in range(self.max_turns):
            if callback:
                callback("turn", {"turn": _turn + 1, "max": self.max_turns})
            # 工具表每轮重新快照：本 run 中途通过 create_tool/update_tool/delete_tool
            # 变动的工具，下一轮立刻对模型可见（schema 数组与 system prompt 工具清单同步，
            # 实现"建完即调"）。to_params() 实时读 registry，开销可忽略。
            tool_params = self.tools.to_params() if self.tools else None
            # ── 流式调用 + 实时回调 ──
            def on_stream(ev: StreamEvent) -> None:
                if ev.type == "thinking_delta" and callback:
                    callback("thinking", {"text": ev.text})
                elif ev.type == "text_delta" and callback:
                    callback("text_delta", {"text": ev.text})

            response = self.llm.send_stream(
                messages=self._sanitize_messages(messages),
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

            # 无工具调用 → 最终回复（历史由 run() 的 finally 落盘）
            if not tool_uses:
                final_text = "\n".join(text_parts)
                if callback:
                    callback("text", {"text": final_text})
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
                # 捕获挣扎信号 → 记忆图(能力边界的原料,见 memory.reflect)
                self._maybe_record_signal(tu["name"], tu.get("input"), result)

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
            messages=self._sanitize_messages(messages),
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
        messages.append({"role": "assistant", "content": [
            {"type": "text", "text": final_text}]})
        return final_text or "Agent reached maximum turns; see tool output above."

    def _save_history(self, messages: list[MessageParam]) -> None:
        """将本轮完整消息链保存到历史，并裁剪超长历史。"""
        # messages 已包含历史前缀 + 本轮所有消息
        # 直接替换历史为完整消息链
        self._history = self._sanitize_messages(list(messages))
        # 裁剪：保留最近 N 条（在安全边界切，避免切断 tool_use/tool_result 对）
        if len(self._history) > self._max_history:
            self._history = self._safe_tail(self._history, self._max_history)

    @staticmethod
    def _finalize_interrupted(messages: list[MessageParam]) -> None:
        """把异常中断的半截对话补成 API 合法形态,保证落盘后下次请求不 400。

        两种需要修补的结尾:
          (1) 末尾是 assistant 且含 tool_use,但没有跟随的 tool_result
              → 补占位 tool_result(否则 DeepSeek 报「tool_use must be followed
                by tool_result」→ 400 断开)。
          (2) 末尾是 user(无回复的文本 / 刚补的 tool_result)
              → 补一条 assistant 占位,维持 user/assistant 轮替。
        末尾已是干净的 assistant 纯文本则不动。"""
        if not messages:
            return
        last = messages[-1]
        if not isinstance(last, dict):
            return
        if last.get("role") == "assistant":
            content = last.get("content")
            tids = ([b.get("id", "") for b in content
                     if isinstance(b, dict) and b.get("type") == "tool_use"]
                    if isinstance(content, list) else [])
            if not tids:
                return  # 纯文本 assistant,已是合法结尾
            messages.append({"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": tid,
                 "content": "(turn interrupted by an error — no result captured)"}
                for tid in tids
            ]})
        # 末尾为 user(文本或刚补的 tool_result)→ 补 assistant 占位保轮替
        messages.append({"role": "assistant", "content": [
            {"type": "text",
             "text": "[this turn ended abnormally due to an error — context was "
                     "preserved so the conversation can continue from here.]"}
        ]})

    # ── 历史完整性 ───────────────────────────────────────

    @staticmethod
    def _is_text_user(m) -> bool:
        """该消息是否为"干净的 user 文本回合"（可作为历史片段的安全起点）。"""
        if not isinstance(m, dict) or m.get("role") != "user":
            return False
        c = m.get("content")
        if isinstance(c, str):
            return True
        if isinstance(c, list):
            return not any(isinstance(b, dict) and b.get("type") == "tool_result"
                           for b in c)
        return False

    @classmethod
    def _safe_tail(cls, msgs: list, n: int) -> list:
        """取最后 n 条，但左边界右移到第一条干净的 user-text 消息，
        防止从 tool_result 起头（会产生孤儿 tool_result → API 400）。"""
        tail = list(msgs[-n:]) if n < len(msgs) else list(msgs)
        while tail and not cls._is_text_user(tail[0]):
            tail.pop(0)
        return tail

    @staticmethod
    def _sanitize_messages(msgs: list) -> list:
        """删除孤儿 tool_result：其 tool_use_id 在前一条 assistant 消息里不存在。

        DeepSeek/Anthropic 严格要求每个 tool_result 有对应的 tool_use，否则 400：
        'tool_result must have a corresponding tool_use block in the previous message'。
        """
        out: list = []
        prev_ids: set = set()
        for m in msgs:
            role = m.get("role", "user") if isinstance(m, dict) else "user"
            content = m.get("content") if isinstance(m, dict) else m
            if role == "assistant":
                if isinstance(content, list):
                    prev_ids = {b.get("id") for b in content
                                if isinstance(b, dict) and b.get("type") == "tool_use"}
                out.append(m)
                continue
            # user：过滤掉 tool_use_id 不在前一条里的 tool_result
            if isinstance(content, list):
                kept = [b for b in content
                        if not (isinstance(b, dict) and b.get("type") == "tool_result"
                                and b.get("tool_use_id") not in prev_ids)]
                if not kept:
                    prev_ids = set()
                    continue          # 整条都是孤儿 → 丢弃，避免空消息
                out.append({"role": "user", "content": kept})
            else:
                out.append(m)
            prev_ids = set()
        return out

    # ── 上下文自动压缩 ─────────────────────────────────────

    def _maybe_compress(self, force: bool = False) -> None:
        """历史过大时，把旧消息摘要化，保留最近几轮原文。"""
        if len(self._history) <= self._keep_recent + 2:
            return
        total = sum(self._msg_size(m) for m in self._history)
        if not force and total < self._compress_threshold:
            return
        old = self._history[:-self._keep_recent]
        # recent 必须在安全边界起头，否则 summary(user) + orphan tool_result → 400
        recent = self._safe_tail(self._history, self._keep_recent)
        try:
            summary = self._summarize(old)
        except Exception:
            return  # 压缩失败不影响主流程
        summary_msg: MessageParam = {
            "role": "user",
            "content": f"[Earlier conversation summary]\n{summary}",
        }
        self._history = self._sanitize_messages([summary_msg] + list(recent))

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

    # ── 涌现式记忆:折叠 ───────────────────────────────────

    def _llm_summarize(self, user_prompt: str,
                       system: str = "You abstract concise concepts. No preamble.") -> str:
        """非流式 LLM 调用，返回纯文本（供记忆折叠复用）。失败返回空串。"""
        try:
            resp = self.llm.send(
                [{"role": "user", "content": user_prompt}], system=system,
            )
            return "".join(getattr(b, "text", "") for b in resp.content
                           if getattr(b, "type", "") == "text")
        except Exception:
            return ""

    # ── 自主自进化:重复检测(最强的"该造工具"信号) ────────

    def _detect_repetition(self) -> str | None:
        """扫描近期历史,若 agent 自己在重复同一操作,返回自进化触发信号。

        廉价预筛(纯 token 数学,零 LLM 消耗)—— 只负责"举手示意"找到候选;
        真正的"该不该造、造什么样的"判断交给 self_evolve(propose_tool 的确定性
        决策 + 必要时 LLM)。这是"真自进化"的入口:agent 自己发现重复,而非用户描述。
        """
        import re as _re
        _T = _re.compile(r"[a-z0-9一-鿿]+")

        def tok(s: str) -> set[str]:
            return {t for t in _T.findall((s or "").lower()) if len(t) > 1}

        def jac(a: set, b: set) -> float:
            return len(a & b) / len(a | b) if a and b else 0.0

        codes: list[str] = []
        name_counts: dict[str, int] = {}
        for m in self._history[-40:]:
            if not isinstance(m, dict):
                continue
            c = m.get("content")
            if not isinstance(c, list):
                continue
            for b in c:
                if not (isinstance(b, dict) and b.get("type") == "tool_use"):
                    continue
                name = b.get("name", "")
                name_counts[name] = name_counts.get(name, 0) + 1
                inp = b.get("input") or {}
                if name == "run_python" and isinstance(inp.get("code"), str):
                    codes.append(inp["code"][-600:])
        # 相似 run_python ≥ 2 → 重复操作
        for i in range(len(codes)):
            for j in range(i + 1, len(codes)):
                if jac(tok(codes[i]), tok(codes[j])) >= 0.40:
                    return ("you have run near-identical Python more than once — a recurring "
                            "operation. Run self_evolve to assess; if VERDICT is BUILD, "
                            "create_tool with test_code then review_tool. Decide and build "
                            "autonomously; do not ask the user to specify the tool.")
        # 同一工具连用 ≥ 3(排除元工具/查询类)
        meta = {"run_python", "list_tools", "find_similar_tools",
                "self_evolve", "propose_tool", "read_file", "consolidate_memory"}
        for nm, c in name_counts.items():
            if c >= 3 and nm not in meta:
                return (f"you have called '{nm}' {c} times — if this is a recurring "
                        f"multi-step pattern, consider self_evolve to assess a dedicated tool.")
        return None

    def _maybe_record_signal(self, name: str, inp, result) -> None:
        """工具调用后,若表现出"挣扎"(失败/手搓绕路),记一条信号事件。

        失败 = 能力边界的最强信号;run_python 多行代码 = 手搓绕路(没现成工具)。
        这些只入记忆图、不进对话上下文,经 reflect pass 才结晶成边界。
        """
        try:
            from memory import record_signal
            r = str(result).lstrip().lower()
            if r.startswith(("error", "tool execution error")):
                record_signal("failure", f"{name} → {str(result).strip()[:160]}", tags=name)
            elif name == "run_python":
                code = (inp or {}).get("code", "") if isinstance(inp, dict) else ""
                if isinstance(code, str):
                    nonblank = [l for l in code.splitlines() if l.strip()]
                    if len(nonblank) >= 4:  # 多行手搓 → 绕路候选
                        record_signal("workaround",
                                      f"manual python ({len(nonblank)} lines): "
                                      f"{nonblank[0][:70]}",
                                      tags="run_python")
        except Exception:
            pass

    def consolidate_memory(self) -> str:
        """触发记忆折叠循环：event → concept → intent（涌现）+ 反思(能力边界)。

        consolidate: 相似 event → concept → 强 concept 结晶 intent(拓扑涌现)。
        reflect: 从失败/绕路信号识别真正的能力边界(挣扎指纹,非词面重复)。
        两者都一次 LLM 批处理,成本可控。返回报告 + 涌现快照。
        """
        from memory import consolidate, reflect, resolve_supersede, context_block
        tools = ([(name, t.description) for name, t in self.tools._tools.items()]
                 if self.tools else [])
        report = consolidate(self._llm_summarize, tools=tools)
        report += "\n" + reflect(self._llm_summarize)
        report += "\n" + resolve_supersede(self._llm_summarize)
        return report + "\n" + context_block()

    def _maybe_auto_consolidate(self) -> None:
        """主动性闸门:挣扎信号攒够(≥2)或事件堆积(≥8)时自动折叠记忆。

        这是"记忆真正被利用"的关键 —— 否则 consolidate_memory 永远要 agent 手动调,
        图不折叠 → concept/intent/gap 永不涌现 → "记忆当需求传感器"整套设计空转。
        自动折叠后,本回合的 system prompt(下方 _build_system_prompt)立刻读到
        新涌现的 ⚡ capability gap,agent 该不该造工具一目了然。

        成本可控:reflect 在 <2 信号时早退、resolve_supersede 在 <2 concept 时早退,
        consolidate 无簇也不调 LLM;只在真实信号/堆积时才花 1-2 次 batch LLM。
        每次折叠都会消耗信号(reflect 标 reflected、consolidate 标 consolidated),
        故天然自限,不会每轮重复触发。
        """
        try:
            import re as _re
            from memory import pending_signals, graph_stats, consolidate, reflect
            signals = pending_signals()
            unconsolidated = 0
            m = _re.search(r"unconsolidated_events=(\d+)", graph_stats())
            if m:
                unconsolidated = int(m.group(1))
            if signals < 2 and unconsolidated < 8:
                return  # 无信号又无堆积 → 不花 LLM
            tools = ([(n, t.description) for n, t in self.tools._tools.items()]
                     if self.tools else [])
            consolidate(self._llm_summarize, tools=tools)
            if signals >= 2:
                reflect(self._llm_summarize)
        except Exception:
            pass  # 折叠失败不影响主流程


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

    # 内置涌现式记忆（CogniFold 式概念图）：写入 event + 维护 intent；
    # 读取自动注入 system prompt（见 _build_system_prompt → context_block）。
    # 刻意不提供 recall/read —— 记忆是图拓扑涌现的，不是检索来的。
    from memory import (remember as mem_remember, forget as mem_forget,
                        update_intent as mem_update_intent)
    from tools import Tool as _Tool
    registry.register(_Tool(
        "remember",
        "Commit ONE event to the memory graph (hippocampal layer). Use for "
        "corrections, durable facts, user preferences, environment/project notes. "
        "Raw material — understanding emerges later via consolidate_memory. "
        "What surfaces in context each turn is the graph's emergent state, so "
        "there is intentionally no read/search tool.",
        {"content": {"type": "string", "description": "the fact / rule / preference / observation"},
         "tags": {"type": "string", "description": "comma-separated tags (optional)", "default": ""}},
        mem_remember,
        required=["content"],
    ))
    registry.register(_Tool(
        "forget",
        "Remove a node from the memory graph by its id (event/concept/intent, "
        "as shown in the MEMORY block).",
        {"node_id": {"type": "string", "description": "the graph node id to remove (e-/c-/i-)"}},
        mem_forget,
        required=["node_id"],
    ))
    registry.register(_Tool(
        "update_intent",
        "Update an intent's status (pending|in_progress|done|skipped) and urgency. "
        "Call when you act on an intent surfaced in the MEMORY block.",
        {"intent_id": {"type": "string", "description": "the intent node id (i-...)"},
         "status": {"type": "string", "description": "pending|in_progress|done|skipped", "default": "pending"},
         "urgency": {"type": "integer", "description": "1-5 (optional)"}},
        mem_update_intent,
        required=["intent_id"],
    ))

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

    # consolidate_memory 需要绑 agent 实例（折叠要复用其 LLM）
    registry.register(Tool(
        "consolidate_memory",
        "Fold the memory graph: cluster recurring events into concepts, crystallize "
        "strongly-supported concepts into intents that surface on their own. Run after "
        "adding several events, or when the MEMORY block shows unconsolidated events "
        "piling up. No arguments.",
        {},
        agent.consolidate_memory,
    ))

    return agent


# ── 救援 Agent（最终兜底）──────────────────────────────────

RESCUE_PROMPT = """\
You are MIRROR's RESCUE agent — a minimal, stable fallback engaged only when \
the primary agent's communication failed mid-task.

Constraints (do not evolve):
1. You have ONLY basic file + code execution tools. No sub-agents, no \
self-evolution, no meta-tools. Work within this set.
2. Complete the user's task directly and reliably. Prefer simple, robust steps.
3. If you cannot fully complete it, say precisely what you did and what remains.
4. Always verify file/code changes with a tool before claiming success.
"""


def create_rescue_agent(api_key: str | None = None, config=None) -> Agent:
    """最小、稳定的救援 agent：仅在主 agent 通信异常时由 TUI 激活。

    设计目标"功能固定不变"：
    - 只注册经过充分测试的基础工具（文件读写/搜索/代码执行）；
    - 不加载自造工具、不注册元工具、不绑定子 agent；
    - 独立空历史，不走上下文压缩（避免主 agent 的历史损坏风险）；
    - 固定 system prompt，短轮次。
    """
    from filetools import create_file_tools
    from tools import ToolRegistry

    reg = ToolRegistry()
    safe = {"read_file", "write_file", "edit_file", "list_dir",
            "glob", "grep", "run_python", "run_shell"}
    for t in create_file_tools():
        if t.name in safe:
            reg.register(t)

    model = getattr(config, "model", "deepseek-v4-pro")
    base_url = getattr(config, "base_url", "https://api.deepseek.com/anthropic")

    return Agent(
        api_key=api_key,
        base_url=base_url,
        model=model,
        max_turns=6,
        max_history=12,
        system_prompt=RESCUE_PROMPT,
        tools=reg,
        toolforge=None,
    )
