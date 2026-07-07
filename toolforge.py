"""
ToolForge — 自我制造工具系统（L0 元工具层）。

让 Agent 能自主创建、测试、调试、复用工具。
自造工具落盘在 tools/custom/<name>/{tool.py, meta.json}，启动时自动加载。

元工具:
  create_tool   创建新工具（代码 + 描述 + 参数 + 可选测试）
  list_tools    列出所有工具及描述（内置 + 自造）
  read_tool     读取某工具源码（debug 时查看）
  update_tool   更新已有工具的代码/描述
  delete_tool   删除工具

工具间组合复用:自造工具的 tool.py 中可直接调用
    result = use("other_tool_name", **kwargs)
（use 由 ToolForge 在加载时注入到模块全局命名空间）
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import types
from datetime import datetime

from tools import Tool, ToolRegistry

_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{1,40}$")
# 不可被自造工具占用（避免覆盖元工具 / 内置工具）
_RESERVED = {
    "create_tool", "list_tools", "read_tool", "update_tool", "delete_tool",
    "propose_tool", "find_similar_tools", "review_tool", "improve_tool",
    "tool_stats", "self_evolve",
    "use",
}
TEST_TIMEOUT = 15

# ── 自进化决策用的停用词与危险模式 ──────────────────────
_STOPWORDS = {
    "a", "an", "the", "of", "to", "in", "on", "for", "and", "or", "with",
    "that", "this", "it", "is", "are", "be", "by", "from", "as", "at",
    "into", "your", "you", "i", "we", "they", "my", "me",
}
_DANGER_PATTERNS = [
    ("eval(",          "uses eval() — arbitrary code execution"),
    ("exec(",          "uses exec() — arbitrary code execution"),
    ("os.system(",     "uses os.system() — shell injection risk"),
    ("__import__(",    "dynamic import"),
    ("subprocess.",    "spawns subprocess"),
    ("open('/etc",     "hardcoded system path"),
]


class ToolForge:
    """管理自造工具的创建、持久化、加载、组合复用。"""

    def __init__(self, registry: ToolRegistry, base_dir: str | None = None):
        self.registry = registry
        base = base_dir or os.getcwd()
        self.custom_dir = os.path.join(base, "tools", "custom")
        os.makedirs(self.custom_dir, exist_ok=True)
        # 使用统计持久化（哪些工具真正在用）
        self._stats_path = os.path.join(self.custom_dir, "_usage.json")
        self._stats = self._load_stats()

    # ── 加载 ──────────────────────────────────────────────

    def load_existing(self) -> list[str]:
        """启动时扫描 tools/custom/，加载所有自造工具。返回已加载名列表。"""
        loaded, failed = [], []
        if not os.path.isdir(self.custom_dir):
            return loaded
        for name in sorted(os.listdir(self.custom_dir)):
            tool_dir = os.path.join(self.custom_dir, name)
            if not os.path.isdir(tool_dir):
                continue
            try:
                self._register_from_disk(name)
                loaded.append(name)
            except Exception as e:
                failed.append(f"{name}: {e}")
        return loaded + [f"[failed] {f}" for f in failed]

    def _register_from_disk(self, name: str) -> None:
        meta = self._read_meta(name)
        fn = self._import_execute(name)
        self.registry.register(Tool(
            name=meta["name"],
            description=meta["description"],
            parameters=self._normalize_parameters(meta["parameters"]),
            fn=fn,
            required=meta.get("required"),
        ))

    def _import_execute(self, name: str):
        """从 tool.py 加载 execute 函数，并注入 use()。

        直接读取源码并 exec 到独立命名空间，彻底绕过 importlib 的字节码缓存
        与 sys.modules —— 保证 update_tool 后立即生效，无旧代码残留。
        """
        tool_path = self._tool_path(name)
        with open(tool_path, "r", encoding="utf-8") as f:
            source = f.read()
        module = types.ModuleType(f"custom_tool_{name}")
        # 注入组合复用入口
        module.use = self.use
        module.__dict__["use"] = self.use
        code_obj = compile(source, tool_path, "exec")
        exec(code_obj, module.__dict__)  # noqa: S102 — 受信任的自造工具沙箱
        execute = getattr(module, "execute", None)
        if not callable(execute):
            raise RuntimeError("tool.py must define a callable execute(...)")
        return execute

    # ── 路径辅助 ──────────────────────────────────────────

    def _tool_dir(self, name: str) -> str:
        return os.path.join(self.custom_dir, name)

    def _tool_path(self, name: str) -> str:
        return os.path.join(self._tool_dir(name), "tool.py")

    def _meta_path(self, name: str) -> str:
        return os.path.join(self._tool_dir(name), "meta.json")

    def _exists(self, name: str) -> bool:
        return os.path.isdir(self._tool_dir(name))

    def _rollback(self, name: str) -> None:
        """删除刚写入但验证失败的工具目录。"""
        import shutil
        d = self._tool_dir(name)
        if os.path.isdir(d):
            shutil.rmtree(d, ignore_errors=True)

    def _read_meta(self, name: str) -> dict:
        with open(self._meta_path(name), "r", encoding="utf-8") as f:
            return json.load(f)

    # ── 核心:创建 ────────────────────────────────────────

    def create_tool(
        self,
        name: str,
        description: str,
        parameters: dict,
        code: str,
        test_code: str = "",
    ) -> str:
        """创建并注册一个新工具。有 test_code 则先验证。"""
        err = self._validate_name(name)
        if err:
            return err
        if self._exists(name):
            return (f"Error: tool '{name}' already exists. "
                    f"Use update_tool to modify it.")
        if not description.strip():
            return "Error: description is required (it decides when the tool is used)."
        if "execute" not in code:
            return "Error: code must define a function `execute(...)`."

        # 落盘
        self._write_tool(name, code, description, parameters, test_code)

        # 尝试 import
        try:
            fn = self._import_execute(name)
        except Exception as e:
            self._rollback(name)
            return (f"Error: import failed, tool NOT created:\n{e}\n\n"
                    f"Fix the code and retry create_tool.")

        # 可选测试 — 失败则回滚，不留坏工具
        if test_code.strip():
            test_err = self._run_test(name, code, test_code)
            if test_err:
                self._rollback(name)
                return (f"Error: test FAILED, tool NOT created:\n{test_err}\n\n"
                        f"Fix the code so the test passes, then retry create_tool.")

        # 注册（注册前必须归一化 schema：_write_tool 只把归一化结果写进
        # meta.json，不会回传；若用 LLM 误传的完整 schema {type:object,properties:...}
        # 注册，坏嵌套会立刻进 registry，下一次请求被 DeepSeek 全量校验 → 400。
        # 重启才恢复，因为重启走 _register_from_disk 读的是已归一化的 meta.json。）
        parameters = self._normalize_parameters(parameters)
        required = self._infer_required(parameters)
        self.registry.register(Tool(name, description, parameters, fn, required))
        self._log(f"create_tool {name}")
        status = f"Tool '{name}' created and registered."
        if not test_code.strip():
            status += " (no test provided — not validated)"
        return status

    # ── 更新 ──────────────────────────────────────────────

    def update_tool(
        self,
        name: str,
        code: str = "",
        description: str = "",
        parameters: dict | None = None,
        test_code: str = "",
    ) -> str:
        if not self._exists(name):
            return f"Error: tool '{name}' not found."
        meta = self._read_meta(name)
        new_code = code.strip() or self._read_code(name)
        new_desc = description.strip() or meta["description"]
        new_params = parameters if parameters is not None else meta["parameters"]
        new_test = test_code.strip() or meta.get("test_code", "")

        if "execute" not in new_code:
            return "Error: code must define `execute(...)`."

        self._write_tool(name, new_code, new_desc, new_params, new_test)
        try:
            fn = self._import_execute(name)
        except Exception as e:
            return f"Error loading updated tool (registry kept the OLD version):\n{e}\nFix and update_tool again."

        # import 成功即部署新代码；测试结果作为告警而非阻断
        # （测试可能因行为变更而过时，不应锁死更新）
        # 注册前归一化 schema（与 create_tool 同理：_write_tool 只写盘不回传）
        new_params = self._normalize_parameters(new_params)
        required = self._infer_required(new_params)
        self.registry.register(Tool(name, new_desc, new_params, fn, required))

        test_note = ""
        if new_test:
            test_err = self._run_test(name, new_code, new_test)
            if test_err:
                test_note = f"\nWARNING: test failed (code deployed anyway — update test_code if behavior changed):\n{test_err}"

        self._log(f"update_tool {name}")
        return f"Tool '{name}' updated.{test_note}"

    # ── 删除 / 查询 ───────────────────────────────────────

    def delete_tool(self, name: str) -> str:
        if not self._exists(name):
            return f"Error: tool '{name}' not found."
        import shutil
        shutil.rmtree(self._tool_dir(name))
        if name in self.registry._tools:
            del self.registry._tools[name]
        self._log(f"delete_tool {name}")
        return f"Tool '{name}' deleted."

    def list_tools(self) -> str:
        lines = ["Available tools:"]
        for name, tool in self.registry._tools.items():
            custom = "*" if self._exists(name) else " "
            desc = tool.description.split("\n")[0][:80]
            lines.append(f"  {custom} {name}: {desc}")
        lines.append("")
        lines.append("(* = custom/self-made. Use create_tool to add new ones.)")
        return "\n".join(lines)

    def read_tool(self, name: str) -> str:
        if not self._exists(name):
            # 可能是内置工具，无源码
            tool = self.registry.get(name)
            if tool:
                return (f"(built-in tool, no source on disk)\n"
                        f"name: {tool.name}\ndescription: {tool.description}\n"
                        f"parameters: {json.dumps(tool.parameters)}")
            return f"Error: tool '{name}' not found."
        code = self._read_code(name)
        meta = self._read_meta(name)
        return (f"# {name}\n"
                f"description: {meta['description']}\n\n"
                f"{code}")

    # ── 组合复用 ──────────────────────────────────────────

    def use(self, name: str, **kwargs):
        """工具间调用入口，供自造工具内部复用其他工具。"""
        return self.registry.execute(name, kwargs)

    # ── 自进化:使用统计 ────────────────────────────────────

    def _load_stats(self) -> dict:
        if os.path.isfile(self._stats_path):
            try:
                with open(self._stats_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                pass
        return {}

    def _save_stats(self) -> None:
        try:
            with open(self._stats_path, "w", encoding="utf-8") as f:
                json.dump(self._stats, f, indent=2, ensure_ascii=False)
        except OSError:
            pass

    def record_usage(self, name: str, ok: bool) -> None:
        """供 ToolRegistry 在每次执行后回调，累计使用统计。"""
        s = self._stats.setdefault(name, {"calls": 0, "ok": 0, "fail": 0, "last_used": ""})
        s["calls"] += 1
        s["ok" if ok else "fail"] += 1
        s["last_used"] = datetime.now().isoformat(timespec="seconds")
        # 每 10 次落盘一次，平衡 IO 与持久性
        total = sum(v.get("calls", 0) for v in self._stats.values())
        if total % 10 == 0:
            self._save_stats()

    def usage_stats(self, detail: str = "all") -> str:
        """工具使用统计：调用次数、成功率、最后使用时间。

        detail: all(默认) | unused(仅闲置工具，删除候选) | top(最常用)。
        闲置工具=有统计但调用极少，或已注册但 0 调用。
        """
        now = datetime.now()
        rows = []
        for name, t in self.registry._tools.items():
            s = self._stats.get(name, {"calls": 0, "ok": 0, "fail": 0, "last_used": ""})
            calls = s.get("calls", 0)
            ok = s.get("ok", 0)
            fail = s.get("fail", 0)
            rate = f"{100*ok/calls:.0f}%" if calls else "-"
            last = s.get("last_used", "")
            custom = "*" if self._exists(name) else " "
            rows.append((calls, name, custom, calls, ok, fail, rate, last))

        if detail == "unused":
            rows = [r for r in rows if r[3] == 0]
            if not rows:
                return "No unused tools — every tool has been called at least once."
            lines = ["Unused / low-use tools (removal candidates):"]
            for _, name, custom, calls, ok, fail, rate, _ in sorted(rows, key=lambda r: r[1]):
                lines.append(f"  {custom}{name}: {calls} call(s)")
            return "\n".join(lines)

        if detail == "top":
            rows.sort(key=lambda r: r[3], reverse=True)
            rows = rows[:8]

        lines = ["Tool usage stats:"]
        for _, name, custom, calls, ok, fail, rate, last in sorted(rows, key=lambda r: -r[3]):
            lines.append(f"  {custom}{name:<22} calls={calls:<5} ok={ok:<5} fail={fail:<3} "
                         f"rate={rate:<4} last={last or 'never'}")
        lines.append("")
        lines.append("(* = custom/self-made. Use detail='unused' to find removal candidates.)")
        return "\n".join(lines)

    # ── 自进化:相似工具查重 ────────────────────────────────

    @staticmethod
    def _tokens(text: str) -> set[str]:
        """分词：小写、按非字母数字切分、去停用词、去单字符。"""
        toks = re.split(r"[^a-z0-9]+", (text or "").lower())
        return {t for t in toks if len(t) > 1 and t not in _STOPWORDS}

    def _rank_similar(self, query: str) -> list[tuple[float, str, str]]:
        """对 query 与所有工具打相似度分，返回 (score, name, desc) 降序列表。"""
        qtok = self._tokens(query)
        if not qtok:
            return []
        scored: list[tuple[float, str, str]] = []
        for name, t in self.registry._tools.items():
            ntok = self._tokens(name.replace("_", " "))
            dtok = self._tokens(t.description)
            union_tok = ntok | dtok
            if not union_tok:
                continue
            inter = len(qtok & union_tok)
            if inter == 0:
                continue
            score = inter / len(qtok | union_tok)  # Jaccard
            # 名字直接命中加权
            if qtok & ntok:
                score += 0.25 * len(qtok & ntok) / max(len(qtok), 1)
            desc1 = t.description.split("\n")[0].strip()[:80]
            scored.append((score, name, desc1))
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored

    def find_similar_tools(self, query: str, top_k: int = 5) -> str:
        """查重：是否已存在覆盖该能力的工具。造工具前必查，避免重复造轮子。"""
        top_k = max(1, min(top_k, 15))
        ranked = self._rank_similar(query)
        if not ranked:
            return f"(no existing tool matches '{query}' — this looks like a real capability gap)"
        lines = [f"Similar tools to '{query}' (by relevance):"]
        for score, name, desc in ranked[:top_k]:
            custom = "*" if self._exists(name) else " "
            lines.append(f"  [{score:.2f}] {custom}{name}: {desc}")
        return "\n".join(lines)

    # ── 自进化:造工具决策引擎 ──────────────────────────────

    @staticmethod
    def _suggest_name(capability: str) -> str:
        """从能力描述启发式地推导一个候选工具名。"""
        toks = [t for t in re.split(r"[^a-z0-9]+", capability.lower()) if t and t not in _STOPWORDS]
        if not toks:
            return ""
        # 动词在前（如 count/read/fetch/parse）+ 名词
        verbs = {"count", "read", "write", "fetch", "parse", "get", "list", "find",
                 "convert", "format", "extract", "sum", "avg", "sort", "check", "validate"}
        lead = [t for t in toks if t in verbs]
        rest = [t for t in toks if t not in verbs]
        cand = (lead + rest)[:3]
        name = "_".join(cand)
        return name[:40].rstrip("_")

    def propose_tool(
        self,
        capability: str,
        reuse_signal: str = "recurring",
        task_context: str = "",
    ) -> str:
        """造工具前的结构化价值判定。返回 BUILD / SKIP 及理由。

        确定性决策（非仅靠 prompt 感觉）：
          1. 查重 —— 已有强相似工具 → SKIP，建议复用
          2. 复用信号 —— 'once' → SKIP（用 run_python）；'recurring'/'few' → 倾向 BUILD
          3. 否则 —— BUILD，给出候选名 + 规范
        判定可被 agent 与用户检验，避免凭感觉造工具。
        """
        capability = (capability or "").strip()
        reuse = (reuse_signal or "").lower().strip()
        if not capability:
            return "Error: capability is required (describe WHAT the tool would do)."

        ranked = self._rank_similar(capability)
        strong = [(s, n, d) for s, n, d in ranked if s >= 0.45]
        moderate = [(s, n, d) for s, n, d in ranked if 0.2 <= s < 0.45]

        if strong:
            top_s, top_n, top_d = strong[0]
            verdict = "SKIP"
            reason = (f"An existing tool already covers this: '{top_n}' "
                      f"(similarity {top_s:.2f}). Reuse it — rebuilding would duplicate.")
            recommendation = f"Use existing tool '{top_n}'."
            action = f"call {top_n}(...) directly"
        elif reuse in ("once", "one-off", "one_off", "single"):
            verdict = "SKIP"
            reason = ("reuse_signal='once': one-off tasks should NOT become persisted tools. "
                      "Use run_python — faster, no clutter.")
            recommendation = "Use run_python for this one-off task."
            action = "run_python with inline code"
        elif reuse in ("recurring", "repeat", "always", "every") or not reuse:
            verdict = "BUILD"
            reason = ("No similar tool exists and reuse is recurring — a dedicated tool "
                      "is justified. Build it once, reuse forever.")
            recommendation = "Proceed: create_tool with precise description + test_code."
            action = "create_tool(name, description, parameters, code, test_code)"
        else:  # 'few' or unknown → 谨慎 BUILD，提示权衡
            verdict = "BUILD"
            reason = (f"No strong match and reuse_signal='{reuse}'. Build only if the "
                      f"operation is genuinely stable; otherwise run_python is lighter.")
            recommendation = "Build if stable & multi-step; else run_python."

        suggested = self._suggest_name(capability)
        # 校正建议名不与现有冲突
        if suggested and self.registry.get(suggested):
            suggested = f"{suggested} (name taken — pick another)"

        similar_str = ", ".join(f"{n}({s:.2f})" for s, n, _ in ranked[:3]) or "(none)"
        moderate_str = ", ".join(f"{n}({s:.2f})" for s, n, _ in moderate[:2]) or "(none)"

        lines = [
            f"VERDICT: {verdict}",
            f"reason: {reason}",
            f"recommendation: {recommendation}",
            f"capability: {capability}",
            f"suggested_name: {suggested or '(could not derive — name it yourself)'}",
            f"reuse_signal: {reuse or 'recurring (default)'}",
            f"similar_existing: {similar_str}",
        ]
        if moderate and not strong:
            lines.append(f"partial_overlap (review before building): {moderate_str}")
        lines.append(f"next_action: {action}")
        if verdict == "BUILD":
            lines.append(
                "tip: description must state WHEN to use it; always include test_code; "
                "compose via use('other_tool', **kwargs) instead of reimplementing."
            )
        return "\n".join(lines)

    # ── 自进化:工具审查 + 改进 ─────────────────────────────

    def review_tool(self, name: str) -> str:
        """对自造工具做静态质量审查：错误处理、测试、安全、输入校验、复杂度。"""
        if not self._exists(name):
            tool = self.registry.get(name)
            if tool:
                return (f"'{name}' is built-in (no source to review).\n"
                        f"description: {tool.description}")
            return f"Error: tool '{name}' not found."

        code = self._read_code(name)
        meta = self._read_meta(name)
        checks: list[tuple[str, str]] = []

        # 错误处理
        has_try = "try" in code and "except" in code
        checks.append(("error handling",
                       "PASS: has try/except" if has_try
                       else "WARN: no try/except — unexpected inputs surface as raw tool errors"))

        # 测试
        has_test = bool(meta.get("test_code", "").strip())
        checks.append(("test coverage",
                       "PASS: has test_code" if has_test
                       else "WARN: no test_code — behavior is unverified"))

        # 安全
        dangers = [label for pat, label in _DANGER_PATTERNS if pat in code]
        checks.append(("safety",
                       "PASS: no dangerous patterns" if not dangers
                       else "WARN: " + "; ".join(dangers)))

        # 裸 except
        checks.append(("bare except",
                       "PASS: none" if "except:" not in code
                       else "WARN: bare 'except:' swallows all errors — catch specific types"))

        # 输入校验
        has_check = ("isinstance" in code or "if not " in code
                     or " is None" in code or ".get(" in code)
        checks.append(("input validation",
                       "PASS: validates inputs" if has_check
                       else "WARN: no visible input validation — bad args may crash"))

        # 组合复用 vs 重复实现
        uses_compose = "use(" in code
        checks.append(("composition",
                       "INFO: composes other tools via use()" if uses_compose
                       else "INFO: standalone — consider use() instead of reimplementing"))

        # 复杂度
        nlines = len([l for l in code.splitlines() if l.strip()])
        checks.append(("size", f"INFO: {nlines} non-blank lines"))

        # 使用统计
        s = self._stats.get(name, {})
        calls = s.get("calls", 0)
        checks.append(("usage", f"INFO: {calls} call(s) so far"))

        lines = [f"Review of '{name}':", ""]
        for label, status in checks:
            tag = status.split(":", 1)[0]
            icon = {"PASS": "[OK]", "WARN": "[!]", "INFO": "[-]"}.get(tag, "[?]")
            lines.append(f"  {icon} {label}: {status}")

        warns = [c for c in checks if c[1].startswith("WARN")]
        if warns:
            lines.append("")
            lines.append("To improve: call improve_tool with focus on the ! items above.")
        return "\n".join(lines)

    def improve_tool(self, name: str, focus: str = "", failure: str = "") -> str:
        """迭代改进入口：汇总审查 + 失败诊断 + 给出可执行的修复指引。

        不直接改代码（那是 update_tool 的职责），而是把问题结构化，
        让 agent 明确「改什么、为什么、怎么改」。
        """
        if not self._exists(name):
            return f"Error: '{name}' is not a custom tool (nothing to improve)."

        review = self.review_tool(name)
        parts = [review, ""]

        if failure.strip():
            parts.append(f"REPORTED FAILURE:\n{failure.strip()[-1200:]}")
            parts.append("")
            parts.append(
                "Diagnose: read the failure above, locate the bug in the code "
                f"(read_tool {name}), fix it, then update_tool {name} with corrected "
                "code AND updated test_code reflecting the intended behavior."
            )

        if focus.strip():
            parts.append(f"FOCUS: {focus.strip()}")
            parts.append(
                "Address this specifically — edit the relevant part of execute(), "
                f"keep the rest stable, then update_tool {name}."
            )

        if not failure.strip() and not focus.strip():
            parts.append(
                "No specific focus given. Tackle every WARN (!) from the review above, "
                "in priority order. After fixing, update_tool with fresh test_code."
            )

        parts.append("")
        parts.append("Loop: update_tool -> review_tool -> repeat until no WARN remains.")
        return "\n".join(parts)

    # ── 自进化:端到端编排 ──────────────────────────────────

    def self_evolve(self, goal: str) -> str:
        """引导一次完整的自进化评估：能力缺口 → 查重 → 判定 → 行动建议。

        给 agent 一个单一入口来思考「我是否需要新工具」，
        返回结构化的下一步，而不是凭直觉直接 create_tool。
        """
        goal = (goal or "").strip()
        if not goal:
            return "Error: describe the goal / recurring pain point you keep hitting manually."

        parts = [
            f"SELF-EVOLUTION ASSESSMENT for: {goal}",
            "",
            "1) CAPABILITY GAP — is this already covered?",
            self.find_similar_tools(goal),
            "",
            "2) BUILD/SKIP DECISION:",
            self.propose_tool(goal, reuse_signal="recurring"),
            "",
        ]
        parts.append(
            "3) NEXT STEP: follow the VERDICT above. If BUILD, write create_tool with "
            "a precise description (states WHEN to use) + test_code, then review_tool "
            "to verify quality. If SKIP, use the recommended existing tool or run_python."
        )
        return "\n".join(parts)

    # ── 内部:写盘 / 测试 / 校验 ───────────────────────────

    def _write_tool(self, name, code, description, parameters, test_code) -> None:
        parameters = self._normalize_parameters(parameters)
        d = self._tool_dir(name)
        os.makedirs(d, exist_ok=True)
        with open(self._tool_path(name), "w", encoding="utf-8") as f:
            f.write(code.rstrip() + "\n")
        meta = {
            "name": name,
            "description": description,
            "parameters": parameters,
            "required": self._infer_required(parameters),
            "test_code": test_code,
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
        with open(self._meta_path(name), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

    def _read_code(self, name: str) -> str:
        with open(self._tool_path(name), "r", encoding="utf-8") as f:
            return f.read()

    def _run_test(self, name: str, code: str, test_code: str) -> str:
        """在隔离 subprocess 里跑 test_code。test_code 可调用 execute(...)。"""
        script = code + "\n\n# ---- test ----\n" + test_code
        try:
            r = subprocess.run(
                [sys.executable, "-c", script],
                capture_output=True, text=True, timeout=TEST_TIMEOUT,
            )
        except subprocess.TimeoutExpired:
            return f"test timed out after {TEST_TIMEOUT}s"
        if r.returncode != 0:
            return (f"exit {r.returncode}\n"
                    f"{r.stderr.strip()[-1500:]}"
                    + (f"\n[stdout]\n{r.stdout.strip()[-500:]}" if r.stdout else ""))
        return ""

    def _validate_name(self, name: str) -> str:
        if not _NAME_RE.match(name):
            return ("Error: invalid name. Use 2-40 chars, lowercase letters/digits/_, "
                    "starting with a letter.")
        if name in _RESERVED:
            return f"Error: '{name}' is a reserved name."
        return ""

    @staticmethod
    def _infer_required(parameters: dict) -> list[str]:
        """无默认值的参数视为必填。约定:值含 default 字段的可选。"""
        req = []
        for k, v in parameters.items():
            if isinstance(v, dict) and "default" in v:
                continue
            req.append(k)
        return req

    @staticmethod
    def _normalize_parameters(parameters) -> dict:
        """归一化为 {属性名: 属性schema} 形式。

        兼容 LLM 误传的完整 schema（含 type=object + properties）——
        否则 Tool.to_param 会把整个 schema 当成一个名为 "type" 的属性，
        触发 DeepSeek 校验报错："object" is not of types "boolean","object"。
        """
        if not isinstance(parameters, dict):
            return {}
        if (parameters.get("type") == "object"
                and isinstance(parameters.get("properties"), dict)):
            return parameters["properties"]
        return parameters

    def _log(self, msg: str) -> None:
        log_path = os.path.join(self.custom_dir, "toolforge.log")
        try:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"[{datetime.now().isoformat(timespec='seconds')}] {msg}\n")
        except OSError:
            pass

    # ── 元工具列表 ────────────────────────────────────────

    def get_meta_tools(self) -> list[Tool]:
        """返回 L0 元工具，供注册到 registry。"""
        return [
            Tool(
                "create_tool",
                "Create a NEW reusable tool when a capability is missing AND will be reused. "
                "Provide name (snake_case), a clear description of WHEN to use it, "
                "JSON-schema parameters, Python code defining `execute(**kwargs)->str`, "
                "and optional test_code. The tool is validated (if test given) and registered. "
                "Do NOT create tools for one-off tasks — use run_python instead.",
                {"name": {"type": "string", "description": "snake_case, e.g. 'word_count'"},
                 "description": {"type": "string", "description": "WHAT and WHEN to use it"},
                 "parameters": {"type": "object", "description": "JSON schema properties"},
                 "code": {"type": "string", "description": "Python defining execute(**kwargs)"},
                 "test_code": {"type": "string", "description": "optional test calling execute()"}},
                self.create_tool,
                required=["name", "description", "parameters", "code"],
            ),
            Tool(
                "list_tools",
                "List all available tools (built-in and self-made) with descriptions. "
                "Call this before creating a tool to avoid duplicates.",
                {},
                lambda **_: self.list_tools(),
            ),
            Tool(
                "read_tool",
                "Read the source code of a (custom) tool. Use when debugging your own tool.",
                {"name": {"type": "string"}},
                self.read_tool,
            ),
            Tool(
                "update_tool",
                "Update an existing custom tool's code/description. Re-validates if test present.",
                {"name": {"type": "string"},
                 "code": {"type": "string", "description": "new execute() code (optional)"},
                 "description": {"type": "string"},
                 "test_code": {"type": "string"}},
                self.update_tool,
                required=["name"],
            ),
            Tool(
                "delete_tool",
                "Delete a custom tool.",
                {"name": {"type": "string"}},
                self.delete_tool,
            ),
            # ── 自进化决策层 ──
            Tool(
                "find_similar_tools",
                "Check whether an existing tool already covers a capability, BEFORE building. "
                "Returns ranked similar tools by relevance. Call this to avoid duplicate tools.",
                {"query": {"type": "string", "description": "the capability you need"},
                 "top_k": {"type": "integer", "description": "max results, default 5"}},
                self.find_similar_tools,
                required=["query"],
            ),
            Tool(
                "propose_tool",
                "Structured BUILD/SKIP decision for a potential new tool. Assesses duplication "
                "(finds similar tools) and reuse signal, returns a verdict with reason + suggested "
                "name. ALWAYS call this before create_tool — it answers 'should I build this at all?'.",
                {"capability": {"type": "string", "description": "WHAT the tool would do"},
                 "reuse_signal": {"type": "string", "description": "once | few | recurring (default recurring)"},
                 "task_context": {"type": "string", "description": "optional: why you think you need it"}},
                self.propose_tool,
                required=["capability"],
            ),
            Tool(
                "review_tool",
                "Static quality review of a custom tool: error handling, test coverage, safety, "
                "input validation, complexity, usage. Use after creating or when a tool misbehaves.",
                {"name": {"type": "string"}},
                self.review_tool,
            ),
            Tool(
                "improve_tool",
                "Guided improvement loop for a custom tool. Summarizes review + any reported failure "
                "and gives concrete fix direction. Pair with update_tool to iterate.",
                {"name": {"type": "string"},
                 "focus": {"type": "string", "description": "specific aspect to fix"},
                 "failure": {"type": "string", "description": "error/test output to diagnose"}},
                self.improve_tool,
                required=["name"],
            ),
            Tool(
                "tool_stats",
                "Show tool usage statistics (calls, success rate, last used). "
                "detail: all (default) | unused (removal candidates) | top (most used). "
                "Reveals which tools earn their place and which are dead weight.",
                {"detail": {"type": "string", "description": "all | unused | top"}},
                self.usage_stats,
            ),
            Tool(
                "self_evolve",
                "One-shot self-evolution assessment for a recurring goal/pain point. "
                "Checks capability gap + BUILD/SKIP decision + next step. Call this when you "
                "notice yourself repeating a multi-step operation and wondering if a tool would help.",
                {"goal": {"type": "string", "description": "the recurring need or capability gap"}},
                self.self_evolve,
                required=["goal"],
            ),
        ]
