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

import importlib.util
import json
import os
import re
import subprocess
import sys
from datetime import datetime

from tools import Tool, ToolRegistry

_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{1,40}$")
# 不可被自造工具占用（避免覆盖元工具 / 内置工具）
_RESERVED = {
    "create_tool", "list_tools", "read_tool", "update_tool", "delete_tool",
    "use",
}
TEST_TIMEOUT = 15


class ToolForge:
    """管理自造工具的创建、持久化、加载、组合复用。"""

    def __init__(self, registry: ToolRegistry, base_dir: str | None = None):
        self.registry = registry
        base = base_dir or os.getcwd()
        self.custom_dir = os.path.join(base, "tools", "custom")
        os.makedirs(self.custom_dir, exist_ok=True)

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
            parameters=meta["parameters"],
            fn=fn,
            required=meta.get("required"),
        ))

    def _import_execute(self, name: str):
        """用 importlib 从 tool.py 加载 execute 函数，并注入 use()。"""
        tool_path = self._tool_path(name)
        mod_name = f"custom_tool_{name}"
        spec = importlib.util.spec_from_file_location(mod_name, tool_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot load {tool_path}")
        module = importlib.util.module_from_spec(spec)
        # 注入组合复用入口
        module.use = self.use
        module.__dict__["use"] = self.use
        spec.loader.exec_module(module)
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

        # 注册
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
            return f"Error loading updated tool:\n{e}\nFix and update_tool again."

        if new_test:
            test_err = self._run_test(name, new_code, new_test)
            if test_err:
                return f"Updated but TEST FAILED:\n{test_err}"

        required = self._infer_required(new_params)
        self.registry.register(Tool(name, new_desc, new_params, fn, required))
        self._log(f"update_tool {name}")
        return f"Tool '{name}' updated."

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

    # ── 内部:写盘 / 测试 / 校验 ───────────────────────────

    def _write_tool(self, name, code, description, parameters, test_code) -> None:
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
        ]
