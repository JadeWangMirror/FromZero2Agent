"""
L1 基础工具集 — 文件操作 + 搜索 + 代码/命令执行。

文件:   read_file / write_file / edit_file / list_dir / move_file / delete_file
搜索:   glob / grep
执行:   run_python / run_shell

所有路径相对 base_dir（默认 cwd），绝对路径也允许。
"""

from __future__ import annotations

import fnmatch
import glob as _glob
import os
import re
import subprocess
import sys

from tools import Tool

MAX_READ_BYTES = 100_000
DEFAULT_TIMEOUT = 30
MAX_TIMEOUT = 120

# grep 默认搜索的扩展名（文本文件）
_TEXT_INCLUDES = ("*.py", "*.js", "*.ts", "*.txt", "*.md", "*.json",
                  "*.yaml", "*.yml", "*.toml", "*.csv", "*.html", "*.css",
                  "*.sh", "*.bat", "*.cfg", "*.ini", "*.xml")


def _resolve(base: str, path: str) -> str:
    path = os.path.expanduser(path)
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(base, path))


def create_file_tools(base_dir: str | None = None) -> list[Tool]:
    base = base_dir or os.getcwd()

    # ── read_file ──────────────────────────────────────────
    def _read_file(path: str) -> str:
        full = _resolve(base, path)
        if not os.path.isfile(full):
            return f"Error: not a file: {full}"
        size = os.path.getsize(full)
        with open(full, "r", encoding="utf-8", errors="replace") as f:
            content = f.read(MAX_READ_BYTES)
        suffix = f"\n\n... (truncated, file is {size} bytes)" if size > MAX_READ_BYTES else ""
        return content + suffix

    # ── write_file ─────────────────────────────────────────
    def _write_file(path: str, content: str) -> str:
        full = _resolve(base, path)
        os.makedirs(os.path.dirname(full) or ".", exist_ok=True)
        with open(full, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Wrote {len(content)} chars to {full}"

    # ── edit_file (精确替换，对标 Claude Code Edit) ────────
    def _edit_file(path: str, old_string: str, new_string: str, replace_all: bool = False) -> str:
        full = _resolve(base, path)
        if not os.path.isfile(full):
            return f"Error: not a file: {full}"
        with open(full, "r", encoding="utf-8") as f:
            content = f.read()
        count = content.count(old_string)
        if count == 0:
            return (f"Error: old_string not found in {full}. "
                    f"Make sure old_string matches exactly (incl. whitespace/indent).")
        if count > 1 and not replace_all:
            return (f"Error: old_string appears {count} times in {full}. "
                    f"Provide more surrounding context to make it unique, "
                    f"or set replace_all=true.")
        if replace_all:
            new_content = content.replace(old_string, new_string)
            n = count
        else:
            new_content = content.replace(old_string, new_string, 1)
            n = 1
        with open(full, "w", encoding="utf-8") as f:
            f.write(new_content)
        return f"Replaced {n} occurrence(s) in {full}"

    # ── list_dir ───────────────────────────────────────────
    def _list_dir(path: str = ".") -> str:
        full = _resolve(base, path)
        if not os.path.isdir(full):
            return f"Error: not a directory: {full}"
        entries = sorted(os.listdir(full))
        lines = [f"{n}{'/' if os.path.isdir(os.path.join(full, n)) else ''}" for n in entries]
        return "\n".join(lines) if lines else "(empty)"

    # ── move_file (移动/重命名) ────────────────────────────
    def _move_file(src: str, dst: str) -> str:
        s, d = _resolve(base, src), _resolve(base, dst)
        if not os.path.exists(s):
            return f"Error: source not found: {s}"
        os.makedirs(os.path.dirname(d) or ".", exist_ok=True)
        os.replace(s, d)
        return f"Moved {s} -> {d}"

    # ── delete_file ────────────────────────────────────────
    def _delete_file(path: str) -> str:
        full = _resolve(base, path)
        if os.path.isdir(full):
            import shutil
            shutil.rmtree(full)
            return f"Deleted directory {full}"
        if os.path.isfile(full):
            os.remove(full)
            return f"Deleted {full}"
        return f"Error: not found: {full}"

    # ── glob (文件名模式匹配) ──────────────────────────────
    def _glob_files(pattern: str, path: str = ".") -> str:
        root = _resolve(base, path)
        full_pattern = os.path.join(root, pattern)
        matches = sorted(_glob.glob(full_pattern, recursive=True))
        if not matches:
            return f"(no matches for {pattern} in {root})"
        # 相对化显示
        try:
            rel = [os.path.relpath(m, base) for m in matches[:200]]
        except ValueError:
            rel = matches[:200]
        suffix = f"\n... ({len(matches)} total)" if len(matches) > 200 else ""
        return "\n".join(rel) + suffix

    # ── grep (内容正则搜索) ────────────────────────────────
    def _grep(pattern: str, path: str = ".", include: str = "", max_matches: int = 100) -> str:
        root = _resolve(base, path)
        if not os.path.isdir(root):
            root = os.path.dirname(root) or "."
        try:
            regex = re.compile(pattern)
        except re.error as e:
            return f"Error: invalid regex: {e}"
        includes = tuple(include.split(",")) if include else _TEXT_INCLUDES
        results: list[str] = []
        walked = 0
        for dirpath, dirnames, filenames in os.walk(root):
            # 跳过常见无关目录
            dirnames[:] = [d for d in dirnames if d not in
                           {".git", "__pycache__", "node_modules", ".venv", "venv"}]
            for fn in filenames:
                if not any(fnmatch.fnmatch(fn, inc) for inc in includes):
                    continue
                fp = os.path.join(dirpath, fn)
                walked += 1
                try:
                    with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                        for lineno, line in enumerate(f, 1):
                            if regex.search(line):
                                rel = os.path.relpath(fp, base)
                                results.append(f"{rel}:{lineno}: {line.rstrip()[:200]}")
                                if len(results) >= max_matches:
                                    results.append(f"... (stopped at {max_matches} matches)")
                                    return "\n".join(results)
                except (OSError, UnicodeDecodeError):
                    continue
        if not results:
            return f"(no matches for /{pattern}/ in {root}, scanned {walked} files)"
        return "\n".join(results)

    # ── run_python ─────────────────────────────────────────
    def _run_python(code: str, timeout: int = DEFAULT_TIMEOUT) -> str:
        timeout = max(1, min(timeout, MAX_TIMEOUT))
        try:
            r = subprocess.run(
                [sys.executable, "-c", code],
                capture_output=True, text=True, timeout=timeout, cwd=base,
            )
            parts = []
            if r.stdout:
                parts.append(r.stdout)
            if r.stderr:
                parts.append(f"[stderr]\n{r.stderr}")
            if r.returncode != 0:
                parts.append(f"[exit code {r.returncode}]")
            return "\n".join(parts) if parts else "(no output)"
        except subprocess.TimeoutExpired:
            return f"Error: timed out after {timeout}s"

    # ── run_shell ──────────────────────────────────────────
    def _run_shell(command: str, timeout: int = DEFAULT_TIMEOUT) -> str:
        timeout = max(1, min(timeout, MAX_TIMEOUT))
        try:
            r = subprocess.run(
                command, shell=True, capture_output=True, text=True,
                timeout=timeout, cwd=base,
            )
            parts = []
            if r.stdout:
                parts.append(r.stdout)
            if r.stderr:
                parts.append(f"[stderr]\n{r.stderr}")
            if r.returncode != 0:
                parts.append(f"[exit code {r.returncode}]")
            return "\n".join(parts) if parts else "(no output)"
        except subprocess.TimeoutExpired:
            return f"Error: timed out after {timeout}s"

    return [
        Tool("read_file", "Read text content of a file (max 100KB).",
             {"path": {"type": "string", "description": "file path"}},
             _read_file),
        Tool("write_file", "Write text to a file (overwrites). Creates parent dirs.",
             {"path": {"type": "string"}, "content": {"type": "string"}},
             _write_file),
        Tool("edit_file",
             "Exact string replacement in a file. old_string must match exactly and be unique "
             "(or set replace_all). Use for precise edits.",
             {"path": {"type": "string"},
              "old_string": {"type": "string", "description": "must match exactly incl whitespace"},
              "new_string": {"type": "string"},
              "replace_all": {"type": "boolean", "description": "replace all occurrences"}},
             _edit_file, required=["path", "old_string", "new_string"]),
        Tool("list_dir", "List directory entries. Directories suffixed with '/'.",
             {"path": {"type": "string", "description": "default '.'"}},
             _list_dir, required=["path"]),
        Tool("move_file", "Move or rename a file/directory.",
             {"src": {"type": "string"}, "dst": {"type": "string"}},
             _move_file),
        Tool("delete_file", "Delete a file or directory (recursive).",
             {"path": {"type": "string"}}, _delete_file),
        Tool("glob", "Find files by name pattern. Supports ** for recursion, e.g. '**/*.py'.",
             {"pattern": {"type": "string", "description": "e.g. '**/*.py' or 'src/*.js'"},
              "path": {"type": "string", "description": "root dir, default '.'"}},
             _glob_files, required=["pattern"]),
        Tool("grep",
             "Search file contents with regex. Returns 'file:line: match'. "
             "Scans text files under path.",
             {"pattern": {"type": "string", "description": "regex pattern"},
              "path": {"type": "string", "description": "root dir, default '.'"},
              "include": {"type": "string", "description": "comma-separated globs, e.g. '*.py,*.js'"},
              "max_matches": {"type": "integer", "description": "default 100"}},
             _grep, required=["pattern"]),
        Tool("run_python", "Execute Python code in a subprocess, return stdout/stderr.",
             {"code": {"type": "string"},
              "timeout": {"type": "integer", "description": f"seconds (default {DEFAULT_TIMEOUT})"}},
             _run_python, required=["code"]),
        Tool("run_shell",
             "Execute a shell command, return stdout/stderr. Use for git, build tools, etc.",
             {"command": {"type": "string"},
              "timeout": {"type": "integer", "description": f"seconds (default {DEFAULT_TIMEOUT})"}},
             _run_shell, required=["command"]),
    ]
