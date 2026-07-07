"""
持久化记忆 — 跨会话保存用户偏好/规则/重要上下文。

设计：
- 写入：agent 调用 `remember` 内置工具（见 agent.create_agent）。
- 读取：每轮自动注入 system prompt（format_for_prompt），无需工具调用、
  不产生多余字段，记忆直接进入上下文。

存储：~/.mirror/memory.json，向后兼容旧 self-made memory 工具的格式。
"""

from __future__ import annotations

import json
import os
from datetime import datetime

MEMORY_DIR = os.path.expanduser("~/.mirror")
MEMORY_FILE = os.path.join(MEMORY_DIR, "memory.json")


def _load() -> dict:
    if not os.path.exists(MEMORY_FILE):
        return {"memories": []}
    try:
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict) and isinstance(data.get("memories"), list):
                return data
            return {"memories": []}
    except (json.JSONDecodeError, OSError):
        return {"memories": []}


def _save(data: dict) -> None:
    os.makedirs(MEMORY_DIR, exist_ok=True)
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _next_id(existing: list[dict]) -> str:
    n = len(existing) + 1
    return f"m{n:03d}"


def remember(content: str, tags: str = "") -> str:
    """持久化一条记忆（agent 工具入口）。"""
    content = (content or "").strip()
    if not content:
        return "[!] remember: 'content' is required."
    data = _load()
    entry = {
        "id": _next_id(data["memories"]),
        "content": content,
        "tags": [t.strip() for t in (tags or "").split(",") if t.strip()],
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
    data["memories"].append(entry)
    _save(data)
    preview = content[:100] + ("..." if len(content) > 100 else "")
    return f"[✓] Remembered (id={entry['id']}): {preview}"


def forget(memory_id: str) -> str:
    """按 id 删除一条记忆。"""
    data = _load()
    before = len(data["memories"])
    data["memories"] = [m for m in data["memories"] if m.get("id") != memory_id]
    if len(data["memories"]) == before:
        return f"[!] forget: id={memory_id} not found."
    _save(data)
    return f"[✓] Forgot id={memory_id}."


def format_for_prompt() -> str:
    """把全部记忆格式化为注入 system prompt 的文本块；无记忆返回空串。"""
    mems = _load().get("memories", [])
    if not mems:
        return ""
    lines = ["", "PERSISTENT MEMORY (auto-loaded — apply across sessions):"]
    for m in mems:
        tags = f" [{', '.join(m.get('tags', []))}]" if m.get("tags") else ""
        lines.append(f"  ({m.get('id', '?')}){tags} {m.get('content', '')}")
    return "\n".join(lines)
