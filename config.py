"""
配置管理 — 从 config.json + 环境变量加载，支持运行时修改与持久化。

优先级: 运行时参数 > config.json > 环境变量 > 默认值。
API Key 始终从 .env / 环境变量读取，不写入 config.json。
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field


@dataclass
class Config:
    """Agent 运行配置。"""

    model: str = "deepseek-v4-pro"
    base_url: str = "https://api.deepseek.com/anthropic"
    max_tokens: int = 4096
    temperature: float = 1.0
    max_turns: int = 10
    max_history: int = 50
    system_prompt: str = ""          # 空 = 使用内置默认 prompt
    theme: str = "spark"             # 预留：主题

    # 不序列化的运行时字段
    _path: str = field(default="config.json", repr=False)

    @classmethod
    def load(cls, path: str = "config.json") -> "Config":
        """加载配置：config.json 覆盖默认值，环境变量再覆盖。"""
        cfg = cls()
        cfg._path = path

        # 1. config.json
        if os.path.isfile(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for k, v in data.items():
                    if hasattr(cfg, k):
                        setattr(cfg, k, v)
            except (json.JSONDecodeError, OSError):
                pass

        # 2. 环境变量（DEEPSEEK_MODEL, DEEPSEEK_BASE_URL, DEEPSEEK_TEMP ...）
        env_map = {
            "DEEPSEEK_MODEL": ("model", str),
            "DEEPSEEK_BASE_URL": ("base_url", str),
            "DEEPSEEK_MAX_TOKENS": ("max_tokens", int),
            "DEEPSEEK_TEMP": ("temperature", float),
            "DEEPSEEK_MAX_TURNS": ("max_turns", int),
            "DEEPSEEK_MAX_HISTORY": ("max_history", int),
        }
        for env_key, (attr, typ) in env_map.items():
            val = os.getenv(env_key)
            if val:
                try:
                    setattr(cfg, attr, typ(val))
                except ValueError:
                    pass

        return cfg

    def save(self, path: str | None = None) -> None:
        """持久化配置到 JSON。"""
        target = path or self._path
        data = {k: v for k, v in asdict(self).items() if not k.startswith("_")}
        with open(target, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def summary(self) -> str:
        """单行摘要，用于 TUI 状态显示。"""
        return (
            f"model={self.model}  temp={self.temperature}  "
            f"max_tokens={self.max_tokens}  turns={self.max_turns}"
        )
