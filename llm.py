"""
LLM 客户端 — 封装 Anthropic 协议调用（DeepSeek 兼容端点）。
支持流式输出（httpx 直连 SSE），实时回调 thinking 增量。
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import httpx
from anthropic.types import Message, MessageParam, ToolParam

# ── 流式事件 ──────────────────────────────────────────────


@dataclass
class StreamEvent:
    type: str          # "thinking_delta" | "text_delta" | "tool_use_start" | "tool_input_delta"
    text: str = ""
    name: str = ""
    input_json: str = ""
    id: str = ""


StreamCallback = Callable[[StreamEvent], None]


class _SimpleMsg:
    """模拟 Message，供 Agent 解析 content。"""

    def __init__(self, content: list):
        self.content = [_SimpleBlock(c) for c in content]


class _SimpleBlock:
    def __init__(self, d: dict):
        self.type = d.get("type", "")
        self.text = d.get("text", "")
        self.id = d.get("id", "")
        self.name = d.get("name", "")
        self.input = d.get("input", {})


# ── LLM Client ────────────────────────────────────────────


class LLMClient:
    """Anthropic 协议 LLM 客户端，用 httpx 处理 SSE 流。"""

    # 各模型上下文窗口（tokens）；未命中则取默认值
    CONTEXT_WINDOWS: dict[str, int] = {
        "deepseek-v4-pro": 128_000,
        "deepseek-v4-flash": 128_000,
        "deepseek-chat": 64_000,
        "deepseek-reasoner": 64_000,
    }
    _DEFAULT_WINDOW = 128_000

    @classmethod
    def window_for(cls, model: str) -> int:
        """按模型名查上下文窗口，未知模型回退到默认值。"""
        return cls.CONTEXT_WINDOWS.get(model, cls._DEFAULT_WINDOW)

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = "https://api.deepseek.com/anthropic",
        model: str = "deepseek-v4-pro",
        max_tokens: int = 4096,
        temperature: float = 1.0,
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_retries = 3
        self.retry_backoff = 1.0
        # 思考预算（extended thinking）；None = 不发送 thinking 字段
        self.thinking_budget: int | None = None
        # 累计 token 用量（跨本次进程所有调用）
        self.usage = {"input": 0, "output": 0, "calls": 0,
                      "cache_read": 0, "cache_creation": 0}
        # 最近一次主对话请求的规模（驱动上下文进度条）
        self.last = {"input": 0, "output": 0}
        # 当前模型上下文窗口
        self.context_window = self.window_for(model)

    def send(
        self,
        messages: list[MessageParam],
        tools: list[ToolParam] | None = None,
        system: str | None = None,
        thinking_budget: int | None = ...,
    ) -> Message:
        """非流式发送（带重试，供摘要等内部调用）。"""
        import anthropic
        client = anthropic.Anthropic(api_key=self.api_key, base_url=self.base_url)
        kwargs: dict = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": messages,
        }
        # 非流式不传 temperature（与 thinking 冲突时由 API 决定）
        tb = self.thinking_budget if thinking_budget is ... else thinking_budget
        if tb:
            kwargs["thinking"] = {"type": "enabled", "budget_tokens": tb}
        else:
            kwargs["temperature"] = self.temperature
        if tools:
            kwargs["tools"] = tools
        if system:
            kwargs["system"] = system
        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                resp = client.messages.create(**kwargs)
                self._tally(resp.usage)
                return resp
            except anthropic.APIStatusError as e:
                code = getattr(e, "status_code", 0)
                if code == 429 or code >= 500 and attempt < self.max_retries - 1:
                    last_exc = e
                    time.sleep(self.retry_backoff * (2 ** attempt))
                    continue
                raise
            except (anthropic.APIConnectionError, anthropic.APITimeoutError) as e:
                last_exc = e
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_backoff * (2 ** attempt))
                    continue
                raise
        assert last_exc is not None
        raise last_exc

    def _tally(self, usage) -> None:
        """把单次响应的 usage 累计到 self.usage。"""
        if usage is None:
            return
        self.usage["input"] += getattr(usage, "input_tokens", 0) or 0
        self.usage["output"] += getattr(usage, "output_tokens", 0) or 0
        self.usage["cache_read"] += getattr(usage, "cache_read_input_tokens", 0) or 0
        self.usage["cache_creation"] += getattr(usage, "cache_creation_input_tokens", 0) or 0
        self.usage["calls"] += 1

    def reset_usage(self) -> None:
        for k in self.usage:
            self.usage[k] = 0
        self.last = {"input": 0, "output": 0}

    def context_ratio(self) -> float:
        """当前上下文占窗口的比例（0~1），驱动进度条。"""
        if self.context_window <= 0:
            return 0.0
        return min(1.0, self.last["input"] / self.context_window)

    def usage_summary(self) -> str:
        u = self.usage
        cache = u["cache_read"] + u["cache_creation"]
        return (f"calls={u['calls']}  in={u['input']}  out={u['output']}  "
                f"total={u['input'] + u['output']}"
                + (f"  (cache={cache})" if cache else ""))

    def send_stream(
        self,
        messages: list[MessageParam],
        tools: list[ToolParam] | None = None,
        system: str | None = None,
        on_event: StreamCallback | None = None,
        thinking_budget: int | None = ...,
    ) -> _SimpleMsg:
        """流式发送，httpx 直连 SSE，实时回调 on_event。带重试。"""
        tb = self.thinking_budget if thinking_budget is ... else thinking_budget
        body: dict = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": messages,
            "stream": True,
        }
        if tb:
            # 开启思考时不能同时传 temperature（Anthropic 协议限制）
            body["thinking"] = {"type": "enabled", "budget_tokens": tb}
        else:
            body["temperature"] = self.temperature
        if tools:
            body["tools"] = tools
        if system:
            body["system"] = system

        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                return self._stream_once(body, headers, on_event)
            except httpx.HTTPStatusError as e:
                code = e.response.status_code
                # 429 / 5xx 可重试；其余 4xx 直接抛
                if code == 429 or code >= 500:
                    last_exc = e
                    if attempt < self.max_retries - 1:
                        time.sleep(self.retry_backoff * (2 ** attempt))
                        continue
                raise
            except (httpx.TimeoutException, httpx.TransportError) as e:
                last_exc = e
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_backoff * (2 ** attempt))
                    continue
                raise
        assert last_exc is not None
        raise last_exc

    def _stream_once(self, body, headers, on_event) -> _SimpleMsg:
        """单次流式请求 + SSE 解析。"""
        blocks: list[dict] = []
        cur_block: dict | None = None
        cur_index: int = -1
        tool_input_buf: str = ""

        with httpx.stream(
            "POST",
            f"{self.base_url}/messages",
            headers=headers,
            json=body,
            timeout=120,
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line or not line.startswith("data: "):
                    continue
                data_str = line[6:]  # skip "data: "
                try:
                    event = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                etype = event.get("type", "")

                # ── message_start ──
                if etype == "message_start":
                    blocks = []
                    u = event.get("message", {}).get("usage", {})
                    if u:
                        it = u.get("input_tokens", 0) or 0
                        self.usage["input"] += it
                        self.last["input"] = it          # 本次请求上下文规模
                        cr = u.get("cache_read_input_tokens", 0) or 0
                        cc = u.get("cache_creation_input_tokens", 0) or 0
                        self.usage["cache_read"] += cr
                        self.usage["cache_creation"] += cc

                # ── content_block_start ──
                elif etype == "content_block_start":
                    cb = event.get("content_block", {})
                    ctype = cb.get("type", "")
                    cur_index = len(blocks)
                    if ctype == "text":
                        cur_block = {"type": "text", "text": ""}
                    elif ctype == "thinking":
                        cur_block = {"type": "thinking", "thinking": ""}
                    elif ctype == "tool_use":
                        cur_block = {
                            "type": "tool_use",
                            "id": cb.get("id", ""),
                            "name": cb.get("name", ""),
                            "input": {},
                        }
                        tool_input_buf = ""
                        if on_event:
                            on_event(StreamEvent(
                                type="tool_use_start",
                                name=cb.get("name", ""),
                                id=cb.get("id", ""),
                            ))
                    else:
                        cur_block = None
                        continue
                    blocks.append(cur_block)

                # ── content_block_delta ──
                elif etype == "content_block_delta":
                    delta = event.get("delta", {})
                    dtype = delta.get("type", "")
                    index = event.get("index", 0)

                    if dtype == "thinking_delta":
                        text = delta.get("thinking", "")
                        if index < len(blocks) and blocks[index].get("type") == "thinking":
                            blocks[index]["thinking"] = blocks[index].get("thinking", "") + text
                        if on_event:
                            on_event(StreamEvent(type="thinking_delta", text=text))

                    elif dtype == "text_delta":
                        text = delta.get("text", "")
                        if index < len(blocks) and blocks[index].get("type") == "text":
                            blocks[index]["text"] = blocks[index].get("text", "") + text
                        if on_event:
                            on_event(StreamEvent(type="text_delta", text=text))

                    elif dtype == "input_json_delta":
                        partial = delta.get("partial_json", "")
                        tool_input_buf += partial
                        if index < len(blocks) and blocks[index].get("type") == "tool_use":
                            try:
                                blocks[index]["input"] = json.loads(tool_input_buf)
                            except json.JSONDecodeError:
                                pass
                        if on_event:
                            on_event(StreamEvent(type="tool_input_delta", input_json=partial))

                # ── content_block_stop ──
                elif etype == "content_block_stop":
                    pass

                # ── message_delta / message_stop ──
                elif etype == "message_delta":
                    u = event.get("usage", {})
                    if u:
                        ot = u.get("output_tokens", 0) or 0
                        self.usage["output"] += ot
                        self.last["output"] = ot       # 本次回复产出量
                elif etype == "message_stop":
                    self.usage["calls"] += 1

        # 过滤掉 thinking blocks（只保留 text + tool_use 给 agent）
        final_blocks = [
            b for b in blocks
            if b and b.get("type") in ("text", "tool_use")
        ]
        return _SimpleMsg(final_blocks)
