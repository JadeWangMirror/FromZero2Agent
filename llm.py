"""
LLM 客户端 — 封装 Anthropic 协议调用（DeepSeek 兼容端点）。
支持流式输出（httpx 直连 SSE），实时回调 thinking 增量。
"""

from __future__ import annotations

import json
import re
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


# ── DSML 工具调用兜底解析 ──────────────────────────────────
# 某些 DeepSeek 流式端点(偶发)不把工具调用转成原生 Anthropic tool_use 块,而是
# 把模型原生的 DSML 工具调用文本塞进 text 或 thinking:
#   <｜DSML｜tool_calls>          ← 注意竖线是全角 ｜(U+FF5C),不是半角 |
#     <｜DSML｜invoke name="read_file">
#       <｜DSML｜parameter name="path" string="true">/x/y</｜DSML｜parameter>
#     </｜DSML｜invoke>
#   </｜DSML｜tool_calls>
# 不解析的话,工具永远不会被执行 → agent 卡死/会话断开。这里把它转回 tool_use 块。
# 关键坑:端点实际吐的是【全角】控制符(｜ < ／ > 等 CJK 渲染),早期正则只认半角 |,
# 检测永远失败、DSML 漏成文本 —— 这就是"DSML 打断会话"的真凶。故全部用字符类兼容半/全角。
# 同时:前缀可选、单双引号皆可、扫 text 与 thinking 两处。

_LB = r"[<＜]"      # <  或 全角＜(U+FF1C)
_RB = r"[>＞]"      # >  或 全角＞(U+FF1E)
_BS = r"[/／]"      # /  或 全角／(U+FF0F)
_PV = r"[|｜]+"     # | 或 全角｜(U+FF5C),可多个(端点实际吐 ｜｜DSML｜｜ 双竖线)
_SP = r"[\s　]*"    # 空白(含全角空格 U+3000)
_DSML_PREF = rf"(?:{_PV}{_SP}DSML{_SP}{_PV}{_SP})?"   # 可选的 "｜DSML｜" 外壳
_DSML_OPEN_INVOKE = re.compile(
    rf"{_LB}{_SP}{_DSML_PREF}invoke\s+name\s*=\s*[\"']([^\"']+)[\"']\s*{_RB}", re.S)
_DSML_CLOSE_INVOKE = re.compile(
    rf"{_LB}{_SP}{_BS}{_SP}{_DSML_PREF}invoke\s*{_RB}", re.S)
_DSML_OPEN_PARAM = re.compile(
    rf"{_LB}{_SP}{_DSML_PREF}parameter\s+name\s*=\s*[\"']([^\"']+)[\"'][^>＞]*{_RB}", re.S)
_DSML_CLOSE_PARAM = re.compile(
    rf"{_LB}{_SP}{_BS}{_SP}{_DSML_PREF}parameter\s*{_RB}", re.S)
# 检测信号: DSML 外壳 / invoke name= / tool_calls —— 任一出现即认定有工具调用文本
_DSML_ANY = re.compile(
    rf"{_LB}{_SP}(?:{_DSML_PREF})(?:invoke\s+name\s*=|tool_calls\b)", re.I)


def _parse_dsml_tool_calls(text: str) -> list[dict]:
    """从文本提取所有 DSML invoke → [{name, input}, ...]。无则空。"""
    calls: list[dict] = []
    pos = 0
    while True:
        mo = _DSML_OPEN_INVOKE.search(text, pos)
        if not mo:
            break
        cmo = _DSML_CLOSE_INVOKE.search(text, mo.end())
        if not cmo:
            break
        body = text[mo.end():cmo.start()]
        params: dict = {}
        for pm in _DSML_OPEN_PARAM.finditer(body):
            cpm = _DSML_CLOSE_PARAM.search(body, pm.end())
            raw = (body[pm.end():cpm.start()] if cpm else "").strip()
            try:
                params[pm.group(1)] = json.loads(raw)   # 数值/布尔自动还原
            except (json.JSONDecodeError, ValueError):
                params[pm.group(1)] = raw               # 字符串原样
        calls.append({"name": mo.group(1), "input": params})
        pos = cmo.end()
    return calls


def _dsml_to_tool_use_blocks(text: str) -> tuple[list[dict], str] | None:
    """text 含 DSML 工具调用 → (tool_use 块列表, DSML 前的自然语言前缀);
    否则 None(走原生路径)。"""
    if not _DSML_ANY.search(text):
        return None
    calls = _parse_dsml_tool_calls(text)
    if not calls:
        return None
    first = _DSML_ANY.search(text)
    pre = text[:first.start()].strip() if first else ""
    blocks = [{"type": "tool_use", "id": f"dsml-{i}",
               "name": c["name"], "input": c["input"]}
              for i, c in enumerate(calls)]
    return blocks, pre


def _dsml_debug_log(raw_text: str, block_types: list, has_native: bool,
                    parsed: list) -> None:
    """每次出现 DSML 就把原文落盘 —— 用于定位"仍打断会话"的真实形态
    (格式是否不同?是否与原生 tool_use 共存?解析为何没命中?)。失败静默。"""
    import os
    import time as _t
    try:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "dsml_debug.log")
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"\n===== {_t.strftime('%Y-%m-%d %H:%M:%S')} =====\n")
            f.write(f"block_types={block_types} has_native_tool_use={has_native} "
                    f"parsed_calls={len(parsed)}\n")
            f.write(f"raw_text ({len(raw_text)} chars):\n{raw_text[:4000]}\n")
            if parsed:
                f.write(f"parsed: {parsed}\n")
    except Exception:
        pass


def _log_partial_dsml(blocks) -> None:
    """流式被异常打断时,若已收到含工具调用文本的片段,落盘定位。
    否则 _finalize_blocks 根本没机会运行,DSML 永远抓不到。失败静默。"""
    try:
        bs = blocks or []
        raw = "".join(
            (b.get("text", "") or b.get("thinking", "") or "")
            for b in bs if isinstance(b, dict))
        if raw and _DSML_ANY.search(raw):
            _dsml_debug_log("[stream interrupted — partial]\n" + raw[:4000],
                            [b.get("type") for b in bs], False, [])
    except Exception:
        pass


def _finalize_blocks(blocks: list[dict]) -> list[dict]:
    """过滤 thinking 块;处理 DSML 工具调用文本。

    text 或 thinking 里出现工具调用文本(无论是否与原生 tool_use 共存、解析是否成功):
      1) 落盘原文(见 _dsml_debug_log)用于定位;
      2) 剥掉该噪声段(不把 DSML 当回复存进历史/显示);
      3) 若无原生 tool_use 且解析成功 → 追加解析出的 tool_use 块。
    无则原样返回(原生路径零干扰)。"""
    final = [b for b in blocks if b and b.get("type") in ("text", "tool_use")]
    txt = "".join(b.get("text", "") for b in final if b.get("type") == "text")
    think_txt = "".join(b.get("thinking", "") for b in blocks
                        if isinstance(b, dict) and b.get("type") == "thinking")
    has_native = any(b.get("type") == "tool_use" for b in final)
    # 优先扫 text;text 里没有才看 thinking(模型偶发把工具调用塞进思考)
    if _DSML_ANY.search(txt):
        scan, source = txt, "text"
    elif _DSML_ANY.search(think_txt):
        scan, source = think_txt, "thinking"
    else:
        scan, source = "", ""
    if scan:
        parsed = _parse_dsml_tool_calls(scan)
        _dsml_debug_log(f"[source={source}]\n--TEXT--\n{txt}\n--THINKING--\n{think_txt}",
                        [b.get("type") for b in blocks], has_native, parsed)
        first = _DSML_ANY.search(scan)
        # text 里出现 → 保留 DSML 之前的干净文本;thinking 里出现 → 保留正文 text
        pre = (scan[:first.start()] if source == "text" else txt).strip()
        tu = [b for b in final if b.get("type") == "tool_use"]
        if parsed and not has_native:
            tu = tu + [{"type": "tool_use", "id": f"dsml-{i}",
                        "name": c["name"], "input": c["input"]}
                       for i, c in enumerate(parsed)]
        final = ([{"type": "text", "text": pre}] if pre else []) + tu
    return final


# ── LLM Client ────────────────────────────────────────────


class LLMClient:
    """Anthropic 协议 LLM 客户端，用 httpx 处理 SSE 流。"""

    # 各模型上下文窗口（tokens）；未命中则取默认值
    # deepseek-v4-pro/flash 实际为 1M 窗口(此前误配 128k,白白浪费 87% 容量)
    CONTEXT_WINDOWS: dict[str, int] = {
        "deepseek-v4-pro": 1_000_000,
        "deepseek-v4-flash": 1_000_000,
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
                _log_partial_dsml(getattr(self, "_dbg_stream_blocks", None))
                raise
            except (httpx.TimeoutException, httpx.TransportError) as e:
                last_exc = e
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_backoff * (2 ** attempt))
                    continue
                _log_partial_dsml(getattr(self, "_dbg_stream_blocks", None))
                raise
        assert last_exc is not None
        _log_partial_dsml(getattr(self, "_dbg_stream_blocks", None))
        raise last_exc

    def _http_error(self, resp, body: dict) -> httpx.HTTPStatusError:
        """构造带响应体 + 针对性诊断的 HTTP 错误，避免只看到干巴巴的 400。"""
        try:
            raw = resp.read().decode("utf-8", errors="replace")
        except Exception:
            raw = ""
        hint = raw.strip()
        try:
            j = json.loads(raw)
            if isinstance(j, dict):
                err = j.get("error")
                if isinstance(err, dict):
                    hint = (err.get("message") or err.get("type") or raw).strip()
                elif isinstance(j.get("message"), str):
                    hint = j["message"].strip()
        except Exception:
            pass
        diag = self._diagnose_400(body) if resp.status_code == 400 else ""
        msg = (f"HTTP {resp.status_code} {resp.reason_phrase} "
               f"from {self.base_url}\n  → {hint[:400]}{diag}")
        return httpx.HTTPStatusError(msg, request=resp.request, response=resp)

    def _diagnose_400(self, body: dict) -> str:
        """针对 DeepSeek anthropic 端点的 400 常见原因给出修复提示。"""
        if "deepseek" not in self.base_url.lower():
            return ""          # 非 DeepSeek 端点不做揣测
        hints = ""
        if body.get("thinking"):
            hints += ("\n  · 请求带了 `thinking`(扩展思考) 字段，DeepSeek 端点可能不支持 —— "
                      "运行 /effort off 后重试")
        model = body.get("model", "")
        valid = ("deepseek-v4-pro", "deepseek-v4-flash")
        if model and model not in valid:
            hints += (f"\n  · 模型名 `{model}` 该端点不认；只支持 "
                      f"deepseek-v4-pro / deepseek-v4-flash —— 运行 /model deepseek-v4-pro")
        return hints

    def _stream_once(self, body, headers, on_event) -> _SimpleMsg:
        """单次流式请求 + SSE 解析。"""
        blocks: list[dict] = []
        self._dbg_stream_blocks = blocks  # 异常时 send_stream 可据此抓 DSML 片段
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
            if resp.status_code >= 400:
                raise self._http_error(resp, body)
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
                        # 部分端点（如 DeepSeek 兼容层）直接在 start 里给出完整 input
                        inline = cb.get("input")
                        cur_block = {
                            "type": "tool_use",
                            "id": cb.get("id", ""),
                            "name": cb.get("name", ""),
                            "input": inline if isinstance(inline, dict) and inline else {},
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
                    # 兜底：最后一段 delta 可能未触发完整 json.loads，stop 时再解析一次
                    idx = event.get("index", cur_index)
                    if (idx < len(blocks)
                            and blocks[idx].get("type") == "tool_use"
                            and tool_input_buf):
                        try:
                            blocks[idx]["input"] = json.loads(tool_input_buf)
                        except json.JSONDecodeError:
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

        # 过滤 thinking 块 + DSML 工具调用兜底(见 _finalize_blocks)
        return _SimpleMsg(_finalize_blocks(blocks))
