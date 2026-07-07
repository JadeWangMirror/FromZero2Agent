"""
TUI — MIRROR Agent，类 Claude Code 终端界面。
"""

from __future__ import annotations

import json
import os
from datetime import datetime

from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Markdown, RichLog, Static, TextArea

from rich.markup import escape as _esc

# 控制字符剥离表:去掉所有 C0 控制字节(保留 \n \t \r)+ DEL。
# 流式里若混入 \x1b(ESC) 等控制字节,markup 转义管不到,会直接污染终端、
# 把屏幕搞成乱码(尤其崩溃后终端未还原时)。_esc 之前先过这道。
_CTRL_TABLE = str.maketrans(
    "", "", "".join(chr(c) for c in range(32) if chr(c) not in "\n\t\r") + "\x7f")


def _clean(s: str) -> str:
    """剥离控制字符(保留换行/制表/回车)。"""
    return s.translate(_CTRL_TABLE) if s else s


from agent import Agent, create_agent

# ── Spinner ─────────────────────────────────────────────────

SPINNER = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

# 主色：MIRROR 蓝
ACCENT = "#4493F8"
ACCENT_DIM = "#1F6FEB"

# 角色谱（刻意区分，避免全蓝分不清）：
#   用户 YOU      = 橙
#   Agent MIRROR  = 蓝（回复标签 + 回复左边框）
#   处理中 Thinking = 琥珀
COL_USER = "#F0883E"
COL_AGENT = "#58A6FF"
COL_WORK = "#D29922"

# ── MIRROR Logo（蓝色渐变）─────────────────────────────────

_MIRROR_RAW = r"""
███╗   ███╗██╗██████╗ ██████╗  ██████╗ ██████╗
████╗ ████║██║██╔══██╗██╔══██╗██╔═══██╗██╔══██╗
██╔████╔██║██║██████╔╝██████╔╝██║   ██║██████╔╝
██║╚██╔╝██║██║██╔══██╗██╔══██╗██║   ██║██╔══██╗
██║ ╚═╝ ██║██║██║  ██║██║  ██║╚██████╔╝██║  ██║
╚═╝     ╚═╝╚═╝╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═╝
"""

_LOGO_GRADIENT = ['#1F6FEB', '#2D7BF0', '#3B82F6', '#4493F8', '#5AA9F6', '#79C0FF']

MIRROR = "\n".join(f"  [bold {c}]{ln}[/]" for c, ln in zip(_LOGO_GRADIENT, _MIRROR_RAW.strip().splitlines()))

def _welcome(model: str) -> str:
    return f"""\
[{ACCENT}]  MIRROR Agent[/]      [dim]v2.0.0  self-evolving[/]
[dim]  model:[/] [bold]{model}[/]
[dim]  live context bar below the input  ·  [/][bold]/help[/][dim] for all commands[/]"""

# ── 斜杠命令清单（用于补全栏）──────────────────────────────

COMMANDS = [
    ("/help",     "show available commands"),
    ("/model",    "switch model  e.g. /model deepseek-v4-flash"),
    ("/models",   "list known models + context windows"),
    ("/effort",   "reasoning depth  off|low|medium|high"),
    ("/temp",     "set temperature  e.g. /temp 0.7"),
    ("/tokens",   "set max_tokens  e.g. /tokens 8192"),
    ("/system",   "custom system prompt  /system reset"),
    ("/context",  "show context window usage"),
    ("/compact",  "compress history now"),
    ("/cost",     "token usage this session"),
    ("/clear",    "clear conversation context"),
    ("/init",     "write MIRROR.md project context"),
    ("/config",   "show config  (/config save to persist)"),
    ("/save",     "save session  /save [path]"),
    ("/load",     "load session  /load <path>"),
    ("/tools",    "list all tools (built-in + self-made)"),
    ("/stats",    "tool usage stats  /stats [all|unused|top]"),
    ("/quit",     "exit MIRROR"),
]

# 无参命令:Enter 直接执行而非补全
_NO_ARG_COMMANDS = {
    "/help", "/clear", "/quit", "/tools", "/config", "/stats",
    "/cost", "/context", "/compact", "/models", "/init",
}

# /effort 等级 → thinking 预算（tokens）；None = 关闭扩展思考
EFFORT_LEVELS: dict[str, int | None] = {
    "off": None,
    "low": 1024,
    "medium": 4096,
    "high": 12000,
}

# 命令补全面板最多同时可见的行数；超出靠窗口滚动（▲/▼ 提示）
PALETTE_MAX_VISIBLE = 6


# ── 折叠式工具调用展示 ───────────────────────────────────────


def _is_error_result(result: str) -> bool:
    """与 tools.py 一致的失败启发式：结果以 error / tool execution error 开头。"""
    r = str(result).lstrip().lower()
    return r.startswith(("error", "tool execution error", "traceback"))


class ToolCallBlock(Widget):
    """一条工具调用的折叠展示：头行常驻（名称+截断参数+状态），点击展开看完整参数与结果。

    仿 Claude Code：默认只露一行摘要，避免长调用过程刷屏；展开后参数/结果在带边框正文里滚动。
    """

    DEFAULT_CSS = """
    ToolCallBlock {
        height: auto;
        margin: 0;
        padding: 0 0 0 1;
    }
    ToolCallBlock .tc-head {
        height: 1;
        padding: 0;
        color: #D2A8FF;
    }
    ToolCallBlock .tc-body {
        height: auto;
        max-height: 22;
        margin: 0 0 0 3;
        padding: 0 1;
        border-left: solid #30363D;
        color: #8B949E;
        background: #0D1117;
        overflow: auto;
    }
    """

    collapsed = reactive(True)

    def __init__(self, name: str, args: dict, role: str | None = None):
        super().__init__()
        self.tool_name = name
        self.tool_args = args or {}
        self.role = role
        self.result: str | None = None
        self.ok = True
        self.done = False

    # ── 文本工具 ──
    @staticmethod
    def _trunc(s: str, n: int) -> str:
        s = str(s)
        return s if len(s) <= n else s[:n] + "…"

    @staticmethod
    def _fmt_chars(n: int) -> str:
        return f"{n / 1000:.1f}k chars" if n >= 1000 else f"{n} chars"

    def _args_str(self, limit: int = 64) -> str:
        parts = []
        for k, v in self.tool_args.items():
            if isinstance(v, str):
                vs = f'"{self._trunc(v, 36)}"'
            else:
                vs = self._trunc(repr(v), 40)
            parts.append(f"{k}={vs}")
        return self._trunc(", ".join(parts), limit)

    def _status(self) -> str:
        if not self.done:
            return "  [#FFA657]●[/][dim] running…[/]"
        if not self.ok:
            return "  [#F85149]✗[/][dim] error[/]"
        n = len(self.result) if self.result else 0
        return f"  [#3FB950]✓[/][dim] {self._fmt_chars(n)}[/]"

    def _head_markup(self) -> str:
        marker = "▸" if self.collapsed else "▾"
        prefix = f"[{self.role}] " if self.role else ""
        return (f"[#6E7681]{marker}[/] [bold #D2A8FF]{prefix}{self.tool_name}[/]"
                f"  [dim]{_esc(self._args_str())}[/]{self._status()}")

    def _body_markup(self) -> str:
        lines = ["[bold #C9D1D9]arguments[/]"]
        if self.tool_args:
            for k, v in self.tool_args.items():
                lines.append(f"  [dim]{k}:[/] {_esc(self._trunc(repr(v), 500))}")
        else:
            lines.append("  [dim](none)[/]")
        if self.done:
            lines.append("")
            res = self.result or "(no output)"
            lines.append(f"[bold #C9D1D9]result[/]  "
                         f"[dim]({self._fmt_chars(len(res))})[/]")
            lines.append(_esc(self._trunc(res, 4000)))
        else:
            lines += ["", "[dim]… running[/]"]
        return "\n".join(lines)

    # ── 生命周期 ──
    def compose(self) -> ComposeResult:
        yield Static(self._head_markup(), classes="tc-head", markup=True)
        yield Static(self._body_markup(), classes="tc-body", markup=True)

    def on_mount(self) -> None:
        self._apply()

    def watch_collapsed(self, _v: bool) -> None:
        self._apply()

    def _apply(self) -> None:
        """同步头行标记与正文显隐。"""
        try:
            self.query_one(".tc-head", Static).update(self._head_markup())
            body = self.query_one(".tc-body", Static)
            body.styles.display = "none" if self.collapsed else "block"
        except Exception:
            pass

    def set_result(self, result: str) -> None:
        """记录结果（worker 线程安全：只改数据，刷新由主线程 refresh_view 触发）。"""
        self.result = str(result)
        self.ok = not _is_error_result(result)
        self.done = True

    def refresh_view(self) -> None:
        """主线程刷新头行+正文。"""
        self._apply()
        try:
            self.query_one(".tc-body", Static).update(self._body_markup())
        except Exception:
            pass

    def on_click(self, event) -> None:
        event.stop()
        self.collapsed = not self.collapsed


# ── App ─────────────────────────────────────────────────────


class MirrorApp(App):
    """MIRROR Agent TUI."""

    CSS = """
    Screen { background: #0D1117; }

    #conv {
        height: 1fr;
        padding: 0 1;
    }

    #conv > Static { height: auto; margin: 0; padding: 0; }

    /* Welcome */
    #welcome-panel {
        height: auto;
        margin: 1 0;
        padding: 1;
        border: solid #4493F8;
    }

    /* Agent markdown — 紧凑行距，消除内部组件默认间距 */
    MirrorMd {
        height: auto;
        margin: 0;
        padding: 0 0 0 1;
        border-left: solid #58A6FF;   /* agent 蓝，与 | MIRROR 标签同色 */
    }
    MirrorMd > * { margin: 0; padding: 0; }

    /* 流式最终回复（渲染中） */
    .stream-text {
        height: auto;
        margin: 0;
        padding: 0 0 0 1;
        border-left: solid #58A6FF;
    }

    /* 思考区 */
    .think-box {
        height: auto;
        max-height: 15;
        margin: 0 0 0 2;
        padding: 0 1;
        border-left: solid #484F58;
    }
    .think-content {
        height: auto;
        min-height: 1;
        color: #8B949E;
        background: #0D1117;
    }

    /* 输入区 */
    #input-area {
        dock: bottom;
        height: auto;
        max-height: 45vh;
        border-top: solid #30363D;
        padding: 0 1 0 1;
        background: #0D1117;
    }

    /* 输入框：带边框 + 聚焦蓝光 */
    #input-row {
        height: auto;
        border: solid #30363D;
        background: #0D1117;
        padding: 0 1;
        margin: 0;
    }
    #input-row:focus-within {
        border: solid #4493F8;
        background: #0F1620;
    }

    #command-palette {
        display: none;
        height: auto;
        max-height: 9;                   /* 6 可见 + ▲/▼ + 边框，绝不遮输入框 */
        background: #161B22;
        border: solid #4493F8;
        border-bottom: none;
        padding: 0 1;
        margin: 0 0 1 0;
        overflow: hidden;                /* 内容由 _render_palette 窗口化裁切 */
    }
    #command-palette.visible { display: block; }

    #input-prefix {
        width: 2;
        color: #4493F8;
        padding: 0;
    }

    #user-input {
        width: 1fr;
        height: auto;
        min-height: 1;
        max-height: 20;
        border: none;
        background: transparent;
        color: #E6EDF3;
        padding: 0;
    }
    #user-input:focus { border: none; }

    /* 动态状态条：两行——大进度条 + 紧凑指标 */
    #status-bar {
        height: 2;
        background: #161B22;
        border-top: solid #30363D;
        padding: 0 1;
        color: #8B949E;
    }
    """

    BINDINGS = [
        Binding("ctrl+q", "quit", "quit"),
        Binding("escape", "focus_input", "focus"),
        Binding("enter", "submit", "send", priority=True),
        Binding("up", "palette_up", "", priority=True),
        Binding("down", "palette_down", "", priority=True),
    ]

    def __init__(self, api_key: str, config=None, **kwargs):
        super().__init__(**kwargs)
        self._key = api_key
        self._cfg = config
        self._model = config.model if config else "deepseek-v4-pro"
        self._agent: Agent | None = None
        self._busy = False
        self._think_box: VerticalScroll | None = None
        self._think_text: RichLog | None = None
        self._has_think = False
        self._spin: Static | None = None
        self._spin_idx = 0
        # 流式最终回复
        self._stream_label: Static | None = None
        self._stream_text: Static | None = None
        # 命令补全栏
        self._pal_visible = False
        self._pal_matches: list[tuple[str, str]] = []
        self._pal_idx = 0
        # 状态条：当前 ReAct 轮次
        self._turn = 0
        self._status: Static | None = None
        # 本轮工具调用折叠块（按到达顺序；嵌套子 agent 调用也入栈）
        self._tool_blocks: list[ToolCallBlock] = []

    # ── compose ──────────────────────────────────────────

    def compose(self) -> ComposeResult:
        self._conv = VerticalScroll(id="conv")
        yield self._conv

        # 输入区: 命令补全栏 + > 前缀 + TextArea + 动态状态条
        with Vertical(id="input-area"):
            self._palette = Static("", id="command-palette")
            yield self._palette
            with Horizontal(id="input-row"):
                yield Static("[bold #4493F8]>[/]", id="input-prefix")
                self._inp = TextArea(
                    "",
                    id="user-input",
                    tab_behavior="focus",
                    show_line_numbers=False,
                )
                yield self._inp
            self._status = Static("", id="status-bar")
            yield self._status

    def on_mount(self) -> None:
        self.title = "MIRROR"
        self.sub_title = self._model

        try:
            self._agent = create_agent(api_key=self._key, config=self._cfg)
        except Exception as e:
            self._mnt(Static(f"[bold #F85149]x Agent init failed: {e}[/]"))
            return

        # 启动画面
        self._mnt(Static(MIRROR))
        self._mnt(Static(_welcome(self._model), id="welcome-panel"))
        self._mnt(Static(f"[dim]config:[/] {self._cfg.summary() if self._cfg else self._model}"))
        self._mnt(Static("[dim]type [/][bold]/help[/][dim] for commands[/]"))
        self._refresh_status()
        self._inp.focus()

    def on_resize(self, event) -> None:
        """终端尺寸变化时重绘自适应进度条。"""
        self._refresh_status()

    # ── helpers ──────────────────────────────────────────

    def _mnt(self, w) -> None:
        self._conv.mount(w)
        self._conv.scroll_end(animate=False)

    def _mnt_t(self, w) -> None:
        self.call_from_thread(self._mnt, w)

    # ── 动态状态条 ───────────────────────────────────────

    @staticmethod
    def _fmt_tok(n: int) -> str:
        """token 数值紧凑化：1.2k / 3.4M。"""
        if n >= 1_000_000:
            return f"{n / 1_000_000:.1f}M"
        if n >= 1000:
            return f"{n / 1000:.1f}k"
        return str(n)

    def _ctx_bar(self, ratio: float, width: int = 24) -> str:
        """上下文进度条：填充随阈值变色 绿→黄→红；空段用浅 shade 始终可见。"""
        ratio = max(0.0, min(1.0, ratio))
        filled = round(ratio * width)
        color = "#F85149" if ratio >= 0.85 else (
                "#D29922" if ratio >= 0.6 else "#3FB950")
        filled_part = f"[{color} bold]{'█' * filled}[/]" if filled else ""
        empty_part = (f"[#30363D]{'░' * (width - filled)}[/]"
                      if width - filled else "")
        return filled_part + empty_part

    def _build_status(self) -> str:
        """构造两行动态状态条：第一行=醒目进度条(自适应宽度)，第二行=紧凑指标。"""
        a = self._agent
        if not a or not self._status:
            return ""
        llm = a.llm
        used = a.context_tokens()
        ratio = (used / llm.context_window) if llm.context_window else 0.0
        pct = int(ratio * 100)
        pct_color = ("#F85149" if ratio >= 0.85
                     else "#D29922" if ratio >= 0.6 else "#3FB950")
        # 进度条自适应宽度，尽量占满第一行
        try:
            w = self._status.container_size.width or 80
        except Exception:
            w = 80
        bar_w = max(20, w - 16)
        bar = self._ctx_bar(ratio, width=bar_w)
        cur = self._fmt_tok(used)
        win = self._fmt_tok(llm.context_window)
        u = llm.usage
        up, down = self._fmt_tok(u["input"]), self._fmt_tok(u["output"])
        cache = u["cache_read"] + u["cache_creation"]
        cache_seg = (f"  [dim]·[/] [dim]cache[/] [#56D4DD]"
                     f"{self._fmt_tok(cache)}[/]") if cache else ""
        ntools = len(a.tools._tools) if a.tools else 0
        effort = next((k for k, v in EFFORT_LEVELS.items()
                       if v == llm.thinking_budget), "off")
        effort_seg = ("" if effort == "off"
                      else f"  [dim]·[/] [#FFA657]effort {effort}[/]")
        line1 = f"[dim]context[/] {bar} [{pct_color}]{pct:>3}%[/]"
        line2 = (
            f"[dim]{cur}/{win}[/]"
            f"  [dim]↑[/][#58A6FF]{up}[/]  [dim]↓[/][#7EE787]{down}[/]"
            f"  [dim]·[/] [dim]{u['calls']} calls[/]{cache_seg}{effort_seg}"
            f"  [dim]·[/] [#D2A8FF]{ntools} tools[/]"
            f"  [dim]·[/] [dim]turn[/] [#FFA657]{self._turn}/{a.max_turns}[/]"
        )
        return line1 + "\n" + line2

    def _refresh_status(self) -> None:
        """刷新底部状态条（主线程安全调用）。"""
        if self._status:
            try:
                self._status.update(self._build_status())
            except Exception:
                pass

    def _refresh_status_t(self) -> None:
        """从 worker 线程刷新状态条。"""
        self.call_from_thread(self._refresh_status)

    def _write_init_md(self) -> str:
        """扫描当前仓库，生成 MIRROR.md 项目上下文文件。"""
        import glob
        py_files = sorted(
            os.path.basename(p) for p in glob.glob("*.py")
            if not p.startswith("_")
        )
        model = self._model
        cfg_line = (self._cfg.summary() if self._cfg
                    else f"model={model}")
        tool_list = ""
        if self._agent and self._agent.toolforge:
            tool_list = self._agent.toolforge.list_tools()
        body = f"""# MIRROR.md — Project Context

> Auto-generated by `/init`. MIRROR loads this to understand the project.

## What this is
MIRROR Agent — a self-evolving terminal agent (Anthropic-protocol LLM +
tool system + self-evolution via ToolForge). Python + [Textual](https://textual.textualize.io) TUI.

## Modules
{chr(10).join(f"- `{f}`" for f in py_files)}

- `agent.py` — ReAct loop, streaming, history, hierarchical sub-agents.
- `tui.py` — Textual TUI (conversation + dynamic status bar + slash commands).
- `llm.py` — LLM client (httpx SSE streaming, retry, usage tracking).
- `tools.py` / `filetools.py` / `webtools.py` — built-in tool registries.
- `toolforge.py` — self-evolution engine (propose/create/review/improve tools).
- `subagents.py` — specialized sub-agent prompts (researcher, coder, …).
- `config.py` — config.json + env loading.

## Runtime
- model: `{model}`
- {cfg_line}
- launch: `python main.py` (or double-click `run.bat`)

## Slash commands
`/model` `/models` `/effort` `/temp` `/tokens` `/system` `/context`
`/compact` `/cost` `/clear` `/init` `/config` `/save` `/load` `/tools`
`/stats` `/help` `/quit`

## Registered tools
{tool_list or "(agent not initialized)"}
"""
        path = "MIRROR.md"
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(body)
            return f"wrote {path} ({len(body)} bytes)"
        except OSError as e:
            return f"x failed to write MIRROR.md: {e}"

    # ── input ────────────────────────────────────────────

    def action_submit(self) -> None:
        """Enter:命令名补全阶段→补全或直接执行;否则提交。"""
        if not self._inp.has_focus:
            return
        if self._busy:
            return
        val = self._inp.text
        stripped = val.strip()
        # 命令阶段:palette 可见且正在输入 /命令(无空格)
        if self._pal_visible and stripped.startswith("/") and " " not in stripped:
            # 无参命令 → 直接执行,免去补全后再 Enter
            if stripped in _NO_ARG_COMMANDS:
                self._inp.clear()
                self._hide_palette()
                self._go(stripped)
                return
            # 需要参数的命令 → 补全到输入框等用户填参数
            if self._pal_matches:
                self._accept_palette()
            return
        if stripped:
            self._inp.clear()
            self._hide_palette()
            self._go(stripped)

    def action_focus_input(self) -> None:
        self._hide_palette()
        self._inp.focus()

    # ── 命令补全栏 ────────────────────────────────────────

    def action_palette_up(self) -> None:
        if self._pal_visible and self._pal_matches:
            self._pal_idx = (self._pal_idx - 1) % len(self._pal_matches)
            self._render_palette()
        else:
            self._move_cursor(-1)

    def action_palette_down(self) -> None:
        if self._pal_visible and self._pal_matches:
            self._pal_idx = (self._pal_idx + 1) % len(self._pal_matches)
            self._render_palette()
        else:
            self._move_cursor(1)

    def _move_cursor(self, delta_row: int) -> None:
        try:
            r, c = self._inp.cursor_location
            self._inp.move_cursor((max(0, r + delta_row), c))
        except Exception:
            pass

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        """输入变化时更新补全栏。"""
        stripped = self._inp.text.strip()
        if stripped.startswith("/") and " " not in stripped:
            self._show_palette(stripped)
        else:
            self._hide_palette()

    def _show_palette(self, prefix: str) -> None:
        matches = [(c, d) for c, d in COMMANDS if c.startswith(prefix)]
        if not matches:
            self._hide_palette()
            return
        self._pal_matches = matches
        self._pal_idx = min(self._pal_idx, len(matches) - 1)
        self._pal_visible = True
        self._palette.add_class("visible")
        self._render_palette()

    def _hide_palette(self) -> None:
        self._pal_visible = False
        self._pal_matches = []
        self._pal_idx = 0
        try:
            self._palette.remove_class("visible")
            self._palette.update("")
        except Exception:
            pass

    def _render_palette(self) -> None:
        """渲染补全面板：只显示选中项附近的一个窗口，超出用 ▲/▼ 滚动提示。"""
        n = len(self._pal_matches)
        if n == 0:
            self._palette.update("")
            return
        maxv = PALETTE_MAX_VISIBLE
        half = maxv // 2
        start = max(0, self._pal_idx - half)
        end = min(n, start + maxv)
        start = max(0, end - maxv)          # 末尾凑不齐则前移，保证窗口尽量满
        lines: list[str] = []
        if start > 0:
            lines.append("[dim]    ▲  more ↑[/]")
        for i in range(start, end):
            cmd, desc = self._pal_matches[i]
            if i == self._pal_idx:
                lines.append(f"[bold #4493F8 on #30363D] ▸ {_esc(cmd):<10} {_esc(desc)} [/]")
            else:
                lines.append(f"[dim]   {_esc(cmd):<10} {_esc(desc)}[/]")
        if end < n:
            lines.append(f"[dim]    ▼  {n - end} more ↓[/]")
        self._palette.update("\n".join(lines))

    def _accept_palette(self) -> None:
        if not self._pal_matches:
            return
        cmd = self._pal_matches[self._pal_idx][0]
        self._inp.load_text(cmd + " ")
        try:
            self._inp.move_cursor((0, len(cmd) + 1))
        except Exception:
            pass
        self._hide_palette()
        self._inp.focus()

    def _sys(self, msg: str) -> None:
        """系统提示行。"""
        self._mnt(Static(f"[dim]{msg}[/]"))

    def _handle_command(self, text: str) -> None:
        """处理斜杠命令。"""
        parts = text.split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1].strip() if len(parts) > 1 else ""

        # /help
        if cmd in ("/help", "/?"):
            self._sys("model:   /model <name>  /models  /effort <off|low|medium|high>  "
                      "/temp <0-2>  /tokens <n>  /system <text|reset>")
            self._sys("context: /context  /compact  /cost  /clear")
            self._sys("session: /config [save]  /save [path]  /load <path>  /init  "
                      "/tools  /stats [all|unused|top]  /quit")

        # /model <name>
        elif cmd == "/model" and arg and self._agent:
            from llm import LLMClient
            known = list(LLMClient.CONTEXT_WINDOWS)
            warn = ""
            if arg not in LLMClient.CONTEXT_WINDOWS:
                warn = (f"  [dim](unknown model; valid: "
                        f"{', '.join(known)})[/]")
            self._agent.llm.model = arg
            self._agent.llm.context_window = LLMClient.window_for(arg)
            self._model = arg
            if self._cfg:
                self._cfg.model = arg
            self.sub_title = arg
            self._sys(f"model -> {arg}{warn}")
            self._refresh_status()

        # /models — 列出已知模型 + 上下文窗口
        elif cmd == "/models":
            from llm import LLMClient
            cur = self._agent.llm.model if self._agent else self._model
            lines = ["[dim]known models:[/]"]
            for m, w in LLMClient.CONTEXT_WINDOWS.items():
                mark = "[#3FB950]●[/] " if m == cur else "[dim]○[/] "
                lines.append(f"  {mark}[bold]{m}[/]  [dim]{w//1000}k ctx[/]")
            lines.append(f"[dim]default (unknown): {LLMClient._DEFAULT_WINDOW//1000}k[/]")
            self._sys("\n".join(lines))

        # /effort <off|low|medium|high> — 扩展思考预算
        elif cmd == "/effort" and self._agent:
            if not arg:
                cur = next((k for k, v in EFFORT_LEVELS.items()
                            if v == self._agent.llm.thinking_budget), "off")
                self._sys(f"effort -> {cur}  "
                          f"[dim](off|low|medium|high)[/]")
            elif arg.lower() in EFFORT_LEVELS:
                self._agent.llm.thinking_budget = EFFORT_LEVELS[arg.lower()]
                self._sys(f"effort -> {arg.lower()}")
                self._refresh_status()
            else:
                self._sys(f"x /effort: {', '.join(EFFORT_LEVELS)}")

        # /temp <value>
        elif cmd == "/temp" and arg and self._agent:
            try:
                v = float(arg)
                self._agent.llm.temperature = v
                if self._cfg:
                    self._cfg.temperature = v
                self._sys(f"temperature -> {v}")
                self._refresh_status()
            except ValueError:
                self._sys("x /temp needs a number, e.g. /temp 0.7")

        # /tokens <n>
        elif cmd == "/tokens" and arg and self._agent:
            try:
                self._agent.llm.max_tokens = int(arg)
                self._sys(f"max_tokens -> {arg}")
                self._refresh_status()
            except ValueError:
                self._sys("x /tokens needs an integer")

        # /temp <value>
        elif cmd == "/temp" and arg and self._agent:
            try:
                v = float(arg)
                self._agent.llm.temperature = v
                if self._cfg:
                    self._cfg.temperature = v
                self._sys(f"temperature -> {v}")
            except ValueError:
                self._sys("x /temp needs a number, e.g. /temp 0.7")

        # /tokens <n>
        elif cmd == "/tokens" and arg and self._agent:
            try:
                self._agent.llm.max_tokens = int(arg)
                self._sys(f"max_tokens -> {arg}")
            except ValueError:
                self._sys("x /tokens needs an integer")

        # /system <text|reset>
        elif cmd == "/system" and self._agent:
            if arg.lower() in ("reset", "clear", "default"):
                self._agent.system_prompt = None
                self._sys("system prompt -> default")
            else:
                self._agent.system_prompt = arg
                self._sys(f"system prompt set ({len(arg)} chars)")

        # /context — 上下文窗口占用详情
        elif cmd == "/context" and self._agent:
            llm = self._agent.llm
            used = self._agent.context_tokens()
            ratio = (used / llm.context_window) if llm.context_window else 0.0
            info = self._agent.context_info()
            est = self._agent.estimate_context_tokens()
            pct = f"{ratio*100:.1f}%"
            bar = self._ctx_bar(ratio, 28)
            self._sys(
                f"{bar}\n"
                f"  [dim]window:[/] {self._fmt_tok(llm.context_window)}  "
                f"[dim]in use:[/] [bold]{self._fmt_tok(used)}[/] "
                f"([bold]{pct}[/])\n"
                f"  [dim]last request:[/] {self._fmt_tok(llm.last['input'])}  "
                f"[dim]estimate:[/] ~{self._fmt_tok(est)}  "
                f"[dim]reply:[/] {self._fmt_tok(llm.last['output'])}\n"
                f"  [dim]history:[/] {info['messages']} msgs  "
                f"{self._fmt_tok(info['chars'])} chars  "
                f"[dim]auto-compact @[/] {self._fmt_tok(info['threshold'])}"
            )

        # /compact — 立即压缩历史
        elif cmd == "/compact" and self._agent:
            before, after = self._agent.compact()
            self._sys(f"compressed: {before} -> {after} messages")
            self._refresh_status()

        # /cost — 本次会话累计用量
        elif cmd == "/cost" and self._agent:
            u = self._agent.llm.usage
            total = u["input"] + u["output"]
            cache = u["cache_read"] + u["cache_creation"]
            self._sys(
                f"  [dim]input[/]   {self._fmt_tok(u['input'])}\n"
                f"  [dim]output[/]  {self._fmt_tok(u['output'])}\n"
                f"  [dim]total[/]   {self._fmt_tok(total)}  "
                f"[dim]across[/] {u['calls']} calls\n"
                + (f"  [dim]cache[/]   {self._fmt_tok(cache)} "
                   f"[dim](read {self._fmt_tok(u['cache_read'])} + "
                   f"write {self._fmt_tok(u['cache_creation'])})[/]"
                   if cache else "")
            )

        # /clear
        elif cmd == "/clear":
            if self._agent:
                self._agent.reset_history()
                self._sys("-- context cleared --")
                self._refresh_status()

        # /tools — 列出全部工具（内置 + 自造）
        elif cmd == "/tools" and self._agent:
            if self._agent.toolforge:
                out = self._agent.toolforge.list_tools()
            else:
                out = f"{len(self._agent.tools._tools)} tools registered"
            self._sys(out)

        # /stats — 工具使用统计
        elif cmd == "/stats" and self._agent and self._agent.toolforge:
            detail = arg or "all"
            self._sys(self._agent.toolforge.usage_stats(detail=detail))

        # /init — 生成项目上下文文件 MIRROR.md
        elif cmd == "/init":
            self._sys(self._write_init_md())

        # /config [save]
        elif cmd == "/config":
            if arg.lower() == "save" and self._cfg:
                self._cfg.save()
                self._sys(f"config saved -> {self._cfg._path}")
            elif self._cfg:
                self._sys(self._cfg.summary())

        # /save [path]
        elif cmd == "/save" and self._agent:
            os.makedirs("sessions", exist_ok=True)
            path = arg or f"sessions/session-{datetime.now():%Y%m%d-%H%M%S}.json"
            data = {
                "model": self._model,
                "history": self._agent.export_history(),
                "saved_at": datetime.now().isoformat(),
            }
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            self._sys(f"session saved -> {path} ({len(data['history'])} msgs)")

        # /load <path>
        elif cmd == "/load" and arg and self._agent:
            if not os.path.isfile(arg):
                self._sys(f"x not found: {arg}")
            else:
                with open(arg, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self._agent.import_history(data.get("history", []))
                self._sys(f"session loaded <- {arg} ({len(data.get('history', []))} msgs)")

        # /quit
        elif cmd in ("/quit", "/exit"):
            self.exit()

        else:
            self._sys(f"x unknown command: {cmd} (try /help)")

    def _go(self, text: str) -> None:
        # 斜杠命令
        if text.startswith("/"):
            self._handle_command(text)
            return
        if text.lower() in ("exit", "quit"):
            self.exit()
            return
        if self._agent is None:
            self._mnt(Static("[bold #F85149]x not initialized[/]"))
            return

        # 用户消息
        self._mnt(Static(f"[bold {COL_USER}]| YOU[/]  {_esc(text)}"))

        # 思考区 — RichLog 增量追加，无额外间距
        self._think_box = VerticalScroll(classes="think-box")
        self._mnt(self._think_box)
        self._think_text = RichLog(
            classes="think-content",
            highlight=True,
            markup=True,
            wrap=True,
        )
        self._think_box.mount(self._think_text)
        self._has_think = False
        self._think_shown = 0            # 新一轮思考:重置节流计数

        # spinner
        self._start_spinner()

        self._busy = True
        self._tool_blocks = []
        self._run(text)

    def _tick(self) -> None:
        if self._spin:
            f = SPINNER[self._spin_idx % len(SPINNER)]
            self._spin_idx += 1
            self._spin.update(f"[{COL_WORK}]{f} Thinking...[/]")
        # spinner 周期内同步刷新状态条（用量/进度实时变化）
        self._refresh_status()

    def _start_spinner(self) -> None:
        """挂载一个 Thinking spinner（主 agent / 救援 agent 共用）。"""
        self._spin = Static("")
        self._spin_idx = 0
        self._mnt(self._spin)
        self._spin_timer = self.set_interval(0.08, self._tick)

    def _start_spinner_t(self) -> None:
        """worker 线程安全版。"""
        self.call_from_thread(self._start_spinner)
        # spinner 周期内同步刷新状态条（用量/进度实时变化）
        self._refresh_status()

    # ── agent worker ─────────────────────────────────────

    @work(exclusive=True, thread=True)
    def _run(self, task: str) -> None:
        think_buf: list[str] = []
        stream_started = [False]
        stream_buf: list[str] = []

        def cb(ev: str, data: dict) -> None:
            if ev == "thinking":
                think_buf.append(data["text"])
                self.call_from_thread(self._set_think, "".join(think_buf))

            elif ev == "turn":
                self._turn = data.get("turn", self._turn)
                self._refresh_status_t()

            elif ev == "text_delta":
                # 第一个 token 到达时初始化流式容器
                if not stream_started[0]:
                    stream_started[0] = True
                    self.call_from_thread(self._begin_stream)
                stream_buf.append(data["text"])
                self.call_from_thread(self._stream_update, "".join(stream_buf))

            elif ev in ("tool_call", "sub:tool_call"):
                role = data.get("role") if ev.startswith("sub") else None
                block = ToolCallBlock(
                    data.get("name", "?"), data.get("args", {}), role=role)
                self._tool_blocks.append(block)
                self._mnt_t(block)

            elif ev in ("tool_result", "sub:tool_result"):
                result = str(data.get("result", ""))
                # 结果归到最近一个未完成的块（正确处理同轮多工具与子 agent 嵌套）
                blk = next((b for b in reversed(self._tool_blocks)
                            if not b.done), None)
                if blk is not None:
                    blk.set_result(result)
                    self.call_from_thread(blk.refresh_view)

            elif ev.startswith("sub:"):
                # 子 agent 的 thinking/text 不显示,避免刷屏
                pass

        try:
            ans = self._agent.run(task, callback=cb)
        except Exception as e:
            # 主 agent 通信异常 → 启动救援 agent（最小、稳定、独立空历史）
            self.call_from_thread(self._done)
            self._mnt_t(Static(
                f"[#D29922]⚠ primary agent failed:[/] [dim]{_esc(str(e).splitlines()[0][:200])}[/]"))
            self._mnt_t(Static(f"[{COL_WORK}]⏳ rescue agent engaged…[/]"))
            self._start_spinner_t()
            try:
                from agent import create_rescue_agent
                rescue = create_rescue_agent(api_key=self._key, config=self._cfg)
                ans = rescue.run(task, callback=cb)
            except Exception as e2:
                self.call_from_thread(self._done)
                self._mnt_t(Static(f"[bold #F85149]x rescue also failed:[/] "
                                   f"[dim]{_esc(str(e2).splitlines()[0][:200])}[/]"))
                self._busy = False
                return
            self.call_from_thread(self._done)
            self.call_from_thread(self._finish_stream, ans)
            self._busy = False
            return

        self.call_from_thread(self._done)
        # 流式已完成 → 替换为完整 Markdown 渲染
        self.call_from_thread(self._finish_stream, ans)
        self._busy = False

    def _set_think(self, content: str) -> None:
        self._has_think = True
        if not self._think_text:
            return
        # 节流:增量太小就跳过重绘 —— 逐 token 全量 clear+write 是 O(n²),长思考时
        # 会拖慢 UI 直到看起来"卡住不更新"。末尾几十字略迟,无伤大雅。
        if len(content) - getattr(self, "_think_shown", 0) < 40:
            return
        # 显示上限放宽(原 2000 太小,长思考一过 2000 就截断 → 像卡住);RichLog 可滚动
        shown = content if len(content) <= 10000 else content[:10000] + "\n…(truncated)"
        self._think_text.clear()
        self._think_text.write(f"[dim]{_esc(_clean(shown))}[/]")
        self._think_shown = len(content)

    def _done(self) -> None:
        if hasattr(self, "_spin_timer") and self._spin_timer:
            self._spin_timer.stop()
        if self._spin:
            self._spin.remove()
            self._spin = None
        if not self._has_think and self._think_box:
            self._think_box.remove()
            self._think_box = None
        self._turn = 0
        self._refresh_status()

    # ── 流式最终回复 ───────────────────────────────────────

    def _begin_stream(self) -> None:
        """第一个 text token 到达：挂载 MIRROR 标签 + 流式文本容器。"""
        self._stream_label = Static(f"[bold {COL_AGENT}]| MIRROR[/]")
        self._stream_text = Static("", classes="stream-text")
        self._mnt(self._stream_label)
        self._mnt(self._stream_text)

    def _stream_update(self, text: str) -> None:
        """流式增量更新累积文本。"""
        if self._stream_text:
            self._stream_text.update(f"[#E6EDF3]{_esc(_clean(text))}[/]")
            self._conv.scroll_end(animate=False)

    def _finish_stream(self, full_text: str) -> None:
        """流式结束：移除流式容器，挂载完整 Markdown 渲染。"""
        # 若未启动流式（极端情况，如纯工具无文本），仍挂标签
        if self._stream_label is None:
            self._stream_label = Static(f"[bold {COL_AGENT}]| MIRROR[/]")
            self._mnt(self._stream_label)
        # 移除流式文本容器
        if self._stream_text:
            self._stream_text.remove()
            self._stream_text = None
        # 挂载完整 Markdown
        self._mnt(Markdown(full_text, classes="MirrorMd"))
        self._stream_label = None


# ── entry ──────────────────────────────────────────────────


def run_tui(api_key: str, config=None) -> None:
    MirrorApp(api_key=api_key, config=config).run()
