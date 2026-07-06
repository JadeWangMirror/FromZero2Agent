"""
TUI — SPARK Agent，类 Claude Code 终端界面。
"""

from __future__ import annotations

import json
import os
from datetime import datetime

from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Markdown, RichLog, Static, TextArea

from agent import Agent, create_agent

# ── Spinner ─────────────────────────────────────────────────

SPINNER = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

# ── SPARK Logo ─────────────────────────────────────────────

SPARK = """\
  [bold #FF6B35]███████╗██████╗  █████╗ ██████╗ ██╗  ██╗[/]
  [bold #FF8C5A]██╔════╝██╔══██╗██╔══██╗██╔══██╗██║ ██╔╝[/]
  [bold #FFAA77]███████╗██████╔╝███████║██████╔╝█████╔╝[/]
  [bold #FFC8A0]╚════██║██╔═══╝ ██╔══██║██╔══██╗██╔═██╗[/]
  [bold #FFD5B8]███████║██║     ██║  ██║██║  ██║██║  ██╗[/]
  [bold #FFE8D5]╚══════╝╚═╝     ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝[/]"""


def _welcome(model: str) -> str:
    return f"""\
[#FF6B35]  SPARK Agent[/]      [dim]v1.0.0[/]
[dim]  model:[/] [bold]{model}[/]
[dim]  tools:[/] [cyan]calculator[/] [dim]|[/] [cyan]get_current_time[/]
[dim]  help: [/][bold]Enter[/][dim] send  |  [/][bold]Ctrl+Q[/][dim] quit  |  [/][bold]Esc[/][dim] focus[/]"""

# ── 斜杠命令清单（用于补全栏）──────────────────────────────

COMMANDS = [
    ("/help",    "show available commands"),
    ("/model",   "switch model  e.g. /model deepseek-chat"),
    ("/temp",    "set temperature  e.g. /temp 0.7"),
    ("/tokens",  "set max_tokens  e.g. /tokens 8192"),
    ("/system",  "custom system prompt  /system reset"),
    ("/clear",   "clear conversation context"),
    ("/config",  "show config  (/config save to persist)"),
    ("/save",    "save session  /save [path]"),
    ("/load",    "load session  /load <path>"),
    ("/tools",   "list all tools (built-in + self-made)"),
    ("/quit",    "exit SPARK"),
]

# 无参命令:Enter 直接执行而非补全
_NO_ARG_COMMANDS = {"/help", "/clear", "/quit", "/tools", "/config"}


# ── App ─────────────────────────────────────────────────────


class SparkApp(App):
    """SPARK Agent TUI."""

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
        border: solid #FF6B35;
    }

    /* Agent markdown — 紧凑行距，消除内部组件默认间距 */
    SparkMd {
        height: auto;
        margin: 0;
        padding: 0 0 0 1;
        border-left: solid #FF6B35;
    }
    SparkMd > * { margin: 0; padding: 0; }

    /* 流式最终回复（渲染中） */
    .stream-text {
        height: auto;
        margin: 0;
        padding: 0 0 0 1;
        border-left: solid #FF6B35;
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
        max-height: 40vh;
        border-top: solid #30363D;
        padding: 0 1 0 1;
        background: #0D1117;
    }

    #command-palette {
        display: none;
        height: auto;
        max-height: 12;
        background: #161B22;
        border: solid #30363D;
        border-bottom: none;
        padding: 0 1;
        overflow: hidden auto;
    }
    #command-palette.visible { display: block; }

    #input-row {
        height: auto;
    }

    #input-prefix {
        width: 2;
        color: #FF6B35;
        padding: 0;
    }

    #user-input {
        width: 1fr;
        height: auto;
        min-height: 1;
        max-height: 20;
        border: none;
        background: #0D1117;
        color: #E6EDF3;
        padding: 0;
    }
    #user-input:focus { border: none; }
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

    # ── compose ──────────────────────────────────────────

    def compose(self) -> ComposeResult:
        self._conv = VerticalScroll(id="conv")
        yield self._conv

        # 输入区: 命令补全栏 + > 前缀 + TextArea
        with Vertical(id="input-area"):
            self._palette = Static("", id="command-palette")
            yield self._palette
            with Horizontal(id="input-row"):
                yield Static("[bold #FF6B35]>[/]", id="input-prefix")
                self._inp = TextArea(
                    "",
                    id="user-input",
                    tab_behavior="focus",
                    show_line_numbers=False,
                )
                yield self._inp

    def on_mount(self) -> None:
        self.title = "SPARK"
        self.sub_title = self._model

        try:
            self._agent = create_agent(api_key=self._key, config=self._cfg)
        except Exception as e:
            self._mnt(Static(f"[bold #F85149]x Agent init failed: {e}[/]"))
            return

        # 启动画面
        self._mnt(Static(SPARK))
        self._mnt(Static(_welcome(self._model), id="welcome-panel"))
        self._mnt(Static(f"[dim]config:[/] {self._cfg.summary() if self._cfg else self._model}"))
        self._mnt(Static("[dim]type [/][bold]/help[/][dim] for commands[/]"))
        self._inp.focus()

    # ── helpers ──────────────────────────────────────────

    def _mnt(self, w) -> None:
        self._conv.mount(w)
        self._conv.scroll_end(animate=False)

    def _mnt_t(self, w) -> None:
        self.call_from_thread(self._mnt, w)

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
        lines = []
        for i, (cmd, desc) in enumerate(self._pal_matches):
            if i == self._pal_idx:
                lines.append(f"[bold #FF6B35 on #30363D] ▸ {cmd:<10} {desc} [/]")
            else:
                lines.append(f"[dim]   {cmd:<10} {desc}[/]")
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
            self._sys("commands: /model <name>  /temp <0-2>  /tokens <n>  "
                      "/system <text|reset>  /clear  /config [save]  "
                      "/save [path]  /load <path>  /quit")

        # /model <name>
        elif cmd == "/model" and arg and self._agent:
            self._agent.llm.model = arg
            self._model = arg
            if self._cfg:
                self._cfg.model = arg
            self.sub_title = arg
            self._sys(f"model -> {arg}")

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

        # /clear
        elif cmd == "/clear":
            if self._agent:
                self._agent.reset_history()
                self._sys("-- context cleared --")

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
        self._mnt(Static(f"[bold #58A6FF]| YOU[/]  {text}"))

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

        # spinner
        self._spin = Static("")
        self._spin_idx = 0
        self._mnt(self._spin)
        self._spin_timer = self.set_interval(0.08, self._tick)

        self._busy = True
        self._run(text)

    def _tick(self) -> None:
        if self._spin:
            f = SPINNER[self._spin_idx % len(SPINNER)]
            self._spin_idx += 1
            self._spin.update(f"[#FF6B35]{f} Thinking...[/]")

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

            elif ev == "text_delta":
                # 第一个 token 到达时初始化流式容器
                if not stream_started[0]:
                    stream_started[0] = True
                    self.call_from_thread(self._begin_stream)
                stream_buf.append(data["text"])
                self.call_from_thread(self._stream_update, "".join(stream_buf))

            elif ev == "tool_call":
                a = ", ".join(f"{k}={v}" for k, v in data["args"].items())
                self._mnt_t(Static(
                    f"[bold #D2A8FF]  tool: {data['name']}[/] [dim]({a})[/]"
                ))

            elif ev == "tool_result":
                d = data["result"]
                if len(d) > 500:
                    d = d[:500] + "..."
                self._mnt_t(Static(f"[#3FB950]  <- {d}[/]"))

            elif ev == "sub:tool_call":
                role = data.get("role", "?")
                a = ", ".join(f"{k}={v}" for k, v in data["args"].items())
                self._mnt_t(Static(
                    f"[#D2A8FF]    [{role}] {data['name']}[/] [dim]({a})[/]"
                ))

            elif ev == "sub:tool_result":
                role = data.get("role", "?")
                d = str(data["result"])
                if len(d) > 200:
                    d = d[:200] + "..."
                self._mnt_t(Static(f"[#3FB950]    [{role}] <- {d}[/]"))

            elif ev.startswith("sub:"):
                # 子 agent 的 thinking/text 不显示,避免刷屏
                pass

        try:
            ans = self._agent.run(task, callback=cb)
        except Exception as e:
            self.call_from_thread(self._done)
            self._mnt_t(Static(f"[bold #F85149]x {e}[/]"))
            self._busy = False
            return

        self.call_from_thread(self._done)
        # 流式已完成 → 替换为完整 Markdown 渲染
        self.call_from_thread(self._finish_stream, ans)
        self._busy = False

    def _set_think(self, content: str) -> None:
        self._has_think = True
        if self._think_text:
            if len(content) > 2000:
                content = content[:2000] + "\n..."
            # RichLog 增量写入 — 清空后重写完整内容
            self._think_text.clear()
            self._think_text.write(f"[dim]{content}[/]")

    def _done(self) -> None:
        if hasattr(self, "_spin_timer") and self._spin_timer:
            self._spin_timer.stop()
        if self._spin:
            self._spin.remove()
            self._spin = None
        if not self._has_think and self._think_box:
            self._think_box.remove()
            self._think_box = None

    # ── 流式最终回复 ───────────────────────────────────────

    def _begin_stream(self) -> None:
        """第一个 text token 到达：挂载 SPARK 标签 + 流式文本容器。"""
        self._stream_label = Static("[bold #FF6B35]| SPARK[/]")
        self._stream_text = Static("", classes="stream-text")
        self._mnt(self._stream_label)
        self._mnt(self._stream_text)

    def _stream_update(self, text: str) -> None:
        """流式增量更新累积文本。"""
        if self._stream_text:
            self._stream_text.update(f"[#E6EDF3]{text}[/]")
            self._conv.scroll_end(animate=False)

    def _finish_stream(self, full_text: str) -> None:
        """流式结束：移除流式容器，挂载完整 Markdown 渲染。"""
        # 若未启动流式（极端情况，如纯工具无文本），仍挂标签
        if self._stream_label is None:
            self._stream_label = Static("[bold #FF6B35]| SPARK[/]")
            self._mnt(self._stream_label)
        # 移除流式文本容器
        if self._stream_text:
            self._stream_text.remove()
            self._stream_text = None
        # 挂载完整 Markdown
        self._mnt(Markdown(full_text, classes="SparkMd"))
        self._stream_label = None


# ── entry ──────────────────────────────────────────────────


def run_tui(api_key: str, config=None) -> None:
    SparkApp(api_key=api_key, config=config).run()
