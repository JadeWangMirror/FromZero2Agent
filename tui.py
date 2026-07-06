"""
TUI — SPARK Agent，类 Claude Code 终端界面。
"""

from __future__ import annotations

from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, VerticalScroll
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
    ]

    def __init__(self, api_key: str, model: str = "deepseek-v4-pro", **kwargs):
        super().__init__(**kwargs)
        self._key = api_key
        self._model = model
        self._agent: Agent | None = None
        self._busy = False
        self._think_box: VerticalScroll | None = None
        self._think_text: RichLog | None = None
        self._has_think = False
        self._spin: Static | None = None
        self._spin_idx = 0

    # ── compose ──────────────────────────────────────────

    def compose(self) -> ComposeResult:
        self._conv = VerticalScroll(id="conv")
        yield self._conv

        # 输入区: 分隔线 + > 前缀 + TextArea (自适应高度)
        with Horizontal(id="input-area"):
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
            self._agent = create_agent(api_key=self._key, model=self._model)
        except Exception as e:
            self._mnt(Static(f"[bold #F85149]x Agent init failed: {e}[/]"))
            return

        # 启动画面
        self._mnt(Static(SPARK))
        self._mnt(Static(_welcome(self._model), id="welcome-panel"))
        self._inp.focus()

    # ── helpers ──────────────────────────────────────────

    def _mnt(self, w) -> None:
        self._conv.mount(w)
        self._conv.scroll_end(animate=False)

    def _mnt_t(self, w) -> None:
        self.call_from_thread(self._mnt, w)

    # ── input ────────────────────────────────────────────

    def action_submit(self) -> None:
        """Enter 提交（仅当 TextArea 聚焦时）。"""
        if not self._inp.has_focus:
            return
        if self._busy:
            return
        text = self._inp.text.strip()
        if text:
            self._inp.clear()
            self._go(text)

    def action_focus_input(self) -> None:
        self._inp.focus()

    def _go(self, text: str) -> None:
        if text.lower() in ("exit", "quit"):
            self.exit()
            return
        if text.lower() in ("/clear", "clear"):
            if self._agent:
                self._agent.reset_history()
                self._mnt(Static("[dim]-- context cleared --[/]"))
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
        buf: list[str] = []

        def cb(ev: str, data: dict) -> None:
            if ev == "thinking":
                buf.append(data["text"])
                self.call_from_thread(self._set_think, "".join(buf))

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

        try:
            ans = self._agent.run(task, callback=cb)
        except Exception as e:
            self.call_from_thread(self._done)
            self._mnt_t(Static(f"[bold #F85149]x {e}[/]"))
            self._busy = False
            return

        self.call_from_thread(self._done)
        self._mnt_t(Static("[bold #FF6B35]| SPARK[/]"))
        self._mnt_t(Markdown(ans, classes="SparkMd"))
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


# ── entry ──────────────────────────────────────────────────


def run_tui(api_key: str, model: str = "deepseek-v4-pro") -> None:
    SparkApp(api_key=api_key, model=model).run()
