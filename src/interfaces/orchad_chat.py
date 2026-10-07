"""
chat.py — A beautiful CLI chatbot interface built with Rich.
Plug in your model's generate() function at the bottom.
"""

from datetime import datetime
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.text import Text
from rich.rule import Rule
from rich.align import Align
from rich.columns import Columns
from rich.live import Live
from rich.spinner import Spinner
from rich import box
import time

# ── Theme ──────────────────────────────────────────────────────────────────────

THEME = {
    "user_name":    "bold chartreuse3",
    "bot_name":     "bold pale_turquoise1",
    "user_bubble":  "chartreuse4",
    "bot_bubble":   "dark_slate_gray2",
    "user_text":    "white",
    "bot_text":     "white",
    "timestamp":    "grey46",
    "border":       "grey30",
    "accent":       "pale_turquoise1",
    "dim":          "grey42",
    "rule":         "grey23",
    "exit_hint":    "grey35",
}

BOT_NAME  = "Lime"        # ← your model name here
USER_NAME = "You"
SPINNER   = "dots2"

# ── Console ────────────────────────────────────────────────────────────────────

console = Console()

# ── Message rendering ──────────────────────────────────────────────────────────

def _timestamp() -> str:
    return datetime.now().strftime("%H:%M")

def render_user_message(text: str) -> None:
    ts   = Text(f" {_timestamp()} ", style=THEME["timestamp"])
    name = Text(f" {USER_NAME} ", style=THEME["user_name"])
    header = Text.assemble(name, ts)

    panel = Panel(
        Text(text, style=THEME["user_text"]),
        border_style=THEME["user_bubble"],
        box=box.ROUNDED,
        padding=(0, 2),
        title=header,
        title_align="right",
    )
    console.print(Align(panel, align="right", width=70))
    console.print()

def render_bot_message(text: str) -> None:
    ts   = Text(f"{_timestamp()} ", style=THEME["timestamp"])
    name = Text(f" {BOT_NAME} ", style=THEME["bot_name"])
    header = Text.assemble(ts, name)

    panel = Panel(
        Text(text, style=THEME["bot_text"]),
        border_style=THEME["bot_bubble"],
        box=box.ROUNDED,
        padding=(0, 2),
        title=header,
        title_align="left",
    )
    console.print(Align(panel, align="left", width=70))
    console.print()

def render_thinking() -> Live:
    """Returns a Live context showing a spinner while the model runs."""
    spinner = Spinner(SPINNER, text=Text(f"  {BOT_NAME} is thinking...", style=THEME["dim"]))
    return Live(spinner, console=console, refresh_per_second=12)

# ── Header / Footer ────────────────────────────────────────────────────────────

def render_header() -> None:
    console.print()
    title = Text(f"  🍋  {BOT_NAME}  ", style=f"bold {THEME['accent']}")
    subtitle = Text("  a minimal language model  ", style=THEME["dim"])
    console.print(
        Panel(
            Align(Text.assemble(title, "\n", subtitle), align="center"),
            border_style=THEME["border"],
            box=box.DOUBLE_EDGE,
            padding=(1, 4),
        )
    )
    console.print(
        Align(
            Text(f"  type  [bold]exit[/bold]  or  [bold]quit[/bold]  to leave  ",
                 style=THEME["exit_hint"]),
            align="center",
        )
    )
    console.print()

def render_divider() -> None:
    console.print(Rule(style=THEME["rule"]))
    console.print()

def render_goodbye() -> None:
    console.print()
    console.print(
        Align(
            Panel(
                Text("  goodbye 👋  ", style=f"bold {THEME['accent']}", justify="center"),
                border_style=THEME["border"],
                box=box.ROUNDED,
                padding=(0, 4),
            ),
            align="center",
        )
    )
    console.print()

def render_error(msg: str) -> None:
    console.print(
        Panel(
            Text(f"⚠  {msg}", style="bold red"),
            border_style="red",
            box=box.ROUNDED,
            padding=(0, 2),
        )
    )
    console.print()

# ── Input ──────────────────────────────────────────────────────────────────────

def get_user_input() -> str:
    console.print(Text(f"  {USER_NAME} › ", style=f"bold {THEME['user_name']}"), end="")
    return console.input("").strip()

# ── Model interface ────────────────────────────────────────────────────────────

def call_model(prompt: str, history: list[dict]) -> str:
    """
    Replace this with your actual model call.
    `history` is a list of {"role": "user"|"bot", "text": str} dicts.
    """
    time.sleep(0.8)   # simulate inference
    return f"[model not connected — you said: '{prompt}']"

# ── Chat loop ──────────────────────────────────────────────────────────────────

class ChatSession:
    def __init__(self):
        self.history: list[dict] = []

    def add(self, role: str, text: str) -> None:
        self.history.append({"role": role, "text": text, "time": _timestamp()})

    def run(self) -> None:
        render_header()

        while True:
            try:
                user_input = get_user_input()
            except (KeyboardInterrupt, EOFError):
                render_goodbye()
                break

            if not user_input:
                continue

            if user_input.lower() in ("exit", "quit", "q"):
                render_goodbye()
                break

            console.print()
            render_user_message(user_input)
            self.add("user", user_input)

            try:
                with render_thinking():
                    response = call_model(user_input, self.history)
            except Exception as e:
                render_error(str(e))
                continue

            render_bot_message(response)
            self.add("bot", response)
            render_divider()

# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ChatSession().run()