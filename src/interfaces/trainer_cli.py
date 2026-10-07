import os
import torch
from torch.utils.data import DataLoader
from datetime import datetime
from collections import deque

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn, MofNCompleteColumn, Progress,
    SpinnerColumn, TaskProgressColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn,
)
from rich.table import Table
from rich.text import Text
from rich import box


console = Console()


# ── Spark-line helper (pure unicode) ─────────────────────────────────────────
_SPARKS = "▁▂▃▄▅▆▇█"

def _sparkline(values: list[float], width: int = 28) -> str:
    if not values:
        return "─" * width
    tail   = values[-width:]
    lo, hi = min(tail), max(tail)
    spread = hi - lo or 1e-9
    return "".join(_SPARKS[int((v - lo) / spread * (len(_SPARKS) - 1))] for v in tail)


# ── Rich layout builder ───────────────────────────────────────────────────────
def _build_display(
    epoch: int,
    total_epochs: int,
    loss: float | None,
    loss_history: list[float],
    best_loss: float,
    elapsed_str: str,
    eta_str: str,
    epoch_progress: Progress,
    step_progress: Progress,
    model_name: str,
) -> Layout:

    layout = Layout()
    layout.split_column(
        Layout(name="header",  size=3),
        Layout(name="body",    size=14),
        Layout(name="footer",  size=3),
    )
    layout["body"].split_row(
        Layout(name="left",  ratio=2),
        Layout(name="right", ratio=3),
    )

    # ── Header ────────────────────────────────────────────────────────────────
    title = Text()
    title.append("◆ ", style="bold bright_yellow")
    title.append(model_name.upper(), style="bold white")
    title.append("  training session", style="dim white")
    layout["header"].update(Panel(title, style="on grey7", border_style="grey30"))

    # ── Left panel: stats ─────────────────────────────────────────────────────
    stats = Table.grid(padding=(0, 2))
    stats.add_column(style="dim cyan", justify="right")
    stats.add_column(style="bold white")

    loss_str  = f"{loss:.6f}" if loss is not None else "—"
    delta_str = "—"
    if len(loss_history) >= 2:
        delta = loss_history[-1] - loss_history[-2]
        arrow = "▼" if delta < 0 else "▲"
        color = "green" if delta < 0 else "red"
        delta_str = Text(f"{arrow} {abs(delta):.6f}", style=color)

    stats.add_row("epoch",    f"{epoch} / {total_epochs}")
    stats.add_row("loss",     loss_str)
    stats.add_row("Δ loss",   delta_str)
    stats.add_row("best",     f"{best_loss:.6f}" if best_loss < float('inf') else "—")
    stats.add_row("elapsed",  elapsed_str)
    stats.add_row("eta",      eta_str)

    layout["left"].update(
        Panel(stats, title="[bold cyan]metrics[/]", border_style="cyan", box=box.SIMPLE_HEAD)
    )

    # ── Right panel: sparkline + progress ─────────────────────────────────────
    spark     = _sparkline(loss_history)
    spark_txt = Text()
    spark_txt.append("loss curve  ", style="dim white")
    spark_txt.append(spark, style="bright_cyan")
    if loss_history:
        spark_txt.append(f"  {loss_history[-1]:.4f}", style="dim white")

    prog_group = Table.grid(padding=(1, 0))
    prog_group.add_row(spark_txt)
    prog_group.add_row(epoch_progress)
    prog_group.add_row(step_progress)

    layout["right"].update(
        Panel(prog_group, title="[bold cyan]progress[/]", border_style="cyan", box=box.SIMPLE_HEAD)
    )

    # ── Footer ────────────────────────────────────────────────────────────────
    foot = Text(justify="center")
    foot.append("ctrl+c", style="bold yellow")
    foot.append(" to interrupt  ·  checkpoints → ", style="dim white")
    foot.append(f"models/{model_name}/", style="dim cyan")
    layout["footer"].update(Panel(foot, style="on grey7", border_style="grey30"))

    return layout


# ── Trainer ───────────────────────────────────────────────────────────────────
class Trainer:
    def __init__(self, device, model, optimizer, loader: DataLoader):
        self.device    = device
        self.model     = model
        self.optimizer = optimizer
        self.loader    = loader

    # ── unchanged ─────────────────────────────────────────────────────────────
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        for x, y in self.loader:
            x, y    = x.to(self.device), y.to(self.device)
            _, loss = self.model.train_step(x, y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(self.loader)

    def save(self, path: str):
        self.model.save(path)

    # ── main entry point ──────────────────────────────────────────────────────
    def train(self,
              epochs: int,
              print_gap: int | None = None,
              save_gap:  int | None = None,
              save: bool = True):

        starting_time   = datetime.now()
        out_folder_path = f"models/{self.model.name}/"
        os.makedirs(out_folder_path, exist_ok=True)

        loss_history: list[float] = []
        best_loss = float("inf")

        # ── progress bars ─────────────────────────────────────────────────────
        epoch_progress = Progress(
            TextColumn("[dim white]epochs "),
            BarColumn(bar_width=22, complete_style="cyan", finished_style="bright_cyan"),
            MofNCompleteColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            expand=False,
        )
        step_progress = Progress(
            TextColumn("[dim white]batch  "),
            BarColumn(bar_width=22, complete_style="yellow", finished_style="bright_yellow"),
            MofNCompleteColumn(),
            SpinnerColumn("dots", style="yellow"),
            expand=False,
        )
        epoch_task = epoch_progress.add_task("", total=epochs)
        step_task  = step_progress.add_task("",  total=len(self.loader))

        def _elapsed_str() -> str:
            elapsed       = datetime.now() - starting_time
            total_seconds = int(elapsed.total_seconds())
            h, r          = divmod(total_seconds, 3600)
            m, s          = divmod(r, 60)
            return f"{h:02d}h {m:02d}m {s:02d}s"

        def _eta_str(epoch_idx: int) -> str:
            if epoch_idx == 0:
                return "—"
            elapsed  = (datetime.now() - starting_time).total_seconds()
            per_ep   = elapsed / epoch_idx
            remaining = per_ep * (epochs - epoch_idx)
            h, r     = divmod(int(remaining), 3600)
            m, s     = divmod(r, 60)
            return f"{h:02d}h {m:02d}m {s:02d}s"

        current_loss: list[float | None] = [None]  # mutable cell for closure

        def _render(epoch_idx: int) -> Layout:
            return _build_display(
                epoch        = epoch_idx,
                total_epochs = epochs,
                loss         = current_loss[0],
                loss_history = loss_history,
                best_loss    = best_loss,
                elapsed_str  = _elapsed_str(),
                eta_str      = _eta_str(epoch_idx),
                epoch_progress = epoch_progress,
                step_progress  = step_progress,
                model_name   = self.model.name,
            )

        # ── live loop ─────────────────────────────────────────────────────────
        with Live(_render(0), console=console, refresh_per_second=10, screen=False) as live:
            for epoch in range(epochs):
                step_progress.reset(step_task)

                # patch train_epoch to tick the step bar
                self.model.train()
                total_loss = 0
                for x, y in self.loader:
                    x, y    = x.to(self.device), y.to(self.device)
                    _, loss = self.model.train_step(x, y)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    total_loss += loss.item()
                    step_progress.advance(step_task)
                    live.update(_render(epoch + 1))

                loss = total_loss / len(self.loader)
                loss_history.append(loss)
                current_loss[0] = loss
                if loss < best_loss:
                    best_loss = loss

                epoch_progress.advance(epoch_task)
                live.update(_render(epoch + 1))

                # ── save logic (unchanged) ────────────────────────────────────
                if save and save_gap and ((epoch % save_gap == 0) or (epoch == epochs - 1)):
                    path = out_folder_path + self.model.save_path() + f"_{epoch+1}.pth"
                    self.save(path)

            if save:
                path = out_folder_path + self.model.save_path() + f"_{epochs}.pth"
                self.save(path)

        # ── final summary ─────────────────────────────────────────────────────
        console.print()
        summary = Table(box=box.MINIMAL_DOUBLE_HEAD, border_style="cyan", show_header=False)
        summary.add_column(style="dim cyan",   justify="right")
        summary.add_column(style="bold white", justify="left")
        summary.add_row("model",      self.model.name)
        summary.add_row("epochs",     str(epochs))
        summary.add_row("final loss", f"{loss_history[-1]:.6f}" if loss_history else "—")
        summary.add_row("best loss",  f"{best_loss:.6f}")
        summary.add_row("total time", _elapsed_str())
        console.print(Panel(summary, title="[bold bright_yellow]◆ training complete[/]", border_style="yellow"))