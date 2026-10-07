# playground.py
# Usage: python playground.py --model path/to/model.pth
# ─────────────────────────────────────────────────────────────────────────────
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import argparse
import math
import torch
import numpy as np
from rich.console import Console
from rich.text import Text
from rich.panel import Panel
from rich.table import Table
from rich import box
from rich.rule import Rule
from rich.live import Live
# IMPORT SRC
from src.tokenizer.bpe_tokenizer import BPETokenizer
from src.models.model import TransformerLM
from src.generator.generator import Generator
from src.utils.config import load_config

console = Console()

# ── Colour mapping: prob → red shade ─────────────────────────────────────────
# high prob  → dim / white   (model is confident, boring)
# low prob   → vivid red     (model was surprised)

def _prob_to_style(prob: float) -> str:
    """White text, background from transparent (dark) → dark red with surprise."""
    surprise = max(0.0, min(1.0, 1.0 - (prob * 1000)))
    r = int(180 * surprise)  # 0 → 180
    g = 0
    b = 0
    return f"white on rgb({r},{g},{b})"


DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
CONTEXT_LEN = 512   # TODO: match your model's context length


# ══════════════════════════════════════════════════════════════════════════════
# UI helpers
# ══════════════════════════════════════════════════════════════════════════════

def _print_banner():
    console.print()
    title = Text()
    title.append("◆ ", style="bold bright_yellow")
    title.append("MODEL  PLAYGROUND", style="bold white")
    title.append("◆ ", style="bold bright_yellow")
    console.print(Panel(title, style="on grey7", border_style="grey30"))
    console.print()


def _print_model_info(config):
    print('Model type: ', config.model.meta.type, config.model.meta.symbol)
    print('Model : ', config.model.meta.model)
    print('Model vesion: ', config.model.meta.version)



def _render_output(tokens_and_probs: list[tuple[int, float]]):
    probs   = [p for _, p in tokens_and_probs]
    avg_p   = sum(probs) / len(probs) if probs else 0
    entropy = -sum(p * math.log(p + 1e-9) for p in probs) / len(probs) if probs else 0
    stats   = Text()
    stats.append(f"tokens: {len(probs)}  ", style="dim white")
    # stats.append(f"avg prob: {avg_p:.3f}  ", style="dim cyan")
    # stats.append(f"avg entropy: {entropy:.3f}", style="dim cyan")
    console.print(stats)
    console.print()


# ══════════════════════════════════════════════════════════════════════════════
# Main loop
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Model playground CLI")
    parser.add_argument("--config_path",       required=True)
    args = parser.parse_args()

    _print_banner()


    # ── Init LM ───────────────────────────────────────────────────────────────────
    with console.status("[cyan]init model…[/]", spinner="dots"):
        CONFIG = load_config(args.config_path)
        DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        CONTEXT_LEN = CONFIG.model.context_len
        tokenizer = BPETokenizer(num_merges=CONFIG.tokenizer.num_merges)
        tokenizer.load(CONFIG.tokenizer.path)
        model = TransformerLM(tokenizer.vocab_size, CONFIG.model).to(DEVICE)
        model.load(CONFIG.model.path)
        generator = Generator(DEVICE, model)

    # ── load ──────────────────────────────────────────────────────────────────
    _print_model_info(CONFIG)

    console.print(Rule(style="grey30"))
    console.print(
        "  [dim white]type a prompt and press [bold white]Enter[/] to generate  ·  "
        "[bold white]ctrl+c[/] or [bold white]exit[/] to quit[/]"
    )
    console.print(Rule(style="grey30"))
    console.print()

    # ── REPL ──────────────────────────────────────────────────────────────────
    while True:
        try:
            prompt = console.input("[bold bright_yellow]▶  prompt:[/]  ")
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]bye.[/]")
            break

        prompt = prompt.strip()
        if not prompt or prompt.lower() in {"exit", "quit", "q"}:
            console.print("[dim]bye.[/]")
            break

        console.print()
        output_text      = Text()
        tokens_and_probs = []

        with Live(Panel(output_text, title="[bold yellow]Model answer[/]",
                        border_style="yellow", padding=(1, 2)),
                console=console, refresh_per_second=15) as live:
            for token, prob in generator.generate(CONTEXT_LEN, tokenizer.encode(prompt), 
                                                  CONFIG.generation.max_tokens, 
                                                  CONFIG.generation.temperature, 
                                                  CONFIG.generation.beam_width, 
                                                  CONFIG.generation.beam_depth):
                prob = np.exp(prob)
                text = tokenizer.decode([token])
                tokens_and_probs.append((text, prob))
                output_text.append(text, style=_prob_to_style(prob))
                live.update(Panel(output_text, title="[bold yellow]Model answer[/]",
                                border_style="yellow", padding=(1, 2)))

        _render_output(tokens_and_probs)


if __name__ == "__main__":
    main()