"""Renderiza alerta_JS001.md em PNG para o README."""

from __future__ import annotations

import re
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "processed" / "alerts" / "alerta_JS001.md"
OUT = ROOT / "docs" / "assets" / "alerta_JS001.png"


def _clean_body(md: str) -> str:
    # Remove front matter YAML
    if md.startswith("---"):
        parts = md.split("---", 2)
        if len(parts) >= 3:
            md = parts[2]
    # Drop payload JSON section
    if "## Payload estruturado" in md:
        md = md.split("## Payload estruturado", 1)[0]
    lines: list[str] = []
    for raw in md.splitlines():
        line = raw.rstrip()
        if line.strip().startswith("```"):
            continue
        line = re.sub(r"^\*\*(.+?)\*\*\s*$", r"\1", line)
        line = re.sub(r"\*\*(.+?)\*\*", r"\1", line)
        line = re.sub(r"^\*\s+", "• ", line)
        line = re.sub(r"^\*\s+", "• ", line.lstrip())
        if line.strip().startswith("---"):
            continue
        lines.append(line)
    # Compact blank lines
    out: list[str] = []
    blank = False
    for ln in lines:
        if not ln.strip():
            if not blank:
                out.append("")
            blank = True
        else:
            out.append(ln)
            blank = False
    return "\n".join(out).strip()


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    text = _clean_body(SRC.read_text(encoding="utf-8"))
    # Limit length for readable card
    max_chars = 2200
    if len(text) > max_chars:
        text = text[:max_chars].rsplit("\n", 1)[0] + "\n…"

    wrapped_lines: list[str] = []
    for paragraph in text.splitlines():
        if not paragraph.strip():
            wrapped_lines.append("")
            continue
        wrapped_lines.extend(textwrap.wrap(paragraph, width=92) or [""])

    # Figure height from line count
    n = len(wrapped_lines)
    fig_h = max(8.5, 1.6 + n * 0.22)
    fig, ax = plt.subplots(figsize=(11, fig_h), dpi=140)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("#f4f6f8")

    card = FancyBboxPatch(
        (0.03, 0.03),
        0.94,
        0.94,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=1.5,
        edgecolor="#1f4e5f",
        facecolor="#ffffff",
        transform=ax.transAxes,
        clip_on=False,
    )
    ax.add_patch(card)

    # Header bar
    header = FancyBboxPatch(
        (0.03, 0.88),
        0.94,
        0.09,
        boxstyle="round,pad=0.01,rounding_size=0.02",
        linewidth=0,
        facecolor="#b42318",
        transform=ax.transAxes,
        clip_on=False,
    )
    ax.add_patch(header)
    ax.text(
        0.5,
        0.925,
        "ALERTA CLÍNICO EDUCACIONAL — JS-001  ·  RISCO ALTO",
        ha="center",
        va="center",
        fontsize=13,
        color="white",
        fontweight="bold",
        transform=ax.transAxes,
    )

    body = "\n".join(wrapped_lines)
    ax.text(
        0.06,
        0.84,
        body,
        ha="left",
        va="top",
        fontsize=8.5,
        color="#1a1a1a",
        family="DejaVu Sans",
        linespacing=1.35,
        transform=ax.transAxes,
        wrap=False,
    )

    ax.text(
        0.5,
        0.045,
        "Fonte: data/processed/alerts/alerta_JS001.md  ·  Uso educacional — não substitui avaliação profissional",
        ha="center",
        va="center",
        fontsize=7,
        color="#555555",
        transform=ax.transAxes,
    )

    fig.savefig(OUT, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
