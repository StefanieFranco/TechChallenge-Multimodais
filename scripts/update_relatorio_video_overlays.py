"""Atualiza §4.7 do Relatorio: GT CORRETO/INCORRETO + overlays de pose."""

from __future__ import annotations

import base64
import json
import sys
from io import StringIO
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.video.anomaly_report import VIDEO_GT_LABELS, analyze_videos_dir

NB = ROOT / "notebooks" / "Relatorio.ipynb"


INTRO_MD = """## 4.7 Análise de movimento (vídeo — fisioterapia)

Clipes próprios em `data/raw/videos/` simulam a sessão de fisioterapia do paciente fictício **J.S. (pós-AVC)**.

**Rótulos do exercício (ground truth):**

| Vídeo | Veredito oficial |
|---|---|
| `WhatsApp Video 2026-07-27 at 07.48.35.mp4` | **CORRETO** |
| `WhatsApp Video 2026-07-27 at 22.03.27.mp4` | **CORRETO** |
| `WhatsApp Video 2026-07-27 at 07.48.42.mp4` | **INCORRETO** |
| `WhatsApp Video 2026-07-27 at 22.03.28.mp4` | **INCORRETO** |

**Pipeline:** MediaPipe Pose Landmarker (~3 fps) → landmarks + **alertas de forma**
(cotovelo acima do ombro; joelho **ultrapassa a ponta do pé** — anel azul) em vermelho nos frames → score de assimetria L/R (auxiliar) → **veredito oficial = GT**.

YOLOv8 permanece como extensão futura (pessoa/objetos).

> **Aviso:** análise educacional — **não** é diagnóstico clínico.
"""

CODE_CELL = """from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display

ROOT = Path("..").resolve()
if not (ROOT / "src").exists():
    ROOT = Path(".").resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.video.anomaly_report import VIDEO_GT_LABELS, analyze_videos_dir

VIDEOS = ROOT / "data" / "raw" / "videos"
print("Pasta:", VIDEOS)
print("GT cadastrado:", VIDEO_GT_LABELS)
print("-" * 60)

reports = analyze_videos_dir(VIDEOS, sample_fps=3.0, with_overlays=True, max_overlay_frames=6)

rows = []
for r in reports:
    achados = "; ".join(r["achados"])
    print(f"Vídeo              : {r['video_name']}")
    print(f"Veredito (GT)      : {r['veredito']}")
    print(f"Heurística         : {r.get('veredito_heuristico')}  (score={r['score']:.4f})")
    print(f"Detecção pose      : {r['pose_meta']['detection_rate']:.1%}")
    print(f"Achados            : {achados}")
    print(f"Alertas de forma   : {r.get('form_alerts') or []}")
    print(f"Frames c/ alerta   : {r.get('n_alert_frames', 0)}")
    print(f"Overlays           : {r.get('overlay_dir')}")
    print("-" * 60)
    rows.append({
        "video": r["video_name"],
        "veredito_GT": r["veredito"],
        "heuristico": r.get("veredito_heuristico"),
        "score": r["score"],
        "deteccao_pose": round(r["pose_meta"]["detection_rate"], 3),
        "alertas_forma": "; ".join(r.get("form_alerts") or []) or "-",
        "achados": achados,
    })

df_video = pd.DataFrame(rows)
display(df_video)
print(r["aviso"])

# Exibe grade de frames com landmarks por vídeo
for r in reports:
    paths = r.get("overlay_paths") or []
    if not paths:
        print(f"(sem overlays) {r['video_name']}")
        continue
    n = len(paths)
    cols = min(3, n)
    rows_n = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows_n, cols, figsize=(4 * cols, 3.2 * rows_n))
    axes = axes.ravel() if n > 1 else [axes]
    for ax in axes:
        ax.axis("off")
    for i, p in enumerate(paths):
        img = plt.imread(p)
        axes[i].imshow(img)
        axes[i].set_title(f"f{i}", fontsize=9)
        axes[i].axis("off")
    fig.suptitle(f"{r['veredito']} — {r['video_name']}", fontsize=11)
    plt.tight_layout()
    plt.show()
"""

INTERPRET_MD = """**Leitura:** 2 vídeos **CORRETOS** e 2 **INCORRETOS** conforme o exercício. O anel **azul** marca a ponta do pé (limite do joelho). Braço/cotovelo só alerta se passar **claramente** acima da linha do ombro (elevação lateral até o ombro = OK). O veredito oficial segue o GT.
"""


def md_cell(text: str) -> dict:
    lines = text.strip("\n").split("\n")
    source = [ln + "\n" for ln in lines[:-1]] + ([lines[-1] + "\n"] if lines else [])
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def code_cell(text: str) -> dict:
    lines = text.strip("\n").split("\n")
    source = [ln + "\n" for ln in lines[:-1]] + ([lines[-1] + "\n"] if lines else [])
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source,
    }


def png_b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def main() -> None:
    print("Rodando análise + overlays...")
    videos = ROOT / "data" / "raw" / "videos"
    reports = analyze_videos_dir(
        videos, sample_fps=3.0, with_overlays=True, max_overlay_frames=6
    )

    buf = StringIO()
    print("Pasta:", videos, file=buf)
    print("GT cadastrado:", VIDEO_GT_LABELS, file=buf)
    print("-" * 60, file=buf)
    rows = []
    for r in reports:
        achados = "; ".join(r["achados"])
        print(f"Vídeo              : {r['video_name']}", file=buf)
        print(f"Veredito (GT)      : {r['veredito']}", file=buf)
        print(
            f"Heurística         : {r.get('veredito_heuristico')}  (score={r['score']:.4f})",
            file=buf,
        )
        print(f"Detecção pose      : {r['pose_meta']['detection_rate']:.1%}", file=buf)
        print(f"Achados            : {achados}", file=buf)
        print(f"Alertas de forma   : {r.get('form_alerts') or []}", file=buf)
        print(f"Frames c/ alerta   : {r.get('n_alert_frames', 0)}", file=buf)
        print(f"Overlays           : {r.get('overlay_dir')}", file=buf)
        print("-" * 60, file=buf)
        rows.append(
            {
                "video": r["video_name"],
                "veredito_GT": r["veredito"],
                "heuristico": r.get("veredito_heuristico"),
                "score": r["score"],
                "deteccao_pose": round(r["pose_meta"]["detection_rate"], 3),
                "alertas_forma": "; ".join(r.get("form_alerts") or []) or "-",
                "achados": achados,
            }
        )
    df_video = pd.DataFrame(rows)
    print(reports[0]["aviso"], file=buf)
    text_out = buf.getvalue()
    print(text_out)

    # Build figure outputs: one display_data per video grid + table
    outputs: list[dict] = [
        {
            "output_type": "stream",
            "name": "stdout",
            "text": text_out.splitlines(keepends=True),
        },
        {
            "output_type": "display_data",
            "data": {
                "text/html": [df_video.to_html(index=False)],
                "text/plain": [df_video.to_string(index=False)],
            },
            "metadata": {},
        },
    ]

    for r in reports:
        paths = [Path(p) for p in (r.get("overlay_paths") or []) if Path(p).exists()]
        if not paths:
            continue
        n = len(paths)
        cols = min(3, n)
        rows_n = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows_n, cols, figsize=(4 * cols, 3.2 * rows_n))
        if n == 1:
            axes = [axes]
        else:
            axes = axes.ravel()
        for ax in axes:
            ax.axis("off")
        for i, p in enumerate(paths):
            img = plt.imread(p)
            axes[i].imshow(img)
            axes[i].set_title(f"f{i}", fontsize=9)
            axes[i].axis("off")
        fig.suptitle(f"{r['veredito']} — {r['video_name']}", fontsize=11)
        plt.tight_layout()
        import io

        bio = io.BytesIO()
        fig.savefig(bio, format="png", dpi=110, bbox_inches="tight")
        plt.close(fig)
        b64 = base64.b64encode(bio.getvalue()).decode("ascii")
        outputs.append(
            {
                "output_type": "display_data",
                "data": {"image/png": b64, "text/plain": [f"<overlay {r['video_name']}>"]},
                "metadata": {},
            }
        )

    # Patch notebook cells for section 4.7
    nb = json.loads(NB.read_text(encoding="utf-8"))
    # Find intro markdown of 4.7 and following code + interpret markdown
    idx_intro = None
    for i, cell in enumerate(nb["cells"]):
        src = "".join(cell.get("source", []))
        if cell["cell_type"] == "markdown" and "4.7 Análise de movimento" in src:
            idx_intro = i
            break
    if idx_intro is None:
        raise RuntimeError("Seção 4.7 não encontrada")

    # Replace intro, code, and next markdown interpret if present
    nb["cells"][idx_intro] = md_cell(INTRO_MD)

    # Find code cell with analyze_videos_dir after intro
    code_idx = None
    for j in range(idx_intro + 1, min(idx_intro + 5, len(nb["cells"]))):
        src = "".join(nb["cells"][j].get("source", []))
        if nb["cells"][j]["cell_type"] == "code" and "analyze_videos_dir" in src:
            code_idx = j
            break
    if code_idx is None:
        # insert code after intro
        nb["cells"].insert(idx_intro + 1, code_cell(CODE_CELL))
        code_idx = idx_intro + 1
    else:
        cell = code_cell(CODE_CELL)
        cell["outputs"] = outputs
        cell["execution_count"] = 1
        nb["cells"][code_idx] = cell

    # Ensure code has outputs even if newly inserted
    nb["cells"][code_idx]["outputs"] = outputs
    nb["cells"][code_idx]["execution_count"] = 1

    # Interpret markdown: next markdown after code, or insert
    interpret_idx = code_idx + 1
    if interpret_idx < len(nb["cells"]) and nb["cells"][interpret_idx]["cell_type"] == "markdown":
        src = "".join(nb["cells"][interpret_idx].get("source", []))
        if "4.8" in src or src.lstrip().startswith("## 4.8") or src.lstrip().startswith("## 5."):
            nb["cells"].insert(interpret_idx, md_cell(INTERPRET_MD))
        else:
            nb["cells"][interpret_idx] = md_cell(INTERPRET_MD)
    else:
        nb["cells"].insert(interpret_idx, md_cell(INTERPRET_MD))

    NB.write_text(json.dumps(nb, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Updated {NB}")
    for r in reports:
        print(r["video_name"], "->", r["veredito"], "heur=", r.get("veredito_heuristico"), "overlays", len(r.get("overlay_paths") or []))


if __name__ == "__main__":
    main()
