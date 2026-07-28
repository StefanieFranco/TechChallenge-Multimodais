"""Executa a análise de vídeo e grava outputs na célula 4.7 do Relatorio."""

from __future__ import annotations

import json
import sys
from io import StringIO
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.video.anomaly_report import analyze_videos_dir

NB = ROOT / "notebooks" / "Relatorio.ipynb"
VIDEOS = ROOT / "data" / "raw" / "videos"


def main() -> None:
    buf = StringIO()

    def log(msg: str = "") -> None:
        print(msg)
        buf.write(msg + "\n")

    log(f"Pasta de vídeos: {VIDEOS}")
    log(f"Arquivos mp4: {sorted(p.name for p in VIDEOS.glob('*.mp4'))}")
    log("-" * 60)

    reports = analyze_videos_dir(VIDEOS, sample_fps=3.0)
    rows = []
    for r in reports:
        achados = "; ".join(r["achados"])
        log(f"Vídeo   : {r['video_name']}")
        log(f"Veredito: {r['veredito']}")
        log(
            f"Score   : {r['score']:.4f}  "
            f"(detecção pose={r['pose_meta']['detection_rate']:.1%})"
        )
        log(f"Achados : {achados}")
        log(
            "Métricas: "
            f"ombro={r['metrics']['shoulder_asym_deg']}, "
            f"quadril={r['metrics']['hip_asym_deg']}, "
            f"joelho={r['metrics']['knee_asym_deg']}, "
            f"tronco={r['metrics']['trunk_lean_deg']}"
        )
        log("-" * 60)
        rows.append(
            {
                "video": r["video_name"],
                "veredito": r["veredito"],
                "score": r["score"],
                "deteccao_pose": round(r["pose_meta"]["detection_rate"], 3),
                "ombro_deg": r["metrics"]["shoulder_asym_deg"],
                "quadril_deg": r["metrics"]["hip_asym_deg"],
                "joelho_deg": r["metrics"]["knee_asym_deg"],
                "tronco_deg": r["metrics"]["trunk_lean_deg"],
                "achados": achados,
            }
        )

    df_video = pd.DataFrame(rows)
    table_txt = df_video.to_string(index=False)
    log(table_txt)
    aviso = reports[0]["aviso"] if reports else ""
    log(aviso)

    nb = json.loads(NB.read_text(encoding="utf-8"))
    target = None
    for cell in nb["cells"]:
        src = "".join(cell.get("source", []))
        if cell["cell_type"] == "code" and "analyze_videos_dir" in src:
            target = cell
            break
    if target is None:
        raise RuntimeError("Célula 4.7 não encontrada")

    # HTML table for nicer notebook display + plain stdout stream
    html = df_video.to_html(index=False)
    target["outputs"] = [
        {
            "output_type": "stream",
            "name": "stdout",
            "text": buf.getvalue().splitlines(keepends=True),
        },
        {
            "output_type": "display_data",
            "data": {
                "text/plain": [table_txt + "\n"],
                "text/html": [html],
            },
            "metadata": {},
        },
    ]
    target["execution_count"] = 1
    NB.write_text(json.dumps(nb, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Outputs gravados em {NB}")


if __name__ == "__main__":
    main()
