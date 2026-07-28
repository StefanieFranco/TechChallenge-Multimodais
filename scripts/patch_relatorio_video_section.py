"""Insere seção de análise de vídeo por assimetria no Relatorio.ipynb."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks" / "Relatorio.ipynb"
DOCS = ROOT / "docs" / "relatorio_tecnico.md"


def md(text: str) -> dict:
    lines = text.strip("\n").split("\n")
    source = [ln + "\n" for ln in lines[:-1]] + ([lines[-1] + "\n"] if lines else [])
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def code(text: str) -> dict:
    lines = text.strip("\n").split("\n")
    source = [ln + "\n" for ln in lines[:-1]] + ([lines[-1] + "\n"] if lines else [])
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source,
    }


VIDEO_MD = """## 4.7 Análise de movimento (vídeo — fisioterapia)

Clipes próprios em `data/raw/videos/` simulam a sessão de fisioterapia do paciente fictício **J.S. (pós-AVC)**.

**Pipeline:** MediaPipe Pose Landmarker (amostragem ~3 fps) → ângulos L/R de ombro, quadril e joelho + inclinação de tronco → score de assimetria ∈ [0, 1] → veredito.

| Veredito | Critério (educacional) |
|---|---|
| **CORRETO** | score < 0,25 |
| **ATENCAO** | 0,25 ≤ score < 0,50 |
| **INCORRETO** | score ≥ 0,50 |

YOLOv8 permanece como extensão futura (pessoa/objetos); neste MVP a pose MediaPipe basta para a heurística de assimetria.

> **Aviso:** heurística acadêmica — **não** é diagnóstico clínico.
"""

VIDEO_CODE = """from pathlib import Path
import sys

import pandas as pd
from IPython.display import display

ROOT = Path("..").resolve()
if not (ROOT / "src").exists():
    ROOT = Path(".").resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.video.anomaly_report import analyze_videos_dir

VIDEOS = ROOT / "data" / "raw" / "videos"
print(f"Pasta de vídeos: {VIDEOS}")
print(f"Arquivos mp4: {sorted(p.name for p in VIDEOS.glob('*.mp4'))}")
print("-" * 60)

reports = analyze_videos_dir(VIDEOS, sample_fps=3.0)

rows = []
for r in reports:
    achados = "; ".join(r["achados"])
    print(f"Vídeo   : {r['video_name']}")
    print(f"Veredito: {r['veredito']}")
    print(f"Score   : {r['score']:.4f}  (detecção pose={r['pose_meta']['detection_rate']:.1%})")
    print(f"Achados : {achados}")
    print(f"Métricas: ombro={r['metrics']['shoulder_asym_deg']}, "
          f"quadril={r['metrics']['hip_asym_deg']}, "
          f"joelho={r['metrics']['knee_asym_deg']}, "
          f"tronco={r['metrics']['trunk_lean_deg']}")
    print("-" * 60)
    rows.append({
        "video": r["video_name"],
        "veredito": r["veredito"],
        "score": r["score"],
        "deteccao_pose": round(r["pose_meta"]["detection_rate"], 3),
        "ombro_deg": r["metrics"]["shoulder_asym_deg"],
        "quadril_deg": r["metrics"]["hip_asym_deg"],
        "joelho_deg": r["metrics"]["knee_asym_deg"],
        "tronco_deg": r["metrics"]["trunk_lean_deg"],
        "achados": achados,
    })

df_video = pd.DataFrame(rows)
display(df_video)
print(r["aviso"])
"""

VIDEO_INTERPRET = """**Leitura no cenário multimodal:** o score de vídeo alimenta a fusão de risco (`src/fusion/risk_fusion.py`) junto com áudio e vitais. Vereditos `ATENCAO`/`INCORRETO` tipicamente refletem assimetria articular ou compensação de tronco — proxy educacional de padrão motor anômalo na reabilitação pós-AVC.
"""


def patch_section3_status(text: str) -> str:
    replacements = [
        (
            "| Análise de vídeo (fisio/cirurgia) | MediaPipe Pose + YOLOv8 | Planejado (`src/video`) |",
            "| Análise de vídeo (fisio/cirurgia) | MediaPipe Pose + assimetria L/R | Executado (`src/video`) |",
        ),
        (
            "| OpenPose / postura | **MediaPipe Pose** (alternativa ao OpenPose) | Planejado |",
            "| OpenPose / postura | **MediaPipe Pose** (alternativa ao OpenPose) | Executado (assimetria) |",
        ),
        (
            "| Relatório de desvios no procedimento | `anomaly_report.py` + LLM | Planejado |",
            "| Relatório de desvios no procedimento | `anomaly_report.py` (heurística) | Executado (MVP) |",
        ),
        (
            "| Vídeo | Webcam própria (exercício simulado) | UCF101 / NTU RGB+D adaptados; documentar simulação |",
            "| Vídeo | Clipe próprio em `data/raw/videos/` + assimetria MediaPipe | UCF101 / NTU RGB+D; YOLOv8 opcional |",
        ),
    ]
    for old, new in replacements:
        text = text.replace(old, new)
    return text


def patch_checklist(text: str) -> str:
    return text.replace(
        '{"id": "E3", "nome": "Pose MediaPipe em vídeo próprio", "feito": False}',
        '{"id": "E3", "nome": "Pose MediaPipe em vídeo próprio", "feito": True}',
    )


def main() -> None:
    nb = json.loads(NB.read_text(encoding="utf-8"))

    # Atualiza status na seção 3 / datasets / checklist
    for cell in nb["cells"]:
        src = "".join(cell.get("source", []))
        if cell["cell_type"] == "markdown" and "Requisitos do enunciado" in src:
            cell["source"] = [patch_section3_status(src)]
        if cell["cell_type"] == "markdown" and "Estratégia do MVP" in src:
            cell["source"] = [patch_section3_status(src)]
        if cell["cell_type"] == "code" and "Pose MediaPipe em vídeo próprio" in src:
            cell["source"] = [patch_checklist(src)]

    # Evita inserir duas vezes
    already = any(
        "4.7 Análise de movimento" in "".join(c.get("source", [])) for c in nb["cells"]
    )
    if not already:
        insert_at = None
        for i, cell in enumerate(nb["cells"]):
            src = "".join(cell.get("source", []))
            if src.lstrip().startswith("## 5. Stack e modelos"):
                insert_at = i
                break
        if insert_at is None:
            raise RuntimeError("Não encontrei a seção 5 para inserir 4.7.")
        new_cells = [md(VIDEO_MD), code(VIDEO_CODE), md(VIDEO_INTERPRET)]
        nb["cells"][insert_at:insert_at] = new_cells

    NB.write_text(json.dumps(nb, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Updated {NB}")

    # docs
    docs = DOCS.read_text(encoding="utf-8")
    old_video_row = (
        "| Vídeo | Webcam própria (exercício simulado) | UCF101 / NTU adaptados | `data/raw/video/` |"
    )
    new_video_row = (
        "| Vídeo | Clipe próprio + MediaPipe assimetria L/R (pós-AVC) | "
        "UCF101 / NTU; YOLOv8 opcional | `data/raw/videos/` |"
    )
    if old_video_row in docs:
        docs = docs.replace(old_video_row, new_video_row)
    note = (
        "\n### Vídeo — fisioterapia (assimetria)\n\n"
        "Clipes em `data/raw/videos/`. Pose via MediaPipe Pose Landmarker; "
        "heurística educacional de assimetria L/R (ombro/quadril/joelho/tronco) "
        "em `src/video/` → veredito CORRETO / ATENCAO / INCORRETO "
        "(ver `notebooks/Relatorio.ipynb` §4.7).\n"
    )
    if "### Vídeo — fisioterapia" not in docs:
        # inserir após bloco ECG se existir, senão no fim da seção 4
        anchor = "Limpeza + união → `data/processed/vitals/arrhythmia_train.parquet`"
        if anchor in docs:
            # after the ECG paragraph block — find next blank after justificativa ECG
            idx = docs.find("Justificativa: datasets clínicos")
            if idx != -1:
                end = docs.find("\n## 5.", idx)
                if end == -1:
                    end = docs.find("\n## ", idx + 1)
                if end != -1:
                    docs = docs[:end] + note + docs[end:]
                else:
                    docs += note
            else:
                docs += note
        else:
            docs += note
    DOCS.write_text(docs, encoding="utf-8")
    print(f"Updated {DOCS}")


if __name__ == "__main__":
    main()
