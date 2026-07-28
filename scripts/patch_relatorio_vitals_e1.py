"""Insere §4.9 (treino Isolation Forest visível) no Relatorio e atualiza docs/status."""

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


INTRO = """## 4.9 Detecção de anomalias em vitais (E1 — Isolation Forest)

Experimento **E1**: série sintética HR/SpO₂/SBP/DBP (noite do paciente J.S.) com anomalias injetadas
e treino de **Isolation Forest** (`sklearn`) via `src/vitals/anomaly_detection.py`.

Experimento detalhado: [`01_vitals_sinteticos.ipynb`](01_vitals_sinteticos.ipynb).  
Artefatos: `data/raw/vitals/synthetic/js001_noite.csv`, `data/processed/vitals/isolation_forest_vitals.joblib` + `_meta.json`.

> **Aviso:** sintético educacional — não é diagnóstico clínico.
"""

TRAIN_CODE = """from pathlib import Path
import sys
import json

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from IPython.display import display

ROOT = Path("..").resolve()
if not (ROOT / "src").exists():
    ROOT = Path(".").resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.vitals.synthetic_vitals import load_or_create_synthetic, summarize_synthetic
from src.vitals.anomaly_detection import (
    run_training_pipeline,
    load_train_meta,
    default_meta_path,
    default_model_path,
    load_model,
    detect_anomalies,
)

sns.set_theme(style="whitegrid", context="notebook")

CSV = ROOT / "data" / "raw" / "vitals" / "synthetic" / "js001_noite.csv"
MODEL = default_model_path(ROOT)
META = default_meta_path(ROOT)

df, csv_path = load_or_create_synthetic(path=CSV, force=not CSV.exists())
print("CSV:", csv_path)
print("Resumo série:", summarize_synthetic(df))

# Treina (ou retreina) e persiste — deixa params/métricas visíveis no Relatorio
result = run_training_pipeline(df, contamination=0.05, n_estimators=200, random_state=42)
meta = result["meta"]

print("-" * 60)
print("HIPERPARÂMETROS / TREINO")
for k in ("model_name", "n_estimators", "contamination", "random_state", "feature_cols", "n_samples"):
    print(f"  {k}: {meta.get(k)}")
print("-" * 60)
print("RESULTADOS")
print(f"  n_anomalies_gt  : {meta.get('n_anomalies_gt')}")
print(f"  n_anomalies_pred: {meta.get('n_anomalies_pred')}")
print(f"  risk_score      : {meta.get('risk_score'):.4f}")
print(f"  metrics         : {meta.get('metrics')}")
print(f"  model_path      : {meta.get('model_path')}")
print(f"  meta_path       : {result['meta_path']}")
print("-" * 60)
display(pd.DataFrame([
    {"campo": "n_estimators", "valor": meta["n_estimators"]},
    {"campo": "contamination", "valor": meta["contamination"]},
    {"campo": "random_state", "valor": meta["random_state"]},
    {"campo": "feature_cols", "valor": ", ".join(meta["feature_cols"])},
    {"campo": "n_samples", "valor": meta["n_samples"]},
    {"campo": "n_anomalies_gt", "valor": meta["n_anomalies_gt"]},
    {"campo": "n_anomalies_pred", "valor": meta["n_anomalies_pred"]},
    {"campo": "risk_score", "valor": round(meta["risk_score"], 4)},
    {"campo": "precision", "valor": meta["metrics"]["precision"] if meta.get("metrics") else None},
    {"campo": "recall", "valor": meta["metrics"]["recall"] if meta.get("metrics") else None},
    {"campo": "f1", "valor": meta["metrics"]["f1"] if meta.get("metrics") else None},
]))
print(meta.get("aviso", ""))
"""

MD_TRAIN = """**Treino visível:** a tabela acima resume hiperparâmetros e métricas contra o ground truth injetado (`is_anomaly`). O `risk_score` ∈ [0, 1] é a média dos top-5% scores normalizados — candidato a `vitals_score` na fusão multimodal.
"""

PLOT_CODE = """points = result["points"]
step = max(1, len(points) // 1800)
vis = points.iloc[::step]

fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
for ax, col in zip(axes, ["HR", "SpO2"]):
    ax.plot(vis["timestamp"], vis[col], color="#457b9d", lw=0.7)
    pred = vis[vis["anomaly_pred"] == 1]
    ax.scatter(pred["timestamp"], pred[col], c="#e63946", s=7, label="pred", zorder=3)
    ax.set_ylabel(col)
    ax.legend(loc="upper right", fontsize=8)
axes[-1].set_xlabel("timestamp")
plt.suptitle("E1 — HR/SpO2 com anomalias preditas (Isolation Forest)")
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(7, 3.5))
sns.histplot(points["anomaly_score_norm"], bins=40, color="#a8dadc", ax=ax)
ax.axvline(points.loc[points["anomaly_pred"] == 1, "anomaly_score_norm"].mean(),
           color="#e63946", ls="--", label="média score (pred=1)")
ax.set_title("Histograma dos scores de anomalia (normalizados)")
ax.legend()
plt.tight_layout()
plt.show()
"""

MD_END = """**Leitura no cenário J.S.:** episódios de SpO₂ baixa / taquicardia injetados na noite sintética devem elevar o score de vitais. Na etapa de fusão, este `risk_score` combina com assimetria de vídeo e `voice_risk_score` de áudio.
"""


def patch_status(text: str) -> str:
    reps = [
        (
            "| Anomalias em vitais | Isolation Forest / PyOD | Planejado (`src/vitals`) |",
            "| Anomalias em vitais | Isolation Forest (sklearn) | Executado (v1 sintético) |",
        ),
    ]
    for old, new in reps:
        text = text.replace(old, new)
    return text


def patch_checklist(text: str) -> str:
    return text.replace(
        '{"id": "E1", "nome": "Vitais sintéticos + Isolation Forest", "feito": False}',
        '{"id": "E1", "nome": "Vitais sintéticos + Isolation Forest", "feito": True}',
    )


def main() -> None:
    nb = json.loads(NB.read_text(encoding="utf-8"))

    for cell in nb["cells"]:
        src = "".join(cell.get("source", []))
        if cell["cell_type"] == "markdown" and "Requisitos do enunciado" in src:
            cell["source"] = [patch_status(src)]
        if cell["cell_type"] == "code" and "Vitais sintéticos + Isolation Forest" in src:
            cell["source"] = [patch_checklist(src)]

    already = any(
        "4.9 Detecção de anomalias em vitais" in "".join(c.get("source", []))
        for c in nb["cells"]
    )
    if not already:
        insert_at = None
        for i, cell in enumerate(nb["cells"]):
            src = "".join(cell.get("source", []))
            if src.lstrip().startswith("## 5. Stack e modelos"):
                insert_at = i
                break
        if insert_at is None:
            raise RuntimeError("Seção 5 não encontrada no Relatorio")
        new_cells = [
            md(INTRO),
            code(TRAIN_CODE),
            md(MD_TRAIN),
            code(PLOT_CODE),
            md(MD_END),
        ]
        nb["cells"][insert_at:insert_at] = new_cells

    # próximo passo sugerido — atualizar se existir
    for cell in nb["cells"]:
        src = "".join(cell.get("source", []))
        if "implementar `01_vitals_sinteticos.ipynb`" in src:
            cell["source"] = [
                src.replace(
                    "**Próximo passo sugerido:** implementar `01_vitals_sinteticos.ipynb` e a primeira versão de `src/vitals/anomaly_detection.py`.",
                    "**E1 concluído:** ver §4.9 e `01_vitals_sinteticos.ipynb`. Próximo: fusão dos scores / áudio Whisper ou prescrição.",
                )
            ]

    NB.write_text(json.dumps(nb, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Updated {NB}")

    docs = DOCS.read_text(encoding="utf-8")
    note = (
        "\n### E1 — Vitais sintéticos + Isolation Forest\n\n"
        "Série sintética em `data/raw/vitals/synthetic/js001_noite.csv`. "
        "Detector: `src/vitals/anomaly_detection.py` → "
        "`data/processed/vitals/isolation_forest_vitals.joblib` + `_meta.json`. "
        "Acompanhar treino/métricas em `notebooks/Relatorio.ipynb` §4.9 "
        "(experimento detalhado: `01_vitals_sinteticos.ipynb`).\n"
    )
    if "### E1 — Vitais sintéticos" not in docs:
        anchor = "### Áudio — features vocais Parkinson"
        if anchor in docs:
            idx = docs.find(anchor)
            next_h = docs.find("\n## ", idx + 1)
            if next_h != -1:
                docs = docs[:next_h] + note + docs[next_h:]
            else:
                docs += note
        else:
            docs += note
    # update vitals line in architecture table if still generic
    docs = docs.replace(
        "| Vitais / texto | `src/vitals` | PyOD / Isolation Forest + checagem de prescrição |",
        "| Vitais / texto | `src/vitals` | Isolation Forest v1 (sintético); prescrição planejada |",
    )
    DOCS.write_text(docs, encoding="utf-8")
    print(f"Updated {DOCS}")


if __name__ == "__main__":
    main()
