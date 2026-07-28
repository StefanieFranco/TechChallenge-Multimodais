"""Cria notebooks/01_vitals_sinteticos.ipynb."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks" / "01_vitals_sinteticos.ipynb"


def md(text: str) -> dict:
    lines = text.strip("\n").split("\n")
    src = [ln + "\n" for ln in lines[:-1]] + ([lines[-1] + "\n"] if lines else [])
    return {"cell_type": "markdown", "metadata": {}, "source": src}


def code(text: str) -> dict:
    lines = text.strip("\n").split("\n")
    src = [ln + "\n" for ln in lines[:-1]] + ([lines[-1] + "\n"] if lines else [])
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": src,
    }


cells = [
    md(
        """# E1 — Vitais sintéticos + Isolation Forest

Experimento detalhado do Tech Challenge: gera HR/SpO₂/SBP/DBP sintéticos (noite do paciente J.S.),
injeta anomalias com ground truth e treina **Isolation Forest** (`src/vitals/anomaly_detection.py`).

Acompanhe o resumo executivo também em `Relatorio.ipynb` §4.9.

> **Aviso educacional:** dados sintéticos — não constituem diagnóstico clínico."""
    ),
    code(
        """from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import display

ROOT = Path("..").resolve()
if not (ROOT / "src").exists():
    ROOT = Path(".").resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

sns.set_theme(style="whitegrid", context="notebook")
print("ROOT:", ROOT)"""
    ),
    md("## 1. Gerar série sintética"),
    code(
        """from src.vitals.synthetic_vitals import (
    load_or_create_synthetic,
    summarize_synthetic,
)

df, csv_path = load_or_create_synthetic(force=True, duration_hours=7.0, seed=42)
print("CSV:", csv_path)
print("Resumo:", summarize_synthetic(df))
display(df.head())
print("is_anomaly counts:\\n", df["is_anomaly"].value_counts())"""
    ),
    md("## 2. Treinar Isolation Forest e persistir artefatos"),
    code(
        """from src.vitals.anomaly_detection import run_training_pipeline

result = run_training_pipeline(df, contamination=0.05, n_estimators=200, random_state=42)
print("Modelo :", result["model_path"])
print("Meta   :", result["meta_path"])
print("Treino :", result["train_info"])
print("risk_score:", round(result["risk_score"], 4))
print("n_anomalies pred:", result["n_anomalies"])
print("Métricas vs GT:", result["metrics"])
display(pd.DataFrame([result["meta"]]).T.rename(columns={0: "valor"}))"""
    ),
    md("## 3. Séries temporais com anomalias destacadas"),
    code(
        """points = result["points"]
# janela de visualização: primeiras 2 h + zoom opcional completo amostrado
step = max(1, len(points) // 2000)
vis = points.iloc[::step].copy()

fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
for ax, col, title in zip(
    axes,
    ["HR", "SpO2", "SBP"],
    ["Heart rate (bpm)", "SpO2 (%)", "SBP (mmHg)"],
):
    ax.plot(vis["timestamp"], vis[col], color="#457b9d", lw=0.8, label=col)
    anom = vis[vis["anomaly_pred"] == 1]
    ax.scatter(anom["timestamp"], anom[col], c="#e63946", s=8, label="pred anomalia", zorder=3)
    gt = vis[vis["is_anomaly"] == 1]
    ax.scatter(gt["timestamp"], gt[col], facecolors="none", edgecolors="#2a9d8f", s=18, label="GT", zorder=2)
    ax.set_ylabel(title)
    ax.legend(loc="upper right", fontsize=8)
axes[-1].set_xlabel("timestamp")
plt.suptitle("Vitais sintéticos — predicão Isolation Forest vs ground truth")
plt.tight_layout()
plt.show()"""
    ),
    md("## 4. Distribuição dos scores"),
    code(
        """fig, ax = plt.subplots(figsize=(7, 4))
sns.histplot(data=points, x="anomaly_score_norm", hue="is_anomaly", bins=40, ax=ax)
ax.set_title("Scores normalizados de anomalia (0=normal GT, 1=anomalia GT)")
plt.tight_layout()
plt.show()"""
    ),
    md("## 5. Comparação rápida com PyOD IForest"),
    code(
        """from pyod.models.iforest import IForest
from src.vitals.anomaly_detection import evaluate_against_ground_truth

X = df[["HR", "SpO2", "SBP", "DBP"]].to_numpy()
clf = IForest(contamination=0.05, n_estimators=200, random_state=42)
clf.fit(X)
pyod_pred = clf.labels_  # 1 = anomaly
pyod_metrics = evaluate_against_ground_truth(df["is_anomaly"], pyod_pred)
print("sklearn IF metrics:", result["metrics"])
print("PyOD IF metrics   :", pyod_metrics)"""
    ),
    md(
        """## Próximos passos

- O `risk_score` deste módulo alimentará `src/fusion/risk_fusion.py` junto com vídeo e áudio.
- Ver resumo no `Relatorio.ipynb` §4.9 para acompanhamento geral do treino."""
    ),
]

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python (.venv TechChallenge)",
            "language": "python",
            "name": "techchallenge-multimodais",
        },
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    },
    "cells": cells,
}
NB.parent.mkdir(parents=True, exist_ok=True)
NB.write_text(json.dumps(nb, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(f"Wrote {NB}")
