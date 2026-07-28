"""Executa E1 e grava outputs nas células §4.9 do Relatorio."""

from __future__ import annotations

import base64
import json
import sys
from io import BytesIO, StringIO
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.vitals.anomaly_detection import default_meta_path, default_model_path, run_training_pipeline
from src.vitals.synthetic_vitals import load_or_create_synthetic, summarize_synthetic

NB = ROOT / "notebooks" / "Relatorio.ipynb"


def fig_b64() -> str:
    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close()
    return base64.b64encode(buf.getvalue()).decode("ascii")


def stream(text: str) -> dict:
    if not text.endswith("\n"):
        text += "\n"
    return {"output_type": "stream", "name": "stdout", "text": text.splitlines(keepends=True)}


def png(b64: str) -> dict:
    return {
        "output_type": "display_data",
        "data": {"image/png": b64, "text/plain": ["<Figure>"]},
        "metadata": {},
    }


def html_df(df: pd.DataFrame) -> dict:
    return {
        "output_type": "display_data",
        "data": {"text/html": [df.to_html(index=False)], "text/plain": [df.to_string(index=False)]},
        "metadata": {},
    }


def main() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    csv = ROOT / "data" / "raw" / "vitals" / "synthetic" / "js001_noite.csv"
    df, csv_path = load_or_create_synthetic(path=csv, force=False)
    result = run_training_pipeline(df, contamination=0.05, n_estimators=200, random_state=42)
    meta = result["meta"]
    points = result["points"]

    buf = StringIO()
    print("CSV:", csv_path, file=buf)
    print("Resumo série:", summarize_synthetic(df), file=buf)
    print("-" * 60, file=buf)
    print("HIPERPARÂMETROS / TREINO", file=buf)
    for k in ("model_name", "n_estimators", "contamination", "random_state", "feature_cols", "n_samples"):
        print(f"  {k}: {meta.get(k)}", file=buf)
    print("-" * 60, file=buf)
    print("RESULTADOS", file=buf)
    print(f"  n_anomalies_gt  : {meta.get('n_anomalies_gt')}", file=buf)
    print(f"  n_anomalies_pred: {meta.get('n_anomalies_pred')}", file=buf)
    print(f"  risk_score      : {meta.get('risk_score'):.4f}", file=buf)
    print(f"  metrics         : {meta.get('metrics')}", file=buf)
    print(f"  model_path      : {meta.get('model_path')}", file=buf)
    print(f"  meta_path       : {result['meta_path']}", file=buf)
    print("-" * 60, file=buf)
    print(meta.get("aviso", ""), file=buf)
    train_txt = buf.getvalue()
    print(train_txt)

    table = pd.DataFrame(
        [
            {"campo": "n_estimators", "valor": meta["n_estimators"]},
            {"campo": "contamination", "valor": meta["contamination"]},
            {"campo": "random_state", "valor": meta["random_state"]},
            {"campo": "feature_cols", "valor": ", ".join(meta["feature_cols"])},
            {"campo": "n_samples", "valor": meta["n_samples"]},
            {"campo": "n_anomalies_gt", "valor": meta["n_anomalies_gt"]},
            {"campo": "n_anomalies_pred", "valor": meta["n_anomalies_pred"]},
            {"campo": "risk_score", "valor": round(meta["risk_score"], 4)},
            {"campo": "precision", "valor": meta["metrics"]["precision"]},
            {"campo": "recall", "valor": meta["metrics"]["recall"]},
            {"campo": "f1", "valor": meta["metrics"]["f1"]},
        ]
    )

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
    b64_series = fig_b64()

    fig, ax = plt.subplots(figsize=(7, 3.5))
    sns.histplot(points["anomaly_score_norm"], bins=40, color="#a8dadc", ax=ax)
    mean_pred = points.loc[points["anomaly_pred"] == 1, "anomaly_score_norm"].mean()
    ax.axvline(mean_pred, color="#e63946", ls="--", label="média score (pred=1)")
    ax.set_title("Histograma dos scores de anomalia (normalizados)")
    ax.legend()
    plt.tight_layout()
    b64_hist = fig_b64()

    nb = json.loads(NB.read_text(encoding="utf-8"))

    def set_out(pred, outputs, n=1):
        for cell in nb["cells"]:
            if cell.get("cell_type") != "code":
                continue
            src = "".join(cell.get("source", []))
            if pred(src):
                cell["outputs"] = outputs
                cell["execution_count"] = n
                return True
        return False

    set_out(
        lambda s: "run_training_pipeline" in s and "HIPERPARÂMETROS" in s,
        [stream(train_txt), html_df(table)],
    )
    set_out(
        lambda s: "Histograma dos scores de anomalia" in s or (
            '["HR", "SpO2"]' in s and "anomaly_pred" in s and "E1" in s
        ),
        [png(b64_series), png(b64_hist)],
    )

    # patient path
    for cell in nb["cells"]:
        src = "".join(cell.get("source", []))
        if "js001_noite.csv" in src and "data/raw/vitals/js001_noite.csv" in src:
            cell["source"] = [
                src.replace(
                    "data/raw/vitals/js001_noite.csv",
                    "data/raw/vitals/synthetic/js001_noite.csv",
                )
            ]

    NB.write_text(json.dumps(nb, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Outputs gravados em {NB}")
    print("meta:", default_meta_path(ROOT))
    print("model:", default_model_path(ROOT))


if __name__ == "__main__":
    main()
