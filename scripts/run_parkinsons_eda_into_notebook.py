"""Gera figuras da EDA Parkinson e grava stdout na célula de load do Relatorio."""

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

from src.audio.parkinsons_analysis import analyze_parkinsons_dir

NB = ROOT / "notebooks" / "Relatorio.ipynb"
FIG_DIR = ROOT / "data" / "processed" / "audio" / "figures"


def fig_to_png_b64() -> str:
    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close()
    return base64.b64encode(buf.getvalue()).decode("ascii")


def png_output(b64: str) -> dict:
    return {
        "output_type": "display_data",
        "data": {"image/png": b64, "text/plain": ["<Figure>"]},
        "metadata": {},
    }


def stream_output(text: str) -> dict:
    return {
        "output_type": "stream",
        "name": "stdout",
        "text": text.splitlines(keepends=True) if text.endswith("\n") else (text + "\n").splitlines(keepends=True),
    }


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")

    result = analyze_parkinsons_dir(ROOT / "data" / "raw" / "parkinsons")
    df = result["df"]
    tele = result["tele"]

    # --- inventário ---
    inv = StringIO()
    print(f"Pasta: {result['root']}", file=inv)
    print(f"Detection: {result['detection_path']}", file=inv)
    print(f"Telemonitoring: {result['telemonitoring_path']}", file=inv)
    print("-" * 60, file=inv)
    print("Resumo:", result["summary"], file=inv)
    if result["tele_summary"]:
        print("Telemonitoring:", result["tele_summary"], file=inv)
    inv_text = inv.getvalue()
    print(inv_text)

    # Plot 1
    fig, ax = plt.subplots(figsize=(6, 4))
    order = ["saudavel", "PD"]
    counts = df["label"].value_counts().reindex(order)
    counts.plot(kind="bar", color=["#457b9d", "#e63946"], ax=ax)
    ax.set_title("Contagem de gravações por classe (status)")
    ax.set_xlabel("Classe")
    ax.set_ylabel("N gravações")
    plt.xticks(rotation=0)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "01_status_counts.png", dpi=120)
    b64_1 = fig_to_png_b64()
    out1_txt = str(counts.to_dict()) + "\n"

    # Plot 2
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    feats = ["MDVP:Jitter(%)", "MDVP:Shimmer", "HNR", "PPE"]
    for ax, feat in zip(axes.ravel(), feats):
        sns.boxplot(data=df, x="label", y=feat, order=["saudavel", "PD"], ax=ax)
        ax.set_title(feat)
    plt.suptitle("Features vocais por classe", y=1.02)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "02_boxplots_features.png", dpi=120)
    b64_2 = fig_to_png_b64()

    # Plot 3
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(
        data=df, x="PPE", hue="label", hue_order=["saudavel", "PD"], kde=True, element="step", ax=ax
    )
    ax.set_title("Distribuição de PPE por classe")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "03_ppe_hist.png", dpi=120)
    b64_3 = fig_to_png_b64()

    # Plot 4
    corr_cols = result["corr_features"]
    corr = df[corr_cols].corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax)
    ax.set_title("Correlação — features vocais + status")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "04_corr_heatmap.png", dpi=120)
    b64_4 = fig_to_png_b64()

    # Plot 5
    b64_5 = None
    out5_txt = "Telemonitoring indisponível.\n"
    if tele is not None and "total_UPDRS" in tele.columns:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.boxplot(data=tele, y="total_UPDRS", color="#a8dadc", ax=ax)
        ax.set_title("Telemonitoring — distribuição total_UPDRS")
        plt.tight_layout()
        fig.savefig(FIG_DIR / "05_updrs_boxplot.png", dpi=120)
        b64_5 = fig_to_png_b64()
        out5_txt = tele["total_UPDRS"].describe().round(2).to_string() + "\n"

    top_txt = StringIO()
    print("Top sujeitos por mediana do voice_risk_score:", file=top_txt)
    print(result["top_subjects"].to_string(index=False), file=top_txt)
    print("-" * 60, file=top_txt)
    print(result["aviso"], file=top_txt)
    top_text = top_txt.getvalue()
    print(top_text)

    # Inject into notebook code cells by marker
    nb = json.loads(NB.read_text(encoding="utf-8"))

    def set_outputs(predicate, outputs, exec_count=1):
        for cell in nb["cells"]:
            if cell.get("cell_type") != "code":
                continue
            src = "".join(cell.get("source", []))
            if predicate(src):
                cell["outputs"] = outputs
                cell["execution_count"] = exec_count
                return True
        return False

    head_html = (
        df[
            [
                "name",
                "subject",
                "label",
                "status",
                "MDVP:Jitter(%)",
                "MDVP:Shimmer",
                "HNR",
                "PPE",
                "voice_risk_score",
            ]
        ]
        .head(10)
        .to_html(index=False)
    )

    set_outputs(
        lambda s: "analyze_parkinsons_dir" in s and "result = analyze_parkinsons_dir" in s,
        [
            stream_output(inv_text),
            {
                "output_type": "display_data",
                "data": {"text/html": [head_html], "text/plain": ["<DataFrame head>"]},
                "metadata": {},
            },
        ],
    )
    set_outputs(
        lambda s: 'value_counts().reindex(order)' in s,
        [png_output(b64_1), stream_output(out1_txt)],
    )
    set_outputs(
        lambda s: "Features vocais por classe" in s,
        [png_output(b64_2)],
    )
    set_outputs(
        lambda s: 'x="PPE"' in s and "histplot" in s,
        [png_output(b64_3)],
    )
    set_outputs(
        lambda s: "corr_features" in s and "heatmap" in s,
        [png_output(b64_4)],
    )
    outs5 = [stream_output(out5_txt)]
    if b64_5:
        outs5 = [png_output(b64_5), stream_output(out5_txt)]
    set_outputs(lambda s: "total_UPDRS" in s and "boxplot" in s, outs5)

    top_html = result["top_subjects"].to_html(index=False)
    set_outputs(
        lambda s: "top_subjects" in s and "aviso" in s,
        [
            stream_output(top_text),
            {
                "output_type": "display_data",
                "data": {"text/html": [top_html], "text/plain": ["<top subjects>"]},
                "metadata": {},
            },
        ],
    )

    # patient path
    for cell in nb["cells"]:
        src = "".join(cell.get("source", []))
        if "js001_checkin_tarde.wav" in src:
            cell["source"] = [
                src.replace(
                    "data/raw/audio/js001_checkin_tarde.wav",
                    "data/raw/parkinsons/parkinsons.data",
                )
            ]

    NB.write_text(json.dumps(nb, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Figuras em {FIG_DIR}")
    print(f"Outputs gravados em {NB}")


if __name__ == "__main__":
    main()
