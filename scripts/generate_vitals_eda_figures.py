"""Gera figuras EDA do conjunto arrhythmia_train.parquet."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.vitals.ecg_preprocess import (  # noqa: E402
    load_waveform_snippet,
    vitals_processed_dir,
    vitals_raw_dir,
)


def main() -> None:
    raw = vitals_raw_dir(ROOT)
    proc = vitals_processed_dir(ROOT)
    fig_dir = proc / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(proc / "arrhythmia_train.parquet")
    sns.set_theme(style="whitegrid", context="notebook")

    fig, ax = plt.subplots(figsize=(6, 4))
    df.groupby("source")["record_id"].nunique().plot(
        kind="bar", color=["#2a9d8f", "#264653"], ax=ax
    )
    ax.set_title("Records únicos por fonte (treino)")
    ax.set_xlabel("Fonte")
    ax.set_ylabel("N records")
    plt.tight_layout()
    fig.savefig(fig_dir / "01_records_por_fonte.png", dpi=120)
    plt.close()

    fig, ax = plt.subplots(figsize=(6, 4))
    df["label"].value_counts().plot(kind="bar", color=["#457b9d", "#e63946"], ax=ax)
    ax.set_title("Distribuição de classes (janelas)")
    ax.set_xlabel("Label")
    ax.set_ylabel("N janelas")
    plt.tight_layout()
    fig.savefig(fig_dir / "02_classes.png", dpi=120)
    plt.close()

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(
        data=df.dropna(subset=["hr_mean"]),
        x="hr_mean",
        hue="label",
        bins=30,
        element="step",
        ax=ax,
    )
    ax.set_title("HR média por classe")
    ax.set_xlabel("hr_mean (bpm)")
    plt.tight_layout()
    fig.savefig(fig_dir / "03_hr_hist.png", dpi=120)
    plt.close()

    normal_row = df[df["label"] == "normal"].iloc[0]
    abn = df[df["label"] == "abnormal"]
    abnormal_row = abn.iloc[0] if len(abn) else None
    fig, axes = plt.subplots(2, 1, figsize=(10, 5))
    snip_n = load_waveform_snippet(
        raw / normal_row["source"],
        normal_row["record_id"],
        start=int(normal_row["start_sample"]),
        n_samples=int(normal_row["fs"] * 5),
    )
    if snip_n:
        axes[0].plot(snip_n[0], color="#2a9d8f", lw=0.8)
        axes[0].set_title(f"NORMAL {normal_row['source']}/{normal_row['record_id']}")
    if abnormal_row is not None:
        snip_a = load_waveform_snippet(
            raw / abnormal_row["source"],
            abnormal_row["record_id"],
            start=int(abnormal_row["start_sample"]),
            n_samples=int(abnormal_row["fs"] * 5),
        )
        if snip_a:
            axes[1].plot(snip_a[0], color="#e63946", lw=0.8)
            axes[1].set_title(
                f"ABNORMAL {abnormal_row['source']}/{abnormal_row['record_id']}"
            )
    axes[1].set_xlabel("Amostra")
    plt.tight_layout()
    fig.savefig(fig_dir / "04_waveforms.png", dpi=120)
    plt.close()

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(data=df.dropna(subset=["hr_std"]), x="label", y="hr_std", ax=ax)
    ax.set_title("hr_std por label")
    plt.tight_layout()
    fig.savefig(fig_dir / "05_hr_std_box.png", dpi=120)
    plt.close()

    print("figures:", sorted(p.name for p in fig_dir.glob("*.png")))
    print("df", df.shape, df["label"].value_counts().to_dict())


if __name__ == "__main__":
    main()
