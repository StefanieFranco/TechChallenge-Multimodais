"""Análise tabular UCI Parkinsons (features vocais) — proxy educacional de áudio."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Features usadas no score educacional (maior = pior, exceto HNR).
RISK_FEATURES_HIGHER_WORSE = (
    "PPE",
    "spread1",
    "MDVP:Jitter(%)",
    "MDVP:Shimmer",
)
RISK_FEATURES_HIGHER_BETTER = ("HNR",)

AVISO = (
    "Proxy educacional com features vocais UCI Parkinson (Oxford). "
    "Nao e diagnostico clinico. Whisper/STT cobre clips .wav do check-in (ver §4.10)."
)


def _project_root() -> Path:
    start = Path.cwd().resolve()
    for candidate in [start, *start.parents]:
        if (candidate / "requirements.txt").exists() and (candidate / "src").exists():
            return candidate
    return start


def default_parkinsons_dir(root: Path | None = None) -> Path:
    root = root or _project_root()
    return root / "data" / "raw" / "parkinsons"


def load_parkinsons(path: str | Path | None = None) -> pd.DataFrame:
    """Carrega o Oxford Parkinson's Detection Dataset (`parkinsons.data`)."""
    if path is None:
        path = default_parkinsons_dir() / "parkinsons.data"
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset Parkinson nao encontrado: {path}")
    df = pd.read_csv(path)
    if "status" not in df.columns:
        raise ValueError("Coluna 'status' ausente no arquivo Parkinson.")
    df = df.copy()
    df["status"] = df["status"].astype(int)
    df["label"] = df["status"].map({0: "saudavel", 1: "PD"})
    # Sujeito aproximado a partir do nome (ex.: phon_R01_S01_1 → S01)
    df["subject"] = df["name"].astype(str).str.extract(r"(S\d+)", expand=False)
    return df


def load_telemonitoring(path: str | Path | None = None) -> pd.DataFrame:
    """Carrega Parkinsons Telemonitoring (`parkinsons_updrs.data`)."""
    if path is None:
        path = default_parkinsons_dir() / "telemonitoring" / "parkinsons_updrs.data"
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Telemonitoring nao encontrado: {path}")
    df = pd.read_csv(path)
    return df


def summarize_dataset(df: pd.DataFrame) -> dict[str, Any]:
    """Contagens por status, n_sujeitos e medias das features-chave."""
    key_cols = [
        c
        for c in (
            "MDVP:Jitter(%)",
            "MDVP:Shimmer",
            "HNR",
            "PPE",
            "spread1",
        )
        if c in df.columns
    ]
    by_status = df["status"].value_counts().sort_index().to_dict()
    means = (
        df.groupby("label")[key_cols].mean().round(4).to_dict()
        if key_cols
        else {}
    )
    n_subjects = int(df["subject"].nunique()) if "subject" in df.columns else None
    return {
        "n_rows": int(len(df)),
        "n_subjects": n_subjects,
        "status_counts": {int(k): int(v) for k, v in by_status.items()},
        "label_counts": df["label"].value_counts().to_dict() if "label" in df else {},
        "feature_means_by_label": means,
        "key_features": key_cols,
    }


def _minmax_series(s: pd.Series) -> pd.Series:
    lo, hi = float(s.min()), float(s.max())
    if hi <= lo:
        return pd.Series(np.zeros(len(s)), index=s.index, dtype=float)
    return (s.astype(float) - lo) / (hi - lo)


def score_voice_risk(
    row_or_df: pd.Series | pd.DataFrame,
    *,
    by_subject: bool = False,
) -> pd.Series | float:
    """Score de risco vocal ∈ [0, 1] (maior = mais alteracao).

    Heuristica educacional: media dos min-max de PPE, spread1, jitter, shimmer
    e (1 - HNR normalizado). Se ``by_subject`` e DataFrame, retorna mediana por sujeito.
    """
    if isinstance(row_or_df, pd.Series):
        df = row_or_df.to_frame().T
        single = True
    else:
        df = row_or_df
        single = False

    parts: list[pd.Series] = []
    for col in RISK_FEATURES_HIGHER_WORSE:
        if col not in df.columns:
            continue
        # spread1 no dataset e tipicamente negativo; valores menos negativos = pior
        s = df[col].astype(float)
        if col == "spread1":
            s = -s  # inverter sinal para "maior = pior"
        parts.append(_minmax_series(s))
    for col in RISK_FEATURES_HIGHER_BETTER:
        if col not in df.columns:
            continue
        parts.append(1.0 - _minmax_series(df[col].astype(float)))

    if not parts:
        raise ValueError("Nenhuma feature de risco disponivel no DataFrame.")

    stacked = pd.concat(parts, axis=1)
    scores = stacked.mean(axis=1).clip(0.0, 1.0)
    scores.name = "voice_risk_score"

    if by_subject and "subject" in df.columns and not single:
        out = (
            pd.DataFrame({"subject": df["subject"].values, "voice_risk_score": scores.values})
            .groupby("subject", as_index=True)["voice_risk_score"]
            .median()
            .sort_values(ascending=False)
        )
        return out

    if single:
        return float(scores.iloc[0])
    return scores


def top_risk_subjects(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Top sujeitos por mediana do score de risco, com label majoritario."""
    scores = score_voice_risk(df, by_subject=True)
    assert isinstance(scores, pd.Series)
    label_mode = (
        df.groupby("subject")["label"]
        .agg(lambda s: s.value_counts().index[0])
        if "subject" in df.columns
        else None
    )
    status_mode = (
        df.groupby("subject")["status"].agg(lambda s: int(s.mode().iloc[0]))
        if "subject" in df.columns
        else None
    )
    out = scores.rename("voice_risk_score").to_frame()
    if label_mode is not None:
        out["label"] = label_mode
    if status_mode is not None:
        out["status"] = status_mode
    out = out.reset_index().rename(columns={"index": "subject"})
    if "subject" not in out.columns and out.index.name == "subject":
        out = out.reset_index()
    return out.head(n).reset_index(drop=True)


def analyze_parkinsons_dir(root: str | Path | None = None) -> dict[str, Any]:
    """API unica: carrega detection + telemonitoring, resume e pontua."""
    root = Path(root) if root else default_parkinsons_dir()
    det_path = root / "parkinsons.data"
    tele_path = root / "telemonitoring" / "parkinsons_updrs.data"

    df = load_parkinsons(det_path)
    summary = summarize_dataset(df)
    scored = df.copy()
    scored["voice_risk_score"] = score_voice_risk(df)
    top = top_risk_subjects(df, n=10)

    tele = None
    tele_summary: dict[str, Any] | None = None
    if tele_path.exists():
        tele = load_telemonitoring(tele_path)
        tele_summary = {
            "n_rows": int(len(tele)),
            "n_subjects": int(tele["subject#"].nunique()) if "subject#" in tele.columns else None,
            "total_UPDRS_mean": float(tele["total_UPDRS"].mean())
            if "total_UPDRS" in tele.columns
            else None,
            "motor_UPDRS_mean": float(tele["motor_UPDRS"].mean())
            if "motor_UPDRS" in tele.columns
            else None,
        }

    return {
        "root": str(root.resolve()),
        "detection_path": str(det_path.resolve()),
        "telemonitoring_path": str(tele_path.resolve()) if tele_path.exists() else None,
        "df": scored,
        "summary": summary,
        "top_subjects": top,
        "tele": tele,
        "tele_summary": tele_summary,
        "aviso": AVISO,
        "corr_features": [
            c
            for c in (
                "MDVP:Jitter(%)",
                "MDVP:Shimmer",
                "HNR",
                "NHR",
                "PPE",
                "spread1",
                "RPDE",
                "DFA",
                "status",
            )
            if c in scored.columns
        ],
    }
