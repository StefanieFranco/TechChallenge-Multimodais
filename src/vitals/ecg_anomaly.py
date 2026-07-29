"""Isolation Forest em features ECG (arrhythmia_train.parquet)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.vitals.anomaly_detection import (
    detect_anomalies,
    evaluate_against_ground_truth,
    save_model,
    save_train_meta,
)
from src.vitals.ecg_preprocess import (
    extract_record_features,
    list_records,
    vitals_processed_dir,
    vitals_raw_dir,
)

ECG_FEATURE_COLS = ("hr_mean", "hr_std", "rr_mean", "rr_std", "sig_mean", "sig_std")
DEFAULT_ECG_MODEL = "isolation_forest_ecg.joblib"
DEFAULT_ECG_META = "isolation_forest_ecg_meta.json"


def _project_root() -> Path:
    start = Path.cwd().resolve()
    for candidate in [start, *start.parents]:
        if (candidate / "requirements.txt").exists() and (candidate / "src").exists():
            return candidate
    return start


def default_ecg_model_path(root: Path | None = None) -> Path:
    root = root or _project_root()
    return root / "data" / "processed" / "vitals" / DEFAULT_ECG_MODEL


def default_ecg_meta_path(root: Path | None = None) -> Path:
    root = root or _project_root()
    return root / "data" / "processed" / "vitals" / DEFAULT_ECG_META


def load_arrhythmia_train(path: str | Path | None = None) -> pd.DataFrame:
    path = Path(path) if path else vitals_processed_dir() / "arrhythmia_train.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Parquet ECG não encontrado: {path}")
    return pd.read_parquet(path)


def prepare_ecg_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Filtra NaN nas features ECG e cria coluna is_anomaly (abnormal=1)."""
    cols = [c for c in ECG_FEATURE_COLS if c in df.columns]
    out = df.dropna(subset=cols).copy()
    out["is_anomaly"] = (out["label"].astype(str) == "abnormal").astype(int)
    return out


def run_ecg_if_pipeline(
    df: pd.DataFrame | None = None,
    *,
    test_size: float = 0.25,
    contamination: float = 0.2,
    n_estimators: int = 200,
    random_state: int = 42,
    model_path: str | Path | None = None,
    meta_path: str | Path | None = None,
    max_high_risk_records: int = 12,
    max_windows_per_hr: int = 5,
) -> dict[str, Any]:
    """Treina IF no mitdb+nsrdb, avalia hold-out e sensibilidade high-risk."""
    if df is None:
        df = load_arrhythmia_train()
    prepared = prepare_ecg_frame(df)
    cols = [c for c in ECG_FEATURE_COLS if c in prepared.columns]

    train_df, test_df = train_test_split(
        prepared,
        test_size=test_size,
        random_state=random_state,
        stratify=prepared["is_anomaly"] if prepared["is_anomaly"].nunique() > 1 else None,
    )

    # Fit no treino; avalia no teste
    train_result = detect_anomalies(
        train_df,
        feature_cols=cols,
        contamination=contamination,
        n_estimators=n_estimators,
        random_state=random_state,
    )
    model = train_result["model"]
    test_result = detect_anomalies(
        test_df,
        feature_cols=cols,
        model=model,
        contamination=contamination,
    )
    metrics_train = evaluate_against_ground_truth(train_df["is_anomaly"], train_result["labels"])
    metrics_test = evaluate_against_ground_truth(test_df["is_anomaly"], test_result["labels"])

    model_path = save_model(model, model_path or default_ecg_model_path())
    high_risk = evaluate_high_risk_sensitivity(
        model,
        feature_cols=cols,
        max_records=max_high_risk_records,
        max_windows=max_windows_per_hr,
    )

    meta = {
        **train_result["train_info"],
        "dataset": "arrhythmia_train.parquet (mitdb+nsrdb)",
        "feature_cols": cols,
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "label_counts_all": prepared["label"].value_counts().to_dict(),
        "metrics_train": metrics_train,
        "metrics_test": metrics_test,
        "risk_score_test": test_result["risk_score"],
        "high_risk": high_risk,
        "model_path": str(model_path),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "aviso": (
            "Isolation Forest educacional em features de ECG PhysioNet. "
            "Não constitui diagnóstico clínico."
        ),
    }
    meta_path = save_train_meta(meta, meta_path or default_ecg_meta_path())

    return {
        "model": model,
        "model_path": model_path,
        "meta_path": meta_path,
        "meta": meta,
        "metrics_train": metrics_train,
        "metrics_test": metrics_test,
        "train_result": train_result,
        "test_result": test_result,
        "feature_cols": cols,
        "high_risk": high_risk,
        "prepared": prepared,
    }


def evaluate_high_risk_sensitivity(
    model,
    *,
    feature_cols: list[str],
    raw_root: Path | None = None,
    max_records: int = 12,
    max_windows: int = 5,
) -> dict[str, Any]:
    """Frações de janelas high-risk marcadas como anomalia pelo IF."""
    raw_root = raw_root or vitals_raw_dir()
    hr_dir = raw_root / "ecg-fragment-high-risk"
    if not hr_dir.exists():
        return {"available": False, "reason": f"pasta ausente: {hr_dir}"}

    record_ids = list_records(hr_dir)[:max_records]
    rows: list[dict[str, Any]] = []
    # Fragmentos high-risk são curtos (~2–10 s) — janela de 10 s falha com frequência.
    for rid in record_ids:
        feat = extract_record_features(
            hr_dir,
            rid,
            source="high-risk",
            window_sec=2.0,
            max_windows=max_windows,
            max_samples=650000,
        )
        for r in feat:
            r["label"] = "abnormal"
            r["is_anomaly"] = 1
        rows.extend(feat)

    if not rows:
        return {"available": False, "reason": "nenhuma feature extraída", "n_records": len(record_ids)}

    hdf = pd.DataFrame(rows).dropna(subset=feature_cols)
    if hdf.empty:
        return {"available": False, "reason": "features NaN", "n_records": len(record_ids)}

    result = detect_anomalies(hdf, feature_cols=feature_cols, model=model)
    rate = float(result["labels"].mean()) if len(result["labels"]) else 0.0
    return {
        "available": True,
        "n_records": len(record_ids),
        "n_windows": int(len(hdf)),
        "n_flagged": int(result["labels"].sum()),
        "sensitivity_rate": rate,
        "risk_score": float(result["risk_score"]),
    }
