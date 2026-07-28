"""Detecção de anomalias em sinais vitais (Isolation Forest / sklearn)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
)

DEFAULT_FEATURE_COLS = ("HR", "SpO2", "SBP", "DBP")
DEFAULT_N_ESTIMATORS = 200
DEFAULT_CONTAMINATION = 0.05
DEFAULT_RANDOM_STATE = 42


def _project_root() -> Path:
    start = Path.cwd().resolve()
    for candidate in [start, *start.parents]:
        if (candidate / "requirements.txt").exists() and (candidate / "src").exists():
            return candidate
    return start


def default_model_path(root: Path | None = None) -> Path:
    root = root or _project_root()
    return root / "data" / "processed" / "vitals" / "isolation_forest_vitals.joblib"


def default_meta_path(root: Path | None = None) -> Path:
    root = root or _project_root()
    return root / "data" / "processed" / "vitals" / "isolation_forest_vitals_meta.json"


def _resolve_features(df: pd.DataFrame, feature_cols: list[str] | None) -> list[str]:
    cols = list(feature_cols) if feature_cols else list(DEFAULT_FEATURE_COLS)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Colunas ausentes no DataFrame de vitais: {missing}")
    return cols


def _anomaly_scores(model: IsolationForest, X: np.ndarray) -> np.ndarray:
    """Maior score = mais anômalo (inverte decision_function do sklearn)."""
    return -model.decision_function(X)


def _normalize_01(scores: np.ndarray) -> np.ndarray:
    lo, hi = float(np.min(scores)), float(np.max(scores))
    if hi <= lo:
        return np.zeros_like(scores, dtype=float)
    return (scores - lo) / (hi - lo)


def _risk_from_scores(scores: np.ndarray, top_frac: float = 0.05) -> float:
    """Média dos top-k% scores normalizados → risk_score ∈ [0, 1]."""
    norm = _normalize_01(scores)
    k = max(1, int(len(norm) * top_frac))
    top = np.sort(norm)[-k:]
    return float(np.mean(top))


def fit_isolation_forest(
    vitals_df: pd.DataFrame,
    *,
    feature_cols: list[str] | None = None,
    contamination: float = DEFAULT_CONTAMINATION,
    n_estimators: int = DEFAULT_N_ESTIMATORS,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> IsolationForest:
    """Treina IsolationForest nas features de vitais."""
    cols = _resolve_features(vitals_df, feature_cols)
    X = vitals_df[cols].to_numpy(dtype=float)
    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X)
    return model


def detect_anomalies(
    vitals_df: pd.DataFrame,
    *,
    feature_cols: list[str] | None = None,
    contamination: float = DEFAULT_CONTAMINATION,
    n_estimators: int = DEFAULT_N_ESTIMATORS,
    random_state: int = DEFAULT_RANDOM_STATE,
    model: IsolationForest | None = None,
    top_frac: float = 0.05,
) -> dict[str, Any]:
    """Detecta anomalias em séries de sinais vitais.

    Args:
        vitals_df: DataFrame com colunas de vitais (ex.: HR, SpO2, SBP, DBP).
        feature_cols: Subconjunto de colunas; padrão DEFAULT_FEATURE_COLS.
        contamination: Fração esperada de anomalias no Isolation Forest.
        n_estimators: Árvores do Isolation Forest (se treinar).
        random_state: Semente.
        model: Modelo já treinado; se None, treina nos dados.
        top_frac: Fração superior dos scores para o risk_score agregado.

    Returns:
        Score de anomalia, flags, risk_score e metadados de treino.
    """
    cols = _resolve_features(vitals_df, feature_cols)
    X = vitals_df[cols].to_numpy(dtype=float)
    fitted_here = model is None
    if model is None:
        model = fit_isolation_forest(
            vitals_df,
            feature_cols=cols,
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=random_state,
        )

    raw_pred = model.predict(X)  # 1 normal, -1 anomalia
    labels = (raw_pred == -1).astype(int)
    scores = _anomaly_scores(model, X)
    norm_scores = _normalize_01(scores)
    risk = _risk_from_scores(scores, top_frac=top_frac)

    points = vitals_df.copy()
    points["anomaly_score"] = scores
    points["anomaly_score_norm"] = norm_scores
    points["anomaly_pred"] = labels

    train_info: dict[str, Any] = {
        "model_name": "sklearn.IsolationForest",
        "n_estimators": int(getattr(model, "n_estimators", n_estimators)),
        "contamination": float(contamination),
        "random_state": int(random_state),
        "feature_cols": cols,
        "n_samples": int(len(vitals_df)),
        "n_anomalies_pred": int(labels.sum()),
        "fitted_on_call": fitted_here,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }

    return {
        "scores": scores,
        "labels": labels,
        "risk_score": risk,
        "n_anomalies": int(labels.sum()),
        "feature_cols": cols,
        "model_name": train_info["model_name"],
        "model": model,
        "points": points,
        "train_info": train_info,
    }


def evaluate_against_ground_truth(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
) -> dict[str, float]:
    """Precision / recall / F1 (classe positiva = anomalia)."""
    yt = np.asarray(y_true).astype(int)
    yp = np.asarray(y_pred).astype(int)
    return {
        "precision": float(precision_score(yt, yp, zero_division=0)),
        "recall": float(recall_score(yt, yp, zero_division=0)),
        "f1": float(f1_score(yt, yp, zero_division=0)),
    }


def save_model(
    model: IsolationForest,
    path: str | Path | None = None,
) -> Path:
    path = Path(path) if path else default_model_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return path


def load_model(path: str | Path | None = None) -> IsolationForest:
    path = Path(path) if path else default_model_path()
    if not path.exists():
        raise FileNotFoundError(f"Modelo não encontrado: {path}")
    return joblib.load(path)


def save_train_meta(meta: dict[str, Any], path: str | Path | None = None) -> Path:
    path = Path(path) if path else default_meta_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_train_meta(path: str | Path | None = None) -> dict[str, Any]:
    path = Path(path) if path else default_meta_path()
    if not path.exists():
        raise FileNotFoundError(f"Meta de treino não encontrada: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def run_training_pipeline(
    vitals_df: pd.DataFrame,
    *,
    contamination: float = DEFAULT_CONTAMINATION,
    n_estimators: int = DEFAULT_N_ESTIMATORS,
    random_state: int = DEFAULT_RANDOM_STATE,
    feature_cols: list[str] | None = None,
    model_path: str | Path | None = None,
    meta_path: str | Path | None = None,
    ground_truth_col: str = "is_anomaly",
) -> dict[str, Any]:
    """Treina, avalia (se houver GT), salva modelo + meta JSON."""
    result = detect_anomalies(
        vitals_df,
        feature_cols=feature_cols,
        contamination=contamination,
        n_estimators=n_estimators,
        random_state=random_state,
    )
    metrics: dict[str, float] | None = None
    n_gt = None
    if ground_truth_col in vitals_df.columns:
        metrics = evaluate_against_ground_truth(
            vitals_df[ground_truth_col], result["labels"]
        )
        n_gt = int(vitals_df[ground_truth_col].sum())

    model_path = save_model(result["model"], model_path)
    meta = {
        **result["train_info"],
        "risk_score": result["risk_score"],
        "n_anomalies_gt": n_gt,
        "metrics": metrics,
        "model_path": str(model_path),
        "aviso": (
            "Dados/treino educacionais (vitais sintéticos). "
            "Nao constitui diagnostico clinico."
        ),
    }
    meta_path = save_train_meta(meta, meta_path)
    result["meta"] = meta
    result["model_path"] = model_path
    result["meta_path"] = meta_path
    result["metrics"] = metrics
    return result
