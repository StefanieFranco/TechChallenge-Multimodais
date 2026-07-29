"""Classificador supervisionado UCI Parkinson (RandomForest) — métricas de teste."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

from src.audio.parkinsons_analysis import default_parkinsons_dir, load_parkinsons

# Features clínicas típicas do Oxford Parkinson's Detection Dataset
DEFAULT_FEATURE_COLS = (
    "MDVP:Fo(Hz)",
    "MDVP:Fhi(Hz)",
    "MDVP:Flo(Hz)",
    "MDVP:Jitter(%)",
    "MDVP:Jitter(Abs)",
    "MDVP:RAP",
    "MDVP:PPQ",
    "Jitter:DDP",
    "MDVP:Shimmer",
    "MDVP:Shimmer(dB)",
    "Shimmer:APQ3",
    "Shimmer:APQ5",
    "MDVP:APQ",
    "Shimmer:DDA",
    "NHR",
    "HNR",
    "RPDE",
    "DFA",
    "spread1",
    "spread2",
    "D2",
    "PPE",
)


def _project_root() -> Path:
    start = Path.cwd().resolve()
    for candidate in [start, *start.parents]:
        if (candidate / "requirements.txt").exists() and (candidate / "src").exists():
            return candidate
    return start


def default_rf_model_path(root: Path | None = None) -> Path:
    root = root or _project_root()
    return root / "data" / "processed" / "audio" / "parkinson_rf.joblib"


def default_rf_meta_path(root: Path | None = None) -> Path:
    root = root or _project_root()
    return root / "data" / "processed" / "audio" / "parkinson_rf_meta.json"


def run_parkinson_classifier(
    df: pd.DataFrame | None = None,
    *,
    feature_cols: list[str] | None = None,
    test_size: float = 0.25,
    random_state: int = 42,
    n_estimators: int = 200,
    model_path: str | Path | None = None,
    meta_path: str | Path | None = None,
) -> dict[str, Any]:
    """Treina RandomForest (saudável vs PD) e retorna métricas de teste."""
    if df is None:
        df = load_parkinsons(default_parkinsons_dir() / "parkinsons.data")

    cols = [c for c in (feature_cols or DEFAULT_FEATURE_COLS) if c in df.columns]
    if not cols:
        raise ValueError("Nenhuma feature Parkinson disponível para o classificador.")
    if "status" not in df.columns:
        raise ValueError("Coluna status ausente.")

    X = df[cols].to_numpy(dtype=float)
    y = df["status"].astype(int).to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        class_weight="balanced",
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    report = classification_report(
        y_test,
        y_pred,
        target_names=["saudavel", "PD"],
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_test, y_pred).tolist()
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision_pd": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall_pd": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1_pd": float(f1_score(y_test, y_pred, zero_division=0)),
    }

    model_path = Path(model_path) if model_path else default_rf_model_path()
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": clf, "feature_cols": cols}, model_path)

    import json

    meta = {
        "model_name": "sklearn.RandomForestClassifier",
        "n_estimators": n_estimators,
        "feature_cols": cols,
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "metrics": metrics,
        "classification_report": report,
        "confusion_matrix": cm,
        "model_path": str(model_path),
        "aviso": (
            "Classificador educacional UCI Parkinson. "
            "Não é diagnóstico clínico de doença de Parkinson."
        ),
    }
    meta_path = Path(meta_path) if meta_path else default_rf_meta_path()
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    importances = (
        pd.Series(clf.feature_importances_, index=cols)
        .sort_values(ascending=False)
        .head(10)
    )

    return {
        "model": clf,
        "feature_cols": cols,
        "metrics": metrics,
        "classification_report": report,
        "confusion_matrix": cm,
        "y_test": y_test,
        "y_pred": y_pred,
        "importances": importances,
        "model_path": model_path,
        "meta_path": meta_path,
        "meta": meta,
    }
