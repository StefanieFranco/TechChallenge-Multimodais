"""Geração de séries sintéticas de sinais vitais (cenário educacional J.S.)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

DEFAULT_FEATURE_COLS = ("HR", "SpO2", "SBP", "DBP")


def _project_root() -> Path:
    start = Path.cwd().resolve()
    for candidate in [start, *start.parents]:
        if (candidate / "requirements.txt").exists() and (candidate / "src").exists():
            return candidate
    return start


def default_synthetic_csv(root: Path | None = None) -> Path:
    root = root or _project_root()
    return root / "data" / "raw" / "vitals" / "synthetic" / "js001_noite.csv"


def generate_synthetic_vitals(
    *,
    duration_hours: float = 7.0,
    fs_hz: float = 1.0,
    seed: int = 42,
    patient_id: str = "JS-001",
) -> pd.DataFrame:
    """Gera série temporal sintética HR/SpO2/SBP/DBP com anomalias injetadas.

    Coluna ``is_anomaly`` marca o ground truth educacional dos trechos alterados.
    """
    rng = np.random.default_rng(seed)
    n = int(duration_hours * 3600 * fs_hz)
    t = np.arange(n) / fs_hz
    ts = pd.to_datetime("2026-07-27 22:00:00") + pd.to_timedelta(t, unit="s")

    # Baseline noturno plausível + ruído leve
    hr = 68 + 3 * np.sin(2 * np.pi * t / 3600) + rng.normal(0, 1.5, n)
    spo2 = 97 + 0.4 * np.sin(2 * np.pi * t / 1800) + rng.normal(0, 0.35, n)
    sbp = 118 + 4 * np.sin(2 * np.pi * t / 2400) + rng.normal(0, 2.0, n)
    dbp = 72 + 2 * np.sin(2 * np.pi * t / 2400) + rng.normal(0, 1.2, n)
    is_anomaly = np.zeros(n, dtype=bool)

    def _window(start_h: float, dur_min: float) -> slice:
        start = int(start_h * 3600 * fs_hz)
        length = int(dur_min * 60 * fs_hz)
        end = min(n, start + length)
        return slice(start, end)

    # 1) Queda de SpO2 (~hora 1.5, 12 min)
    sl = _window(1.5, 12)
    spo2[sl] = spo2[sl] - rng.uniform(8, 12)
    spo2[sl] = np.clip(spo2[sl], 82, 91)
    is_anomaly[sl] = True

    # 2) Taquicardia (~hora 3.2, 10 min)
    sl = _window(3.2, 10)
    hr[sl] = hr[sl] + rng.uniform(35, 45)
    is_anomaly[sl] = True

    # 3) Pico de PA sistólica (~hora 5.0, 8 min)
    sl = _window(5.0, 8)
    sbp[sl] = sbp[sl] + rng.uniform(30, 40)
    dbp[sl] = dbp[sl] + rng.uniform(10, 15)
    is_anomaly[sl] = True

    # 4) Episódio combinado SpO2 + HR (~hora 6.2, 6 min)
    sl = _window(6.2, 6)
    spo2[sl] = np.clip(spo2[sl] - rng.uniform(7, 10), 84, 92)
    hr[sl] = hr[sl] + rng.uniform(25, 35)
    is_anomaly[sl] = True

    df = pd.DataFrame(
        {
            "timestamp": ts,
            "patient_id": patient_id,
            "HR": np.clip(hr, 40, 200).round(2),
            "SpO2": np.clip(spo2, 70, 100).round(2),
            "SBP": np.clip(sbp, 70, 220).round(2),
            "DBP": np.clip(dbp, 40, 130).round(2),
            "is_anomaly": is_anomaly.astype(int),
        }
    )
    return df


def generate_stable_vitals(
    *,
    duration_hours: float = 4.0,
    fs_hz: float = 1.0,
    seed: int = 7,
    patient_id: str = "MR-001",
) -> pd.DataFrame:
    """Série sintética estável (sem anomalias injetadas) — contraste educacional."""
    rng = np.random.default_rng(seed)
    n = int(duration_hours * 3600 * fs_hz)
    t = np.arange(n) / fs_hz
    ts = pd.to_datetime("2026-07-28 08:00:00") + pd.to_timedelta(t, unit="s")

    hr = 72 + 2.5 * np.sin(2 * np.pi * t / 3600) + rng.normal(0, 1.0, n)
    spo2 = 97.5 + 0.3 * np.sin(2 * np.pi * t / 1800) + rng.normal(0, 0.25, n)
    sbp = 120 + 3 * np.sin(2 * np.pi * t / 2400) + rng.normal(0, 1.5, n)
    dbp = 75 + 1.5 * np.sin(2 * np.pi * t / 2400) + rng.normal(0, 1.0, n)

    return pd.DataFrame(
        {
            "timestamp": ts,
            "patient_id": patient_id,
            "HR": np.clip(hr, 55, 95).round(2),
            "SpO2": np.clip(spo2, 95, 100).round(2),
            "SBP": np.clip(sbp, 100, 140).round(2),
            "DBP": np.clip(dbp, 60, 90).round(2),
            "is_anomaly": np.zeros(n, dtype=int),
        }
    )


def default_stable_csv(root: Path | None = None) -> Path:
    root = root or _project_root()
    return root / "data" / "raw" / "vitals" / "synthetic" / "mr001_estavel.csv"


def load_or_create_stable(
    path: str | Path | None = None,
    *,
    force: bool = False,
    **gen_kwargs: Any,
) -> tuple[pd.DataFrame, Path]:
    """Carrega ou gera série estável MR-001."""
    path = Path(path) if path else default_stable_csv()
    if path.exists() and not force:
        df = pd.read_csv(path, parse_dates=["timestamp"])
        return df, path
    return save_synthetic_vitals(
        df=generate_stable_vitals(**gen_kwargs),
        path=path,
    )


def save_synthetic_vitals(
    df: pd.DataFrame | None = None,
    path: str | Path | None = None,
    **gen_kwargs: Any,
) -> tuple[pd.DataFrame, Path]:
    """Gera (se preciso) e salva CSV sintético."""
    path = Path(path) if path else default_synthetic_csv()
    path.parent.mkdir(parents=True, exist_ok=True)
    if df is None:
        df = generate_synthetic_vitals(**gen_kwargs)
    df.to_csv(path, index=False)
    return df, path


def load_or_create_synthetic(
    path: str | Path | None = None,
    *,
    force: bool = False,
    **gen_kwargs: Any,
) -> tuple[pd.DataFrame, Path]:
    """Carrega CSV existente ou gera e salva."""
    path = Path(path) if path else default_synthetic_csv()
    if path.exists() and not force:
        df = pd.read_csv(path, parse_dates=["timestamp"])
        return df, path
    return save_synthetic_vitals(path=path, **gen_kwargs)


def summarize_synthetic(df: pd.DataFrame) -> dict[str, Any]:
    n = len(df)
    n_anom = int(df["is_anomaly"].sum()) if "is_anomaly" in df.columns else 0
    return {
        "n_samples": n,
        "n_anomalies_gt": n_anom,
        "anomaly_rate_gt": float(n_anom / n) if n else 0.0,
        "duration_hours": float(
            (df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]).total_seconds() / 3600
        )
        if "timestamp" in df.columns and len(df) > 1
        else None,
        "feature_means": {
            c: float(df[c].mean()) for c in DEFAULT_FEATURE_COLS if c in df.columns
        },
    }
