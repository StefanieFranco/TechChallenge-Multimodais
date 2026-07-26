"""Detecção de anomalias em sinais vitais (PyOD / Isolation Forest)."""

from typing import Any

import pandas as pd


def detect_anomalies(vitals_df: pd.DataFrame) -> dict[str, Any]:
    """Detecta anomalias em séries de sinais vitais.

    Args:
        vitals_df: DataFrame com colunas de vitais (ex.: HR, SpO2, BP).

    Returns:
        Score de anomalia, flags e pontos suspeitos.
    """
    raise NotImplementedError("Implementar detecção com PyOD / Isolation Forest.")
