"""Checagem de aderência / inconsistências de prescrição."""

from __future__ import annotations

from typing import Any

import pandas as pd

# Prescrição fictícia educacional do paciente JS-001 (pós-AVC / reabilitação).
DEFAULT_JS001_PRESCRIPTION: dict[str, Any] = {
    "patient_id": "JS-001",
    "medications": [
        {"name": "AAS", "dose": "100 mg", "freq": "1x/dia"},
        {"name": "Sinvastatina", "dose": "40 mg", "freq": "noite"},
        {"name": "Losartana", "dose": "50 mg", "freq": "1x/dia"},
    ],
    "targets": {
        "SpO2_min": 94.0,
        "HR_min": 50.0,
        "HR_max": 100.0,
        "SBP_max": 160.0,
        "DBP_max": 100.0,
    },
    "notes": "Oxigenoterapia se SpO2 < 94%; acionar plantão se taquicardia sustentada.",
}


def check_prescription(
    vitals_df: pd.DataFrame,
    prescription: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Avalia sinais vitais à luz da prescrição médica.

    Args:
        vitals_df: Séries de sinais vitais.
        prescription: Dados da prescrição (medicamentos, alvos, etc.).

    Returns:
        Achados, violações e score relacionado à prescrição [0,1].
    """
    rx = prescription or DEFAULT_JS001_PRESCRIPTION
    targets = dict(rx.get("targets") or {})
    violations: list[str] = []
    counts: dict[str, int] = {}

    def _count(mask: pd.Series, label: str) -> None:
        n = int(mask.sum())
        if n > 0:
            counts[label] = n
            violations.append(f"{label}: {n} amostras fora do alvo")

    if "SpO2" in vitals_df.columns and "SpO2_min" in targets:
        _count(vitals_df["SpO2"] < float(targets["SpO2_min"]), "SpO2 abaixo do mínimo")
    if "HR" in vitals_df.columns:
        if "HR_min" in targets:
            _count(vitals_df["HR"] < float(targets["HR_min"]), "FC abaixo do mínimo")
        if "HR_max" in targets:
            _count(vitals_df["HR"] > float(targets["HR_max"]), "FC acima do máximo")
    if "SBP" in vitals_df.columns and "SBP_max" in targets:
        _count(vitals_df["SBP"] > float(targets["SBP_max"]), "PAS acima do máximo")
    if "DBP" in vitals_df.columns and "DBP_max" in targets:
        _count(vitals_df["DBP"] > float(targets["DBP_max"]), "PAD acima do máximo")

    n = max(1, len(vitals_df))
    frac = sum(counts.values()) / (n * max(1, len(counts) or 1))
    # Score educacional: proporção de violações, saturando em 1.0
    score = float(min(1.0, 0.2 + 0.8 * min(1.0, sum(counts.values()) / max(1, n * 0.05))))
    if not violations:
        score = 0.0

    achados = list(violations)
    if not achados:
        achados = ["Sinais vitais dentro dos alvos da prescrição (janela analisada)."]

    return {
        "patient_id": rx.get("patient_id"),
        "medications": rx.get("medications") or [],
        "targets": targets,
        "notes": rx.get("notes"),
        "violations": violations,
        "achados": achados,
        "counts": counts,
        "n_samples": int(len(vitals_df)),
        "score": score,
        "aviso": (
            "Checagem educacional de alvos prescritos vs série de vitais. "
            "Não substitui decisão clínica."
        ),
    }
