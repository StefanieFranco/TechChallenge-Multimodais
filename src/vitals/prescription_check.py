"""Checagem de aderência / inconsistências de prescrição."""

from typing import Any

import pandas as pd


def check_prescription(
    vitals_df: pd.DataFrame,
    prescription: dict[str, Any],
) -> dict[str, Any]:
    """Avalia sinais vitais à luz da prescrição médica.

    Args:
        vitals_df: Séries de sinais vitais.
        prescription: Dados da prescrição (medicamentos, alvos, etc.).

    Returns:
        Achados e score relacionado à prescrição.
    """
    raise NotImplementedError("Implementar checagem de prescrição.")
