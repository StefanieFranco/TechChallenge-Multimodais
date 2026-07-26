"""Relatório de anomalias a partir da estimativa de pose."""

from typing import Any


def build_anomaly_report(pose_data: dict[str, Any]) -> dict[str, Any]:
    """Gera score e descrição de anomalias motoras.

    Args:
        pose_data: Saída de `pose_estimation.estimate_pose`.

    Returns:
        Relatório com score de risco e achados.
    """
    raise NotImplementedError("Implementar relatório de anomalias de vídeo.")
