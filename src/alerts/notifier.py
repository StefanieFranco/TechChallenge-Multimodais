"""Envio de alertas a partir do risco fusionado / relatório LLM."""

from typing import Any


def notify(
    message: str,
    risk_level: str,
    payload: dict[str, Any] | None = None,
) -> None:
    """Dispara notificação de alerta.

    Args:
        message: Texto do alerta.
        risk_level: Nível de risco (ex.: baixo, médio, alto).
        payload: Dados estruturados opcionais.
    """
    raise NotImplementedError("Implementar canal de notificação.")
