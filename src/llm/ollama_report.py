"""Gera resumo/alerta clínico via Ollama."""

from typing import Any


def generate_report(
    fusion_result: dict[str, Any],
    context: dict[str, Any] | None = None,
    model: str = "llama3.2",
    host: str = "http://localhost:11434",
) -> str:
    """Chama o Ollama para produzir resumo e alerta textual.

    Args:
        fusion_result: Resultado da fusão de risco.
        context: Contexto adicional (transcrição, achados, etc.).
        model: Nome do modelo no Ollama.
        host: Endpoint da API Ollama.

    Returns:
        Texto do relatório/alerta.
    """
    raise NotImplementedError("Implementar geração de relatório com Ollama.")
