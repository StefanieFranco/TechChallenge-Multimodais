"""Análise de fala: features de fadiga / disartria."""

from pathlib import Path
from typing import Any


def analyze_speech(audio_path: str | Path, transcript: dict[str, Any] | None = None) -> dict[str, Any]:
    """Extrai features acústicas/linguísticas e score de risco de fala.

    Args:
        audio_path: Caminho do arquivo de áudio.
        transcript: Resultado opcional da transcrição Whisper.

    Returns:
        Features e score de risco associado à fala.
    """
    raise NotImplementedError("Implementar análise de fadiga/disartria.")
