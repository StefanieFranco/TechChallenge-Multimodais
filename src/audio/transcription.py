"""Transcrição de áudio com Whisper."""

from pathlib import Path
from typing import Any


def transcribe(audio_path: str | Path, model_size: str = "base") -> dict[str, Any]:
    """Transcreve áudio usando Whisper.

    Args:
        audio_path: Caminho do arquivo de áudio.
        model_size: Tamanho do modelo Whisper (tiny, base, small, ...).

    Returns:
        Texto transcrito e segmentos com timestamps.
    """
    raise NotImplementedError("Implementar transcrição com Whisper.")
