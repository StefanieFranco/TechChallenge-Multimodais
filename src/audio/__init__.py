"""Módulo de análise de áudio (transcrição e fala)."""

from src.audio.parkinsons_analysis import (
    analyze_parkinsons_dir,
    load_parkinsons,
    score_voice_risk,
)

__all__ = [
    "analyze_parkinsons_dir",
    "load_parkinsons",
    "score_voice_risk",
]
