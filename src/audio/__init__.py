"""Módulo de análise de áudio (transcrição e fala)."""

from src.audio.parkinsons_analysis import (
    analyze_parkinsons_dir,
    load_parkinsons,
    score_voice_risk,
)
from src.audio.speech_analysis import analyze_speech
from src.audio.transcription import transcribe

__all__ = [
    "analyze_parkinsons_dir",
    "load_parkinsons",
    "score_voice_risk",
    "transcribe",
    "analyze_speech",
]
