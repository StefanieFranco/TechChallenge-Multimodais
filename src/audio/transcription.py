"""Transcrição de áudio com Whisper."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def _load_audio_array(audio_path: Path, target_sr: int = 16000) -> np.ndarray:
    """Carrega WAV sem depender de ffmpeg (soundfile + librosa)."""
    import librosa
    import soundfile as sf

    audio, sr = sf.read(str(audio_path), always_2d=False)
    if getattr(audio, "ndim", 1) > 1:
        audio = np.mean(audio, axis=1)
    audio = np.asarray(audio, dtype=np.float32)
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    # Whisper espera float32 em [-1, 1]
    peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
    if peak > 1.0:
        audio = audio / peak
    return audio


def transcribe(audio_path: str | Path, model_size: str = "base") -> dict[str, Any]:
    """Transcreve áudio usando Whisper (openai-whisper).

    Args:
        audio_path: Caminho do arquivo de áudio.
        model_size: Tamanho do modelo Whisper (tiny, base, small, ...).

    Returns:
        Texto transcrito e segmentos com timestamps.
    """
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Áudio não encontrado: {audio_path}")

    import whisper

    model = whisper.load_model(model_size)
    try:
        audio = _load_audio_array(audio_path)
        result = model.transcribe(audio, language="pt", fp16=False)
    except Exception:
        # Fallback: caminho direto (requer ffmpeg no PATH)
        result = model.transcribe(str(audio_path), language="pt", fp16=False)

    segments = [
        {
            "id": int(seg.get("id", i)),
            "start": float(seg.get("start", 0.0)),
            "end": float(seg.get("end", 0.0)),
            "text": str(seg.get("text", "")).strip(),
        }
        for i, seg in enumerate(result.get("segments") or [])
    ]
    return {
        "text": str(result.get("text", "")).strip(),
        "segments": segments,
        "language": result.get("language") or "pt",
        "model_size": model_size,
        "audio_path": str(audio_path.resolve()),
    }
