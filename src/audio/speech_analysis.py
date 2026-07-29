"""Análise de fala: termos críticos / proxy de fadiga–disartria em texto."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

# Léxico educacional (PT-BR) — Azure Text Analytics equivalente por regras.
CRITICAL_TERMS: tuple[str, ...] = (
    "falta de ar",
    "falta d'ar",
    "dispneia",
    "dor no peito",
    "dor torácica",
    "dor toracica",
    "tontura",
    "tonteira",
    "caí",
    "cai",
    "caí no chão",
    "queda",
    "não tomei remédio",
    "nao tomei remedio",
    "não tomei o remédio",
    "esqueci o remédio",
    "esqueci o remedio",
    "cansaço",
    "cansaco",
    "muito cansado",
    "confusão",
    "confusao",
    "fraqueza",
    "palpitação",
    "palpitacao",
    "desmaio",
)


def _normalize(text: str) -> str:
    t = text.lower()
    t = t.replace("á", "a").replace("à", "a").replace("ã", "a").replace("â", "a")
    t = t.replace("é", "e").replace("ê", "e")
    t = t.replace("í", "i")
    t = t.replace("ó", "o").replace("ô", "o").replace("õ", "o")
    t = t.replace("ú", "u").replace("ü", "u")
    t = t.replace("ç", "c")
    t = re.sub(r"\s+", " ", t)
    return t


# Termos já normalizados para matching robusto
_NORM_TERMS = tuple(sorted({_normalize(t) for t in CRITICAL_TERMS}, key=len, reverse=True))


def find_critical_terms(text: str) -> list[str]:
    """Retorna termos críticos encontrados no texto (forma canônica)."""
    norm = _normalize(text)
    hits: list[str] = []
    for term in _NORM_TERMS:
        if term and term in norm:
            hits.append(term)
    return hits


def speech_risk_from_text(text: str) -> dict[str, Any]:
    """Score heurístico [0,1] a partir de densidade de termos críticos."""
    hits = find_critical_terms(text)
    # 0 hits → 0; 1 → 0.45; 2 → 0.7; 3+ → 0.9+
    if not hits:
        score = 0.0
    elif len(hits) == 1:
        score = 0.45
    elif len(hits) == 2:
        score = 0.7
    else:
        score = min(1.0, 0.55 + 0.15 * len(hits))
    return {
        "score": float(score),
        "hits": hits,
        "critical_terms": hits,
        "n_hits": len(hits),
        "transcript": text,
    }


def analyze_speech(audio_path: str | Path, transcript: dict[str, Any] | None = None) -> dict[str, Any]:
    """Extrai features linguísticas (termos críticos) e score de risco de fala.

    Args:
        audio_path: Caminho do arquivo de áudio.
        transcript: Resultado opcional da transcrição Whisper.

    Returns:
        Features e score de risco associado à fala.
    """
    audio_path = Path(audio_path)
    if transcript is None:
        from src.audio.transcription import transcribe

        transcript = transcribe(audio_path)

    text = str(transcript.get("text") or "")
    risk = speech_risk_from_text(text)
    return {
        **risk,
        "audio_path": str(audio_path.resolve()),
        "language": transcript.get("language"),
        "model_size": transcript.get("model_size"),
        "n_segments": len(transcript.get("segments") or []),
        "transcript_result": transcript,
        "aviso": (
            "Análise educacional por léxico de termos críticos (equivalente local a "
            "Azure Text Analytics). Não é diagnóstico clínico."
        ),
    }
