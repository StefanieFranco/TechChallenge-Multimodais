"""Gera WAV de check-in do paciente J.S. (TTS Windows SAPI ou fallback PCM)."""

from __future__ import annotations

import struct
import subprocess
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "raw" / "audio" / "js001_checkin.wav"
SCRIPT_TXT = ROOT / "data" / "raw" / "audio" / "js001_checkin.txt"

CHECKIN_TEXT = (
    "Olá, sou o paciente J.S. Hoje estou com bastante cansaço e um pouco de falta de ar "
    "depois da fisioterapia. Também senti tontura ao levantar. "
    "Esqueci o remédio da pressão esta manhã. "
    "A dor no peito foi leve e passou rápido."
)


def _write_tone_wav(path: Path, duration_s: float = 4.0, sr: int = 16000) -> None:
    """Fallback: tom simples + arquivo de texto ao lado (Whisper pode falhar no tom)."""
    import math

    path.parent.mkdir(parents=True, exist_ok=True)
    n = int(duration_s * sr)
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        for i in range(n):
            # envelope + freq variável só para ter energia acústica
            t = i / sr
            amp = 0.3 * (0.5 + 0.5 * math.sin(2 * math.pi * 0.5 * t))
            val = int(amp * 32767 * math.sin(2 * math.pi * (180 + 40 * math.sin(2 * math.pi * 0.7 * t)) * t))
            wf.writeframes(struct.pack("<h", val))


def _tts_windows(path: Path, text: str) -> bool:
    """Usa System.Speech (Windows) para gravar WAV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ps = f"""
Add-Type -AssemblyName System.Speech
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
$synth.Rate = -1
$synth.Volume = 100
$synth.SetOutputToWaveFile('{str(path).replace("'", "''")}')
$synth.Speak('{text.replace("'", "''")}')
$synth.Dispose()
"""
    try:
        r = subprocess.run(
            ["powershell", "-NoProfile", "-Command", ps],
            capture_output=True,
            text=True,
            timeout=60,
        )
        return r.returncode == 0 and path.exists() and path.stat().st_size > 1000
    except Exception:
        return False


def main() -> None:
    SCRIPT_TXT.parent.mkdir(parents=True, exist_ok=True)
    SCRIPT_TXT.write_text(CHECKIN_TEXT, encoding="utf-8")
    ok = _tts_windows(OUT, CHECKIN_TEXT)
    if not ok:
        print("TTS Windows falhou — gravando WAV de fallback (tom).")
        _write_tone_wav(OUT)
        # Para demo sem fala: speech_analysis usa o texto do script se passado
        print(f"Texto de referência: {SCRIPT_TXT}")
    else:
        print(f"WAV gerado via SAPI: {OUT}")
    print(f"Bytes: {OUT.stat().st_size}")


if __name__ == "__main__":
    main()
