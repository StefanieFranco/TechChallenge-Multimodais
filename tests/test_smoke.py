"""Smoke test da estrutura do projeto."""

from pathlib import Path


def test_project_layout_exists():
    root = Path(__file__).resolve().parents[1]
    expected = [
        "requirements.txt",
        "src/video/pose_estimation.py",
        "src/audio/transcription.py",
        "src/vitals/anomaly_detection.py",
        "src/fusion/risk_fusion.py",
        "src/llm/ollama_report.py",
        "src/alerts/notifier.py",
        "docs/relatorio_tecnico.md",
    ]
    for path in expected:
        assert (root / path).exists(), f"Arquivo ausente: {path}"
