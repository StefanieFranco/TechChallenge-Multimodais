"""Estimativa de pose com MediaPipe / YOLOv8."""

from pathlib import Path
from typing import Any


def estimate_pose(video_path: str | Path) -> dict[str, Any]:
    """Extrai keypoints de pose a partir de um vídeo.

    Args:
        video_path: Caminho do arquivo de vídeo.

    Returns:
        Dicionário com frames, landmarks e metadados.
    """
    raise NotImplementedError("Implementar estimativa de pose (MediaPipe/YOLOv8).")
