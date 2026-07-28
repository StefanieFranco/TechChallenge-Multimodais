"""Estimativa de pose com MediaPipe Pose Landmarker (Tasks API)."""

from __future__ import annotations

import urllib.request
from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# Índices MediaPipe Pose (33 landmarks).
LM = {
    "LEFT_SHOULDER": 11,
    "RIGHT_SHOULDER": 12,
    "LEFT_ELBOW": 13,
    "RIGHT_ELBOW": 14,
    "LEFT_WRIST": 15,
    "RIGHT_WRIST": 16,
    "LEFT_HIP": 23,
    "RIGHT_HIP": 24,
    "LEFT_KNEE": 25,
    "RIGHT_KNEE": 26,
    "LEFT_ANKLE": 27,
    "RIGHT_ANKLE": 28,
}

_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)


def _project_root() -> Path:
    start = Path.cwd().resolve()
    for candidate in [start, *start.parents]:
        if (candidate / "requirements.txt").exists() and (candidate / "src").exists():
            return candidate
    return start


def default_model_path(root: Path | None = None) -> Path:
    root = root or _project_root()
    return root / "models" / "mediapipe" / "pose_landmarker_lite.task"


def ensure_pose_model(model_path: Path | None = None) -> Path:
    """Garante que o .task do Pose Landmarker existe (download se necessário)."""
    path = Path(model_path) if model_path else default_model_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        return path
    print(f"[download] MediaPipe pose model -> {path}")
    urllib.request.urlretrieve(_MODEL_URL, path)
    return path


def _landmarks_to_array(landmarks) -> np.ndarray:
    """Converte lista de NormalizedLandmark em array (33, 4): x, y, z, visibility."""
    rows = []
    for lm in landmarks:
        vis = getattr(lm, "visibility", None)
        if vis is None:
            vis = getattr(lm, "presence", 0.0)
        rows.append([lm.x, lm.y, lm.z, float(vis)])
    return np.asarray(rows, dtype=np.float32)


def estimate_pose(
    video_path: str | Path,
    *,
    sample_fps: float = 3.0,
    model_path: Path | None = None,
    min_detection_confidence: float = 0.5,
) -> dict[str, Any]:
    """Extrai keypoints de pose a partir de um vídeo.

    Amostra frames a ~``sample_fps`` para custo acadêmico razoável.
    Usa MediaPipe Tasks PoseLandmarker (VIDEO mode).

    Args:
        video_path: Caminho do arquivo de vídeo.
        sample_fps: Taxa de amostragem alvo (frames analisados por segundo).
        model_path: Caminho opcional do ``.task``; senão usa ``models/mediapipe/``.
        min_detection_confidence: Limiar de confiança do landmarker.

    Returns:
        Dicionário com frames amostrados, landmarks e metadados.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Vídeo não encontrado: {video_path}")

    model = ensure_pose_model(model_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Não foi possível abrir o vídeo: {video_path}")

    native_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if native_fps <= 1e-3:
        native_fps = 30.0
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    frame_interval = max(1, int(round(native_fps / max(sample_fps, 0.1))))

    options = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model)),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=min_detection_confidence,
        min_pose_presence_confidence=min_detection_confidence,
        min_tracking_confidence=min_detection_confidence,
    )

    sampled: list[dict[str, Any]] = []
    detected = 0
    frame_idx = 0

    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            if frame_idx % frame_interval == 0:
                rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                timestamp_ms = int((frame_idx / native_fps) * 1000)
                result = landmarker.detect_for_video(mp_image, timestamp_ms)
                entry: dict[str, Any] = {
                    "frame_idx": frame_idx,
                    "timestamp_ms": timestamp_ms,
                    "landmarks": None,
                }
                if result.pose_landmarks:
                    arr = _landmarks_to_array(result.pose_landmarks[0])
                    entry["landmarks"] = arr
                    detected += 1
                sampled.append(entry)
            frame_idx += 1

    cap.release()
    n_sampled = len(sampled)
    detection_rate = (detected / n_sampled) if n_sampled else 0.0

    return {
        "video_path": str(video_path.resolve()),
        "video_name": video_path.name,
        "fps": native_fps,
        "sample_fps": sample_fps,
        "frame_interval": frame_interval,
        "n_frames_total": n_total or frame_idx,
        "n_frames_sampled": n_sampled,
        "n_frames_detected": detected,
        "detection_rate": detection_rate,
        "width": width,
        "height": height,
        "landmark_names": LM,
        "frames": sampled,
        "backend": "mediapipe_pose_landmarker_lite",
    }
