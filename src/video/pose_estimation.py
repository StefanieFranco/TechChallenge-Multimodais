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
    "LEFT_HEEL": 29,
    "RIGHT_HEEL": 30,
    "LEFT_FOOT_INDEX": 31,  # ponta do pé (referência do alerta de joelho)
    "RIGHT_FOOT_INDEX": 32,
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


# Conexões esqueléticas (índices MediaPipe Pose) para overlay.
POSE_EDGES: tuple[tuple[int, int], ...] = (
    (11, 12),  # shoulders
    (11, 13),
    (13, 15),  # left arm
    (12, 14),
    (14, 16),  # right arm
    (11, 23),
    (12, 24),  # torso
    (23, 24),  # hips
    (23, 25),
    (25, 27),  # left leg
    (27, 31),  # left ankle -> toe
    (24, 26),
    (26, 28),  # right leg
    (28, 32),  # right ankle -> toe
)

# Margem normalizada: joelho só alerta depois de ultrapassar a ponta do pé.
_KNEE_PAST_TOE_MARGIN = 0.01
# Elevação lateral na linha do ombro = OK.
# Correto tipicamente sobe até ~0.06; incorreto passa ~0.09+.
_ELBOW_ABOVE_SHOULDER_MARGIN = 0.07
# Só avalia joelho×pé com flexão (agachamento); em pé (elevação lateral) ignora.
_KNEE_MAX_ANGLE_DEG = 145.0
_VIS_THR_ALERT = 0.4


def _visible_xy(lm: np.ndarray, idx: int, thr: float = _VIS_THR_ALERT) -> tuple[float, float] | None:
    if idx >= len(lm) or float(lm[idx, 3]) < thr:
        return None
    return float(lm[idx, 0]), float(lm[idx, 1])


def _angle_deg_2d(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]) -> float | None:
    ba = np.array([a[0] - b[0], a[1] - b[1]], dtype=float)
    bc = np.array([c[0] - b[0], c[1] - b[1]], dtype=float)
    na, nc = np.linalg.norm(ba), np.linalg.norm(bc)
    if na < 1e-8 or nc < 1e-8:
        return None
    cos = float(np.clip(np.dot(ba, bc) / (na * nc), -1.0, 1.0))
    return float(np.degrees(np.arccos(cos)))


def detect_form_alerts(landmarks: np.ndarray, *, vis_thr: float = _VIS_THR_ALERT) -> dict[str, Any]:
    """Detecta alertas de forma educacionais (2D).

    - cotovelo/punho claramente acima do ombro (com margem): elevação lateral
      até a linha do ombro = OK
    - joelho passa a ponta do pé (FOOT_INDEX)
    """
    lm = np.asarray(landmarks, dtype=np.float32)
    alerts: list[str] = []
    hot_indices: set[int] = set()

    # Cotovelo/punho claramente acima do ombro (L/R)
    for side, sh_i, el_i, wr_i in (
        ("L", LM["LEFT_SHOULDER"], LM["LEFT_ELBOW"], LM["LEFT_WRIST"]),
        ("R", LM["RIGHT_SHOULDER"], LM["RIGHT_ELBOW"], LM["RIGHT_WRIST"]),
    ):
        sh = _visible_xy(lm, sh_i, vis_thr)
        el = _visible_xy(lm, el_i, vis_thr)
        wr = _visible_xy(lm, wr_i, vis_thr)
        if sh is None or el is None:
            continue
        # Ponto mais alto do braço (menor y)
        arm_y = el[1]
        if wr is not None:
            arm_y = min(arm_y, wr[1])
        if arm_y < sh[1] - _ELBOW_ABOVE_SHOULDER_MARGIN:
            alerts.append(f"braco {side} acima do ombro")
            hot_indices.update({sh_i, el_i})
            if wr is not None:
                hot_indices.add(wr_i)

    # Joelho passa a ponta do pé (FOOT_INDEX) — ponto de referência do exercício
    for side, hip_i, kn_i, toe_i, an_i in (
        ("L", LM["LEFT_HIP"], LM["LEFT_KNEE"], LM["LEFT_FOOT_INDEX"], LM["LEFT_ANKLE"]),
        ("R", LM["RIGHT_HIP"], LM["RIGHT_KNEE"], LM["RIGHT_FOOT_INDEX"], LM["RIGHT_ANKLE"]),
    ):
        hip = _visible_xy(lm, hip_i, vis_thr)
        kn = _visible_xy(lm, kn_i, vis_thr)
        toe = _visible_xy(lm, toe_i, vis_thr)
        an = _visible_xy(lm, an_i, vis_thr)
        if hip is None or kn is None:
            continue
        ref = toe if toe is not None else an
        ref_i = toe_i if toe is not None else an_i
        if ref is None:
            continue
        # Em pé / perna estendida (elevação lateral): não aplica regra de agachamento
        ang = _angle_deg_2d(hip, kn, ref)
        if ang is None or ang > _KNEE_MAX_ANGLE_DEG:
            continue
        forward = ref[0] - hip[0]
        if abs(forward) < 1e-3:
            forward = kn[0] - hip[0]
        if abs(forward) < 1e-3:
            continue
        direction = 1.0 if forward >= 0 else -1.0
        kn_fwd = (kn[0] - hip[0]) * direction
        toe_fwd = (ref[0] - hip[0]) * direction
        if kn_fwd > toe_fwd + _KNEE_PAST_TOE_MARGIN:
            alerts.append(f"joelho {side} passou a ponta do pe")
            hot_indices.update({kn_i, ref_i})

    return {
        "alerts": alerts,
        "hot_indices": sorted(hot_indices),
        "has_alert": bool(alerts),
    }


def _safe_stem(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in name)[:80]


def draw_landmarks_bgr(
    frame_bgr: np.ndarray,
    landmarks: np.ndarray,
    *,
    vis_thr: float = 0.4,
    form_alerts: dict[str, Any] | None = None,
) -> np.ndarray:
    """Desenha pontos/esqueleto; destaca em vermelho articulações com alerta."""
    out = frame_bgr.copy()
    h, w = out.shape[:2]
    lm = np.asarray(landmarks, dtype=np.float32)
    if form_alerts is None:
        form_alerts = detect_form_alerts(lm, vis_thr=vis_thr)
    hot = set(form_alerts.get("hot_indices") or [])
    alerts = list(form_alerts.get("alerts") or [])

    pts: dict[int, tuple[int, int]] = {}
    for i in range(len(lm)):
        if float(lm[i, 3]) < vis_thr:
            continue
        x = int(np.clip(lm[i, 0], 0, 1) * (w - 1))
        y = int(np.clip(lm[i, 1], 0, 1) * (h - 1))
        pts[i] = (x, y)
        if i in hot:
            cv2.circle(out, (x, y), 8, (0, 0, 255), -1, lineType=cv2.LINE_AA)
            cv2.circle(out, (x, y), 12, (0, 0, 255), 2, lineType=cv2.LINE_AA)
        else:
            cv2.circle(out, (x, y), 4, (0, 255, 255), -1, lineType=cv2.LINE_AA)

    for a, b in POSE_EDGES:
        if a in pts and b in pts:
            color = (0, 0, 255) if (a in hot and b in hot) else (0, 200, 0)
            thickness = 3 if (a in hot and b in hot) else 2
            cv2.line(out, pts[a], pts[b], color, thickness, lineType=cv2.LINE_AA)

    # Marca a ponta do pé (referência do limite do joelho) em azul
    for toe_i in (LM["LEFT_FOOT_INDEX"], LM["RIGHT_FOOT_INDEX"]):
        if toe_i in pts:
            cv2.circle(out, pts[toe_i], 10, (255, 128, 0), 2, lineType=cv2.LINE_AA)
            cv2.putText(
                out,
                "limite pe",
                (pts[toe_i][0] + 8, pts[toe_i][1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 128, 0),
                1,
                cv2.LINE_AA,
            )

    # Faixa de alertas no topo
    y0 = 28
    cv2.putText(
        out,
        "ALERTA FORMA" if alerts else "forma OK",
        (12, y0),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 0, 255) if alerts else (0, 200, 0),
        2,
        cv2.LINE_AA,
    )
    for j, msg in enumerate(alerts[:4]):
        cv2.putText(
            out,
            msg,
            (12, y0 + 28 * (j + 1)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
    return out


def render_pose_overlays(
    video_path: str | Path,
    *,
    pose_data: dict[str, Any] | None = None,
    max_frames: int = 6,
    sample_fps: float = 3.0,
    output_root: Path | None = None,
    prefer_alerts: bool = True,
) -> dict[str, Any]:
    """Gera PNGs com landmarks e alertas de forma sobre frames do vídeo.

    Prioriza frames com alerta (cotovelo acima do ombro / joelho à frente do pé)
    quando ``prefer_alerts=True``.
    """
    video_path = Path(video_path)
    if pose_data is None:
        pose_data = estimate_pose(video_path, sample_fps=sample_fps)

    root = _project_root()
    out_root = Path(output_root) if output_root else (root / "data" / "processed" / "videos" / "overlays")
    stem = _safe_stem(video_path.stem)
    out_dir = out_root / stem
    out_dir.mkdir(parents=True, exist_ok=True)

    for old in out_dir.glob("frame_*.png"):
        old.unlink()

    candidates = [fr for fr in (pose_data.get("frames") or []) if fr.get("landmarks") is not None]
    if not candidates:
        return {"paths": [], "output_dir": str(out_dir), "n_frames": 0, "alerts_summary": []}

    enriched: list[dict[str, Any]] = []
    for fr in candidates:
        fa = detect_form_alerts(fr["landmarks"])
        enriched.append({**fr, "form_alerts": fa})

    alert_frames = [fr for fr in enriched if fr["form_alerts"]["has_alert"]]
    normal_frames = [fr for fr in enriched if not fr["form_alerts"]["has_alert"]]

    chosen: list[dict[str, Any]] = []
    if prefer_alerts and alert_frames:
        # pega alertas espaçados + completa com normais se faltar
        if len(alert_frames) <= max_frames:
            chosen = list(alert_frames)
        else:
            idxs = np.linspace(0, len(alert_frames) - 1, max_frames, dtype=int)
            chosen = [alert_frames[i] for i in idxs]
        remain = max_frames - len(chosen)
        if remain > 0 and normal_frames:
            idxs = np.linspace(0, len(normal_frames) - 1, remain, dtype=int)
            chosen.extend(normal_frames[i] for i in idxs)
    else:
        if len(enriched) <= max_frames:
            chosen = enriched
        else:
            idxs = np.linspace(0, len(enriched) - 1, max_frames, dtype=int)
            chosen = [enriched[i] for i in idxs]

    # ordena por tempo no vídeo
    chosen.sort(key=lambda fr: int(fr["frame_idx"]))

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Não foi possível abrir o vídeo: {video_path}")

    paths: list[str] = []
    per_frame_alerts: list[list[str]] = []
    for i, fr in enumerate(chosen):
        frame_idx = int(fr["frame_idx"])
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame_bgr = cap.read()
        if not ok:
            continue
        fa = fr["form_alerts"]
        drawn = draw_landmarks_bgr(frame_bgr, fr["landmarks"], form_alerts=fa)
        cv2.putText(
            drawn,
            f"frame {frame_idx}",
            (12, drawn.shape[0] - 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        path = out_dir / f"frame_{i:02d}.png"
        cv2.imwrite(str(path), drawn)
        paths.append(str(path.resolve()))
        per_frame_alerts.append(list(fa.get("alerts") or []))

    cap.release()

    # resumo único de alertas no vídeo
    all_alerts: list[str] = []
    for msgs in per_frame_alerts:
        for m in msgs:
            if m not in all_alerts:
                all_alerts.append(m)

    return {
        "paths": paths,
        "output_dir": str(out_dir.resolve()),
        "n_frames": len(paths),
        "video_name": video_path.name,
        "alerts_summary": all_alerts,
        "per_frame_alerts": per_frame_alerts,
        "n_alert_frames_available": len(alert_frames),
    }
