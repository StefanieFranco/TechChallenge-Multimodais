"""Relatório de anomalias motoras por assimetria L/R (cenário pós-AVC)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.video.pose_estimation import LM, estimate_pose

# Limiares educacionais (score ∈ [0, 1]): maior = mais assimétrico / risco motor.
SCORE_CORRETO_MAX = 0.25
SCORE_ATENCAO_MAX = 0.50

# Diferença angular (graus) que contribui fortemente ao score.
ANGLE_ASYM_SOFT = 12.0
ANGLE_ASYM_HARD = 35.0
TRUNK_SOFT = 8.0
TRUNK_HARD = 20.0


def _angle_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float | None:
    """Ângulo em B formado por pontos A-B-C (coordenadas 2D x,y)."""
    ba = a[:2] - b[:2]
    bc = c[:2] - b[:2]
    na = np.linalg.norm(ba)
    nc = np.linalg.norm(bc)
    if na < 1e-8 or nc < 1e-8:
        return None
    cos = float(np.clip(np.dot(ba, bc) / (na * nc), -1.0, 1.0))
    return float(np.degrees(np.arccos(cos)))


def _visible(lm: np.ndarray, idx: int, thr: float = 0.4) -> bool:
    return float(lm[idx, 3]) >= thr


def _joint_angles(lm: np.ndarray) -> dict[str, float | None]:
    """Ângulos articulares relevantes para assimetria pós-AVC."""
    out: dict[str, float | None] = {
        "shoulder_l": None,
        "shoulder_r": None,
        "hip_l": None,
        "hip_r": None,
        "knee_l": None,
        "knee_r": None,
        "trunk_lean": None,
    }

    def pts(*keys: str) -> list[np.ndarray] | None:
        idxs = [LM[k] for k in keys]
        if not all(_visible(lm, i) for i in idxs):
            return None
        return [lm[i] for i in idxs]

    sh_l = pts("LEFT_ELBOW", "LEFT_SHOULDER", "LEFT_HIP")
    if sh_l:
        out["shoulder_l"] = _angle_deg(*sh_l)
    sh_r = pts("RIGHT_ELBOW", "RIGHT_SHOULDER", "RIGHT_HIP")
    if sh_r:
        out["shoulder_r"] = _angle_deg(*sh_r)

    hip_l = pts("LEFT_SHOULDER", "LEFT_HIP", "LEFT_KNEE")
    if hip_l:
        out["hip_l"] = _angle_deg(*hip_l)
    hip_r = pts("RIGHT_SHOULDER", "RIGHT_HIP", "RIGHT_KNEE")
    if hip_r:
        out["hip_r"] = _angle_deg(*hip_r)

    kn_l = pts("LEFT_HIP", "LEFT_KNEE", "LEFT_ANKLE")
    if kn_l:
        out["knee_l"] = _angle_deg(*kn_l)
    kn_r = pts("RIGHT_HIP", "RIGHT_KNEE", "RIGHT_ANKLE")
    if kn_r:
        out["knee_r"] = _angle_deg(*kn_r)

    # Inclinação do tronco: vetor mid-ombro → mid-quadril vs vertical.
    need = ["LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_HIP", "RIGHT_HIP"]
    if all(_visible(lm, LM[k]) for k in need):
        mid_sh = 0.5 * (lm[LM["LEFT_SHOULDER"], :2] + lm[LM["RIGHT_SHOULDER"], :2])
        mid_hip = 0.5 * (lm[LM["LEFT_HIP"], :2] + lm[LM["RIGHT_HIP"], :2])
        trunk = mid_hip - mid_sh
        norm = np.linalg.norm(trunk)
        if norm > 1e-8:
            # Vertical em imagem: +y para baixo → vetor (0, 1).
            cos = float(np.clip(np.dot(trunk / norm, np.array([0.0, 1.0])), -1.0, 1.0))
            out["trunk_lean"] = float(np.degrees(np.arccos(cos)))

    return out


def _norm_asym(diff_deg: float, soft: float, hard: float) -> float:
    """Mapeia diferença em graus para [0, 1]."""
    if diff_deg <= soft:
        return 0.0
    if diff_deg >= hard:
        return 1.0
    return (diff_deg - soft) / (hard - soft)


def _pair_diff(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return abs(left - right)


def build_anomaly_report(pose_data: dict[str, Any]) -> dict[str, Any]:
    """Gera score e descrição de anomalias motoras por assimetria L/R.

    Heurística educacional (não diagnóstico): compara ângulos de ombro, quadril
    e joelho esquerda/direita e a inclinação de tronco — alinhado ao cenário
    J.S. pós-AVC (assimetria / compensação).

    Args:
        pose_data: Saída de ``estimate_pose``.

    Returns:
        Relatório com score de risco, veredito e achados.
    """
    frames = pose_data.get("frames") or []
    shoulder_diffs: list[float] = []
    hip_diffs: list[float] = []
    knee_diffs: list[float] = []
    trunk_leans: list[float] = []

    for fr in frames:
        lm = fr.get("landmarks")
        if lm is None:
            continue
        lm = np.asarray(lm, dtype=np.float32)
        ang = _joint_angles(lm)
        d_sh = _pair_diff(ang["shoulder_l"], ang["shoulder_r"])
        d_hip = _pair_diff(ang["hip_l"], ang["hip_r"])
        d_kn = _pair_diff(ang["knee_l"], ang["knee_r"])
        if d_sh is not None:
            shoulder_diffs.append(d_sh)
        if d_hip is not None:
            hip_diffs.append(d_hip)
        if d_kn is not None:
            knee_diffs.append(d_kn)
        if ang["trunk_lean"] is not None:
            trunk_leans.append(float(ang["trunk_lean"]))

    def med(xs: list[float]) -> float | None:
        return float(np.median(xs)) if xs else None

    metrics = {
        "shoulder_asym_deg": med(shoulder_diffs),
        "hip_asym_deg": med(hip_diffs),
        "knee_asym_deg": med(knee_diffs),
        "trunk_lean_deg": med(trunk_leans),
        "n_pose_frames": int(pose_data.get("n_frames_detected") or 0),
        "detection_rate": float(pose_data.get("detection_rate") or 0.0),
    }

    components: dict[str, float] = {}
    if metrics["shoulder_asym_deg"] is not None:
        components["ombro"] = _norm_asym(
            metrics["shoulder_asym_deg"], ANGLE_ASYM_SOFT, ANGLE_ASYM_HARD
        )
    if metrics["hip_asym_deg"] is not None:
        components["quadril"] = _norm_asym(
            metrics["hip_asym_deg"], ANGLE_ASYM_SOFT, ANGLE_ASYM_HARD
        )
    if metrics["knee_asym_deg"] is not None:
        components["joelho"] = _norm_asym(
            metrics["knee_asym_deg"], ANGLE_ASYM_SOFT, ANGLE_ASYM_HARD
        )
    if metrics["trunk_lean_deg"] is not None:
        components["tronco"] = _norm_asym(metrics["trunk_lean_deg"], TRUNK_SOFT, TRUNK_HARD)

    if not components:
        score = 1.0 if metrics["detection_rate"] < 0.2 else 0.5
        achados = ["pose insuficiente para medir assimetria"]
        veredito = "ATENCAO"
    else:
        # Pesos: joelho e tronco mais relevantes no cenário de marcha/fisio.
        weights = {"ombro": 0.2, "quadril": 0.25, "joelho": 0.3, "tronco": 0.25}
        num = sum(components[k] * weights[k] for k in components)
        den = sum(weights[k] for k in components)
        score = float(num / den) if den > 0 else 0.0

        # Baixa detecção eleva incerteza (não marca CORRETO cegamente).
        if metrics["detection_rate"] < 0.35:
            score = max(score, 0.35)

        achados = []
        labels = {
            "ombro": ("assimetria de ombro", metrics["shoulder_asym_deg"]),
            "quadril": ("assimetria de quadril", metrics["hip_asym_deg"]),
            "joelho": ("assimetria de joelho", metrics["knee_asym_deg"]),
            "tronco": ("compensação / inclinação de tronco", metrics["trunk_lean_deg"]),
        }
        for key, val in sorted(components.items(), key=lambda kv: kv[1], reverse=True):
            if val >= 0.35:
                nome, deg = labels[key]
                achados.append(f"{nome} elevada ({deg:.1f} deg)")
        if not achados:
            achados.append("simetria articular dentro dos limiares educacionais")

        if score < SCORE_CORRETO_MAX:
            veredito = "CORRETO"
        elif score < SCORE_ATENCAO_MAX:
            veredito = "ATENCAO"
        else:
            veredito = "INCORRETO"

    return {
        "video_name": pose_data.get("video_name"),
        "video_path": pose_data.get("video_path"),
        "score": round(score, 4),
        "veredito": veredito,
        "achados": achados,
        "metrics": metrics,
        "components": {k: round(v, 4) for k, v in components.items()},
        "thresholds": {
            "CORRETO": f"score < {SCORE_CORRETO_MAX}",
            "ATENCAO": f"{SCORE_CORRETO_MAX} <= score < {SCORE_ATENCAO_MAX}",
            "INCORRETO": f"score >= {SCORE_ATENCAO_MAX}",
        },
        "aviso": (
            "Heurística acadêmica de assimetria L/R (cenário J.S. pós-AVC). "
            "Não constitui diagnóstico clínico."
        ),
    }


def analyze_video(
    video_path: str | Path,
    *,
    sample_fps: float = 3.0,
) -> dict[str, Any]:
    """Encadeia estimativa de pose + relatório de assimetria."""
    pose = estimate_pose(video_path, sample_fps=sample_fps)
    report = build_anomaly_report(pose)
    report["pose_meta"] = {
        "fps": pose["fps"],
        "n_frames_total": pose["n_frames_total"],
        "n_frames_sampled": pose["n_frames_sampled"],
        "n_frames_detected": pose["n_frames_detected"],
        "detection_rate": pose["detection_rate"],
        "backend": pose["backend"],
    }
    return report


def analyze_videos_dir(
    videos_dir: str | Path,
    *,
    sample_fps: float = 3.0,
    pattern: str = "*.mp4",
) -> list[dict[str, Any]]:
    """Analisa todos os vídeos que casam com ``pattern`` em uma pasta."""
    root = Path(videos_dir)
    paths = sorted(root.glob(pattern))
    return [analyze_video(p, sample_fps=sample_fps) for p in paths]
