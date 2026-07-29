"""Combina scores de vídeo, áudio e vitais em um risco final."""

from __future__ import annotations

from pathlib import Path
from typing import Any

LEVEL_LOW = 0.33
LEVEL_HIGH = 0.66
ALERT_BOOST = 0.15
JS_INCORRECT_VIDEO = "WhatsApp Video 2026-07-27 at 22.03.28.mp4"


def _project_root() -> Path:
    start = Path.cwd().resolve()
    for candidate in [start, *start.parents]:
        if (candidate / "requirements.txt").exists() and (candidate / "src").exists():
            return candidate
    return start


def _clip01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def risk_level(score: float) -> str:
    """Mapeia score [0,1] para baixo / moderado / alto."""
    if score < LEVEL_LOW:
        return "baixo"
    if score < LEVEL_HIGH:
        return "moderado"
    return "alto"


def fuse_risk_scores(
    video_score: float,
    audio_score: float,
    vitals_score: float,
    weights: tuple[float, float, float] = (1 / 3, 1 / 3, 1 / 3),
) -> dict[str, Any]:
    """Fusão ponderada dos scores de risco por modalidade.

    Args:
        video_score: Score de risco do módulo de vídeo [0, 1].
        audio_score: Score de risco do módulo de áudio [0, 1].
        vitals_score: Score de risco do módulo de vitais [0, 1].
        weights: Pesos (vídeo, áudio, vitais).

    Returns:
        Score final, nível, breakdown e recomendação curta.
    """
    wv, wa, ww = (float(w) for w in weights)
    total = wv + wa + ww
    if total <= 0:
        raise ValueError("Soma dos pesos deve ser > 0.")
    wv, wa, ww = wv / total, wa / total, ww / total

    vs = _clip01(float(video_score))
    aus = _clip01(float(audio_score))
    vts = _clip01(float(vitals_score))
    fused = _clip01(wv * vs + wa * aus + ww * vts)
    level = risk_level(fused)

    if level == "alto":
        recomendacao = (
            "Priorizar reavaliação multiprofissional (fisioterapia + enfermagem + médico) "
            "e conferir sinais vitais / aderência à prescrição."
        )
    elif level == "moderado":
        recomendacao = (
            "Manter vigilância ampliada; reforçar técnica do exercício e monitorar SpO₂/FC "
            "nas próximas janelas."
        )
    else:
        recomendacao = "Manter plano atual e reavaliar na próxima sessão de monitoramento."

    return {
        "risk_score": fused,
        "level": level,
        "breakdown": {
            "video": vs,
            "audio": aus,
            "vitals": vts,
        },
        "weights": {"video": wv, "audio": wa, "vitals": ww},
        "recomendacao": recomendacao,
        "thresholds": {"baixo_max": LEVEL_LOW, "moderado_max": LEVEL_HIGH},
    }


def build_js_scenario_scores(
    root: Path | None = None,
    *,
    video_name: str = JS_INCORRECT_VIDEO,
    with_overlays: bool = False,
) -> dict[str, Any]:
    """Monta os 3 scores do paciente fictício J.S. a partir dos artefatos locais.

    - Vídeo: clip INCORRETO (agachamento) + boost se houver alertas de forma.
    - Áudio: mediana voice_risk_score do sujeito PD com maior risco (proxy UCI).
    - Vitais: risk_score do Isolation Forest sintético (noite JS-001).
    """
    root = root or _project_root()

    from src.audio.parkinsons_analysis import analyze_parkinsons_dir, score_voice_risk
    from src.video.anomaly_report import analyze_video
    from src.vitals.anomaly_detection import detect_anomalies, load_model
    from src.vitals.synthetic_vitals import load_or_create_synthetic

    video_path = root / "data" / "raw" / "videos" / video_name
    if not video_path.exists():
        raise FileNotFoundError(f"Vídeo do cenário J.S. não encontrado: {video_path}")

    video_report = analyze_video(video_path, with_overlays=with_overlays)
    video_score = float(video_report["score"])
    n_alerts = int(video_report.get("n_alert_frames") or 0)
    if n_alerts <= 0 and video_report.get("form_alerts"):
        n_alerts = len(video_report["form_alerts"])
    if n_alerts > 0 or (video_report.get("veredito") == "INCORRETO"):
        # Boost educacional: exercício incorreto / alertas de forma elevam risco motor.
        video_score = _clip01(max(video_score, 0.55) + (ALERT_BOOST if n_alerts > 0 else 0.1))

    pk = analyze_parkinsons_dir(root / "data" / "raw" / "parkinsons")
    by_subj = score_voice_risk(pk["df"], by_subject=True)
    top = pk["top_subjects"]
    if top is not None and len(top):
        audio_subject = str(top.iloc[0]["subject"])
        audio_score = float(top.iloc[0]["voice_risk_score"])
    else:
        audio_subject = str(by_subj.idxmax())
        audio_score = float(by_subj.max())

    csv = root / "data" / "raw" / "vitals" / "synthetic" / "js001_noite.csv"
    model_path = root / "data" / "processed" / "vitals" / "isolation_forest_vitals.joblib"
    df, _ = load_or_create_synthetic(path=csv, force=False)
    model = load_model(model_path) if model_path.exists() else None
    vitals_result = detect_anomalies(df, model=model)
    vitals_score = float(vitals_result["risk_score"])

    fusion = fuse_risk_scores(video_score, audio_score, vitals_score)

    return {
        "patient_id": "JS-001",
        "video": {
            "score": video_score,
            "raw_asymmetry_score": float(video_report["score"]),
            "veredito": video_report.get("veredito"),
            "form_alerts": video_report.get("form_alerts") or [],
            "n_alert_frames": n_alerts,
            "video_name": video_report.get("video_name"),
            "achados": video_report.get("achados") or [],
        },
        "audio": {
            "score": audio_score,
            "subject_proxy": audio_subject,
            "source": "UCI_Parkinson_voice_risk_score",
        },
        "vitals": {
            "score": vitals_score,
            "n_anomalies": vitals_result["n_anomalies"],
            "model_name": vitals_result["model_name"],
            "feature_cols": vitals_result["feature_cols"],
        },
        "fusion": fusion,
        "video_report": video_report,
        "vitals_result": vitals_result,
        "parkinsons": pk,
    }
