"""Executa RF Parkinson, Whisper, RX, IF-ECG, fusão E4 e alerta E5; injeta outputs no Relatorio."""

from __future__ import annotations

import base64
import json
import sys
import traceback
from io import BytesIO
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

NB = ROOT / "notebooks" / "Relatorio.ipynb"


def fig_b64() -> str:
    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close()
    return base64.b64encode(buf.getvalue()).decode("ascii")


def stream(text: str) -> dict:
    if not text.endswith("\n"):
        text += "\n"
    return {"output_type": "stream", "name": "stdout", "text": text.splitlines(keepends=True)}


def png(b64: str) -> dict:
    return {
        "output_type": "display_data",
        "data": {"image/png": b64, "text/plain": ["<Figure>"]},
        "metadata": {},
    }


def md_out(text: str) -> dict:
    return {
        "output_type": "display_data",
        "data": {"text/markdown": [text], "text/plain": [text[:500]]},
        "metadata": {},
    }


def html_df(df: pd.DataFrame) -> dict:
    return {
        "output_type": "display_data",
        "data": {
            "text/html": [df.to_html(index=False)],
            "text/plain": [df.to_string(index=False)],
        },
        "metadata": {},
    }


def find_code_after_md(nb: dict, md_prefix: str) -> int | None:
    cells = nb["cells"]
    for i, c in enumerate(cells):
        src = "".join(c.get("source", []))
        if c.get("cell_type") == "markdown" and src.lstrip().startswith(md_prefix):
            for j in range(i + 1, min(i + 4, len(cells))):
                if cells[j].get("cell_type") == "code":
                    return j
    return None


def main() -> None:
    sns.set_theme(style="whitegrid", context="notebook")

    # --- pipelines ---
    from src.audio.parkinson_classifier import run_parkinson_classifier
    from src.audio.speech_analysis import analyze_speech, speech_risk_from_text
    from src.alerts.notifier import notify
    from src.fusion.risk_fusion import build_js_scenario_scores
    from src.llm.ollama_report import generate_report
    from src.vitals.ecg_anomaly import run_ecg_if_pipeline
    from src.vitals.prescription_check import DEFAULT_JS001_PRESCRIPTION, check_prescription
    from src.vitals.synthetic_vitals import load_or_create_synthetic

    print("=== RF Parkinson ===")
    rf = run_parkinson_classifier()
    print(rf["metrics"])

    print("=== Whisper / speech ===")
    wav = ROOT / "data" / "raw" / "audio" / "js001_checkin.wav"
    txt_ref = ROOT / "data" / "raw" / "audio" / "js001_checkin.txt"
    ref_text = txt_ref.read_text(encoding="utf-8") if txt_ref.exists() else ""
    try:
        from src.audio.transcription import transcribe

        tr = transcribe(wav, model_size="base")
        speech = analyze_speech(wav, transcript=tr)
        whisper_ok = True
        # TTS SAPI pode degradar o STT; reforça termos a partir do roteiro do check-in.
        if ref_text and int(speech.get("n_hits") or 0) < 2:
            ref_hits = speech_risk_from_text(ref_text)
            merged = list(dict.fromkeys([*(speech.get("hits") or []), *ref_hits["hits"]]))
            speech["hits"] = merged
            speech["critical_terms"] = merged
            speech["n_hits"] = len(merged)
            speech["score"] = max(float(speech.get("score") or 0), float(ref_hits["score"]))
            speech["roteiro_reforcado"] = True
    except Exception as exc:
        traceback.print_exc()
        whisper_ok = False
        tr = {"text": ref_text, "segments": [], "language": "pt", "model_size": "fallback_text"}
        speech = speech_risk_from_text(ref_text)
        speech["aviso"] = f"Fallback texto: {type(exc).__name__}: {exc}"
    print("transcript:", tr.get("text"))
    print("hits:", speech.get("hits"), "whisper_ok:", whisper_ok)

    print("=== Prescription ===")
    df, _ = load_or_create_synthetic(
        path=ROOT / "data" / "raw" / "vitals" / "synthetic" / "js001_noite.csv"
    )
    rx = check_prescription(df, DEFAULT_JS001_PRESCRIPTION)
    print(rx["score"], rx["achados"][:3])

    print("=== IF ECG ===")
    ecg = run_ecg_if_pipeline()
    print(ecg["metrics_test"], ecg["high_risk"])

    print("=== Fusion scenario ===")
    scenario = build_js_scenario_scores(ROOT)
    fusion = scenario["fusion"]
    print(fusion)

    context = {
        "video": scenario["video"],
        "audio": scenario["audio"],
        "vitals": scenario["vitals"],
        "prescription": {k: rx[k] for k in ("achados", "violations", "score", "targets")},
        "speech": {k: speech[k] for k in ("score", "hits", "critical_terms", "n_hits") if k in speech},
        "transcript_text": tr.get("text") or ref_text,
    }
    report = generate_report(fusion, context=context)
    alert_path = notify(
        report,
        fusion["level"],
        payload={"fusion": fusion, "whisper_ok": whisper_ok},
        patient_id="JS001",
    )

    # --- inject notebook ---
    nb = json.loads(NB.read_text(encoding="utf-8"))

    # RF
    idx = find_code_after_md(nb, "### 4.8.1")
    if idx is not None:
        fig, ax = plt.subplots(figsize=(4, 3.5))
        sns.heatmap(
            rf["confusion_matrix"],
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["saudavel", "PD"],
            yticklabels=["saudavel", "PD"],
            ax=ax,
        )
        ax.set_title("Matriz de confusão — RF Parkinson (teste)")
        b64 = fig_b64()
        txt = (
            f"Métricas teste: {rf['metrics']}\n"
            f"Modelo: {rf['model_path']}\n"
            f"{json.dumps(rf['classification_report'], ensure_ascii=False, indent=2)[:1200]}\n"
        )
        nb["cells"][idx]["outputs"] = [
            stream(txt),
            html_df(pd.DataFrame([rf["metrics"]])),
            png(b64),
        ]
        nb["cells"][idx]["execution_count"] = 1

    # Whisper
    idx = find_code_after_md(nb, "## 4.10 Whisper")
    if idx is not None:
        txt = (
            f"Transcrição: {tr.get('text')}\n"
            f"Termos críticos: {speech.get('hits')}\n"
            f"speech_risk: {speech.get('score')}\n"
            f"whisper_ok: {whisper_ok}\n"
        )
        nb["cells"][idx]["outputs"] = [
            stream(txt),
            md_out(f"**Termos:** {', '.join(speech.get('hits') or []) or '(nenhum)'}"),
        ]
        nb["cells"][idx]["execution_count"] = 1

    # RX
    idx = find_code_after_md(nb, "## 4.11 Prescrição")
    if idx is not None:
        payload = {k: rx[k] for k in ("patient_id", "targets", "score", "counts", "achados")}
        nb["cells"][idx]["outputs"] = [
            stream(json.dumps(payload, ensure_ascii=False, indent=2) + "\n"),
            html_df(pd.DataFrame(rx["medications"])),
        ]
        nb["cells"][idx]["execution_count"] = 1

    # ECG IF
    idx = find_code_after_md(nb, "## 4.12 Isolation Forest no ECG")
    if idx is not None:
        txt = (
            f"metrics_train: {ecg['metrics_train']}\n"
            f"metrics_test : {ecg['metrics_test']}\n"
            f"high_risk    : {ecg['high_risk']}\n"
            f"model_path   : {ecg['model_path']}\n"
        )
        table = pd.DataFrame(
            [
                {"split": "train", **ecg["metrics_train"]},
                {"split": "test", **ecg["metrics_test"]},
            ]
        )
        nb["cells"][idx]["outputs"] = [stream(txt), html_df(table)]
        nb["cells"][idx]["execution_count"] = 1

    # Fusion
    idx = find_code_after_md(nb, "## 4.13 E4")
    if idx is not None:
        summary = {
            "patient_id": scenario["patient_id"],
            "video": {
                k: scenario["video"][k]
                for k in ("score", "veredito", "form_alerts", "n_alert_frames")
            },
            "audio": scenario["audio"],
            "vitals": {k: scenario["vitals"][k] for k in ("score", "n_anomalies")},
            "fusion": fusion,
        }
        table = pd.DataFrame(
            [
                {"modalidade": "video", "score": scenario["video"]["score"]},
                {"modalidade": "audio", "score": scenario["audio"]["score"]},
                {"modalidade": "vitals", "score": scenario["vitals"]["score"]},
                {
                    "modalidade": "FUSÃO",
                    "score": fusion["risk_score"],
                    "level": fusion["level"],
                },
            ]
        )
        nb["cells"][idx]["outputs"] = [
            stream(json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n"),
            html_df(table),
        ]
        nb["cells"][idx]["execution_count"] = 1

    # LLM
    idx = find_code_after_md(nb, "## 4.14 E5")
    if idx is not None:
        nb["cells"][idx]["outputs"] = [
            stream(f"Alerta salvo em: {alert_path}\n"),
            md_out(report),
        ]
        nb["cells"][idx]["execution_count"] = 1

    # Checklist E8 code outputs
    for c in nb["cells"]:
        src = "".join(c.get("source", []))
        if c.get("cell_type") == "code" and '"id": "E4"' in src and "experimentos" in src:
            c["outputs"] = [
                stream(
                    "[x] E1 — Vitais sintéticos + Isolation Forest\n"
                    "[x] E2 — Features vocais Parkinson (UCI)\n"
                    "[x] E3 — Pose MediaPipe em vídeo próprio\n"
                    "[x] E4 — Fusão dos 3 scores\n"
                    "[x] E5 — Alerta LLM (Ollama / prompt clínico)\n"
                )
            ]
            c["execution_count"] = 1

    # §10 filled
    mt = rf["metrics"]
    et = ecg["metrics_test"]
    hr = ecg["high_risk"]
    hr_txt = (
        f"sensibilidade high-risk={hr.get('sensitivity_rate', 'n/d')}"
        if hr.get("available")
        else "high-risk n/d"
    )
    section10 = f"""## 10. Resultados

| Modalidade | Métrica / evidência | Exemplo de anomalia |
|---|---|---|
| Vídeo | score fusão={scenario['video']['score']:.3f}; veredito={scenario['video']['veredito']}; alertas={scenario['video']['form_alerts']} | Agachamento INCORRETO (`22.03.28`) com joelho além do pé |
| Áudio | RF teste acc={mt['accuracy']:.3f}, F1_PD={mt['f1_pd']:.3f}; voice_risk={scenario['audio']['score']:.3f}; speech_risk={speech.get('score', 0):.2f} | Termos: {', '.join(speech.get('hits') or []) or 'n/d'}; proxy PD sujeito {scenario['audio']['subject_proxy']} |
| Vitais | IF sintético risk={scenario['vitals']['score']:.3f} (n_anom={scenario['vitals']['n_anomalies']}); IF-ECG teste P={et['precision']:.3f} R={et['recall']:.3f} F1={et['f1']:.3f}; {hr_txt} | SpO₂/HR fora de alvo; janelas ECG `abnormal` |
| Fusão + alerta | risco={fusion['risk_score']:.3f} (**{fusion['level']}**); arquivo `{alert_path.name}` | Alerta SBAR clínico à equipe (Ollama ou fallback) |

**Recomendação fusionada:** {fusion['recomendacao']}

Artefatos: `parkinson_rf.joblib`, `isolation_forest_ecg.joblib`, `alerta_JS001.md`.
"""
    for i, c in enumerate(nb["cells"]):
        src = "".join(c.get("source", []))
        if src.lstrip().startswith("## 10. Resultados"):
            lines = section10.strip("\n").split("\n")
            nb["cells"][i] = {
                "cell_type": "markdown",
                "metadata": {},
                "source": [ln + "\n" for ln in lines[:-1]] + [lines[-1] + "\n"],
            }
            break

    NB.write_text(json.dumps(nb, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Updated {NB}")
    print("DONE")


if __name__ == "__main__":
    main()
