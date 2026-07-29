"""Atualiza pesos clínicos na fusão, regenera §4.13/§4.14/§10 no Relatorio."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
NB = ROOT / "notebooks" / "Relatorio.ipynb"

MD_413 = """## 4.13 E4 — Fusão multimodal dos 3 scores

`src/fusion/risk_fusion.py` combina vídeo (clip incorreto + alertas), áudio (`voice_risk_score` PD)
e vitais (IF sintético JS-001) em risco global + breakdown.

### Por que esses pesos?

Pesos **iguais (1/3 cada)** seriam a *baseline* neutra. No MVP ajustamos para **prioridade clínica**
de monitoramento contínuo:

| Modalidade | Peso | Motivo |
|---|---|---|
| Vitais | **0.55** | Maior ameaça fisiológica (SpO₂/FC) |
| Vídeo (motor) | **0.25** | Segurança/técnica do exercício |
| Áudio | **0.20** | Proxy UCI (menos específico do J.S.) |

O **breakdown** por modalidade continua visível — os pesos só definem a agregação, não escondem evidências.
"""

CODE_413 = """from pathlib import Path
import sys
import json

import pandas as pd
from IPython.display import display

ROOT = Path("..").resolve()
if not (ROOT / "src").exists():
    ROOT = Path(".").resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.fusion.risk_fusion import CLINICAL_WEIGHTS, WEIGHTS_RATIONALE, build_js_scenario_scores

scenario = build_js_scenario_scores(ROOT)
fusion = scenario["fusion"]
print("Pesos clínicos:", CLINICAL_WEIGHTS)
print("Racional:", WEIGHTS_RATIONALE)
print(json.dumps({
    "patient_id": scenario["patient_id"],
    "video": {k: scenario["video"][k] for k in ("score", "veredito", "form_alerts", "n_alert_frames")},
    "audio": scenario["audio"],
    "vitals": {k: scenario["vitals"][k] for k in ("score", "n_anomalies")},
    "fusion": fusion,
}, ensure_ascii=False, indent=2))
display(pd.DataFrame([
    {"modalidade": "video", "score": scenario["video"]["score"], "peso": fusion["weights"]["video"]},
    {"modalidade": "audio", "score": scenario["audio"]["score"], "peso": fusion["weights"]["audio"]},
    {"modalidade": "vitals", "score": scenario["vitals"]["score"], "peso": fusion["weights"]["vitals"]},
    {"modalidade": "FUSÃO", "score": fusion["risk_score"], "peso": 1.0, "level": fusion["level"]},
]))
"""


def _cell_source(text: str) -> list[str]:
    lines = text.strip("\n").split("\n")
    return [ln + "\n" for ln in lines[:-1]] + ([lines[-1] + "\n"] if lines else [])


def stream(text: str) -> dict:
    if not text.endswith("\n"):
        text += "\n"
    return {"output_type": "stream", "name": "stdout", "text": text.splitlines(keepends=True)}


def html_df(df) -> dict:
    return {
        "output_type": "display_data",
        "data": {
            "text/html": [df.to_html(index=False)],
            "text/plain": [df.to_string(index=False)],
        },
        "metadata": {},
    }


def md_out(text: str) -> dict:
    return {
        "output_type": "display_data",
        "data": {"text/markdown": [text], "text/plain": [text[:500]]},
        "metadata": {},
    }


def main() -> None:
    import pandas as pd

    from src.alerts.notifier import notify
    from src.audio.speech_analysis import speech_risk_from_text
    from src.fusion.risk_fusion import CLINICAL_WEIGHTS, WEIGHTS_RATIONALE, build_js_scenario_scores
    from src.llm.ollama_report import JS001_PROFILE, generate_report
    from src.vitals.prescription_check import check_prescription
    from src.vitals.synthetic_vitals import load_or_create_synthetic

    print("Recalculando cenário com pesos", CLINICAL_WEIGHTS)
    scenario = build_js_scenario_scores(ROOT)
    fusion = scenario["fusion"]
    print(fusion)

    df, _ = load_or_create_synthetic(
        path=ROOT / "data" / "raw" / "vitals" / "synthetic" / "js001_noite.csv"
    )
    rx = check_prescription(df)
    txt = (ROOT / "data" / "raw" / "audio" / "js001_checkin.txt").read_text(encoding="utf-8")
    speech = speech_risk_from_text(txt)
    context = {
        "video": scenario["video"],
        "audio": scenario["audio"],
        "vitals": scenario["vitals"],
        "prescription": {k: rx[k] for k in ("achados", "violations", "score", "targets")},
        "speech": speech,
        "transcript_text": txt,
        "weights_rationale": WEIGHTS_RATIONALE,
        "patient_profile": JS001_PROFILE,
    }
    report = generate_report(fusion, context=context)
    alert_path = notify(
        report,
        fusion["level"],
        payload={"fusion": fusion},
        patient_id="JS001",
    )

    nb = json.loads(NB.read_text(encoding="utf-8"))
    cells = nb["cells"]

    # Update §4.13 markdown + following code
    for i, c in enumerate(cells):
        src = "".join(c.get("source", []))
        if c.get("cell_type") == "markdown" and src.lstrip().startswith("## 4.13 E4"):
            cells[i]["source"] = _cell_source(MD_413)
            # next code cell
            for j in range(i + 1, min(i + 3, len(cells))):
                if cells[j].get("cell_type") == "code":
                    cells[j]["source"] = _cell_source(CODE_413)
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
                            {
                                "modalidade": "video",
                                "score": scenario["video"]["score"],
                                "peso": fusion["weights"]["video"],
                            },
                            {
                                "modalidade": "audio",
                                "score": scenario["audio"]["score"],
                                "peso": fusion["weights"]["audio"],
                            },
                            {
                                "modalidade": "vitals",
                                "score": scenario["vitals"]["score"],
                                "peso": fusion["weights"]["vitals"],
                            },
                            {
                                "modalidade": "FUSÃO",
                                "score": fusion["risk_score"],
                                "peso": 1.0,
                                "level": fusion["level"],
                            },
                        ]
                    )
                    txt_out = (
                        f"Pesos clínicos: {CLINICAL_WEIGHTS}\n"
                        f"Racional: {WEIGHTS_RATIONALE}\n"
                        + json.dumps(summary, ensure_ascii=False, indent=2, default=str)
                        + "\n"
                    )
                    cells[j]["outputs"] = [stream(txt_out), html_df(table)]
                    cells[j]["execution_count"] = 1
                    break
            break

    # Update §4.14 markdown + outputs
    md_414 = """## 4.14 E5 — Alerta LLM clínico (Ollama + notifier)

Prompt SBAR em linguagem médico-clínica (`src/llm/ollama_report.py`), com perfil do paciente J.S.
pós-AVC e seção educacional **Sobre o AVC** (o que é + sequelas + ligação com os achados).
Se Ollama estiver off, usa **fallback template** clínico. Notificação: `data/processed/alerts/alerta_JS001.md`.
"""
    for i, c in enumerate(cells):
        src = "".join(c.get("source", []))
        if c.get("cell_type") == "markdown" and src.lstrip().startswith("## 4.14 E5"):
            cells[i]["source"] = _cell_source(md_414)
            for j in range(i + 1, min(i + 3, len(cells))):
                if cells[j].get("cell_type") == "code":
                    cells[j]["outputs"] = [
                        stream(f"Alerta salvo em: {alert_path}\n"),
                        md_out(report),
                    ]
                    cells[j]["execution_count"] = 1
                    break
            break

    # §10 — update fusion line if present
    for i, c in enumerate(cells):
        src = "".join(c.get("source", []))
        if src.lstrip().startswith("## 10. Resultados"):
            # rebuild fusion row in table
            lines = src.splitlines(keepends=True)
            new_lines = []
            for ln in lines:
                if ln.startswith("| Fusão + alerta |"):
                    new_lines.append(
                        f"| Fusão + alerta | risco={fusion['risk_score']:.3f} (**{fusion['level']}**); "
                        f"pesos vídeo/áudio/vitais={fusion['weights']['video']:.2f}/"
                        f"{fusion['weights']['audio']:.2f}/{fusion['weights']['vitals']:.2f}; "
                        f"arquivo `{alert_path.name}` | Alerta SBAR clínico (vitais priorizados) |\n"
                    )
                elif ln.startswith("**Recomendação fusionada:**"):
                    new_lines.append(f"**Recomendação fusionada:** {fusion['recomendacao']}\n")
                    new_lines.append(f"\n**Pesos:** {WEIGHTS_RATIONALE}\n")
                elif ln.startswith("**Pesos:**"):
                    continue  # avoid duplicate if re-run
                else:
                    new_lines.append(ln)
            cells[i]["source"] = new_lines
            break

    NB.write_text(json.dumps(nb, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Updated {NB}")
    print(f"Novo risco: {fusion['risk_score']:.4f} ({fusion['level']})")


if __name__ == "__main__":
    main()
