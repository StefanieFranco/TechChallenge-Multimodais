"""Gera cenário MR-001 (baixo risco) e insere §4.15 no Relatorio."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
NB = ROOT / "notebooks" / "Relatorio.ipynb"

MD_415 = """## 4.15 Caso contraste — MR-001 (risco baixo)

Segundo exemplo educacional: paciente **M.R.** em fisioterapia preventiva, com exercício **CORRETO**,
proxy vocal saudável (UCI) e vitais **estáveis** (sem anomalias injetadas).

Objetivo: mostrar que a mesma fusão (pesos 0.25/0.20/0.55) também classifica janelas **seguras**,
em contraste com JS-001 (risco alto).
"""

CODE_415 = """from pathlib import Path
import sys
import json

import pandas as pd
from IPython.display import Markdown, display

ROOT = Path("..").resolve()
if not (ROOT / "src").exists():
    ROOT = Path(".").resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.fusion.risk_fusion import build_js_scenario_scores, build_low_risk_scenario_scores
from src.llm.ollama_report import JS001_PROFILE, MR001_PROFILE, generate_report
from src.alerts.notifier import notify
from src.audio.speech_analysis import speech_risk_from_text

high = build_js_scenario_scores(ROOT)
low = build_low_risk_scenario_scores(ROOT)

cmp = pd.DataFrame([
    {"caso": "JS-001 (alto)", "video": high["video"]["score"], "audio": high["audio"]["score"],
     "vitals": high["vitals"]["score"], "fusao": high["fusion"]["risk_score"], "nivel": high["fusion"]["level"]},
    {"caso": "MR-001 (baixo)", "video": low["video"]["score"], "audio": low["audio"]["score"],
     "vitals": low["vitals"]["score"], "fusao": low["fusion"]["risk_score"], "nivel": low["fusion"]["level"]},
])
display(cmp)

checkin_ok = (
    "Olá, sou a paciente M.R. A sessão de fisioterapia correu bem, sem falta de ar, "
    "sem tontura e tomei os remédios corretamente. Me sinto bem."
)
speech = speech_risk_from_text(checkin_ok)
ctx = {
    "video": low["video"],
    "audio": low["audio"],
    "vitals": low["vitals"],
    "speech": speech,
    "transcript_text": checkin_ok,
    "patient_profile": MR001_PROFILE,
    "contrast_with": {
        "patient_id": JS001_PROFILE["patient_id"],
        "risk_score": high["fusion"]["risk_score"],
        "level": high["fusion"]["level"],
    },
}
report = generate_report(low["fusion"], context=ctx)
path = notify(report, low["fusion"]["level"], payload={"fusion": low["fusion"]}, patient_id="MR001")
print("Alerta/relatório salvo em:", path)
print(json.dumps({"MR001": low["fusion"], "JS001_ref": high["fusion"]["risk_score"]}, ensure_ascii=False, indent=2))
display(Markdown(report))
"""


def _src(text: str) -> list[str]:
    lines = text.strip("\n").split("\n")
    return [ln + "\n" for ln in lines[:-1]] + ([lines[-1] + "\n"] if lines else [])


def stream(text: str) -> dict:
    if not text.endswith("\n"):
        text += "\n"
    return {"output_type": "stream", "name": "stdout", "text": text.splitlines(keepends=True)}


def html_df(df) -> dict:
    return {
        "output_type": "display_data",
        "data": {"text/html": [df.to_html(index=False)], "text/plain": [df.to_string(index=False)]},
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
    from src.fusion.risk_fusion import build_js_scenario_scores, build_low_risk_scenario_scores
    from src.llm.ollama_report import JS001_PROFILE, MR001_PROFILE, generate_report

    print("=== Alto (JS-001) ===")
    high = build_js_scenario_scores(ROOT)
    print(high["fusion"])

    print("=== Baixo (MR-001) ===")
    low = build_low_risk_scenario_scores(ROOT)
    print(low["fusion"])

    checkin_ok = (
        "Olá, sou a paciente M.R. A sessão de fisioterapia correu bem, sem falta de ar, "
        "sem tontura e tomei os remédios corretamente. Me sinto bem."
    )
    (ROOT / "data" / "raw" / "audio" / "mr001_checkin.txt").write_text(checkin_ok, encoding="utf-8")
    speech = speech_risk_from_text(checkin_ok)
    ctx = {
        "video": low["video"],
        "audio": low["audio"],
        "vitals": low["vitals"],
        "speech": speech,
        "transcript_text": checkin_ok,
        "patient_profile": MR001_PROFILE,
        "contrast_with": {
            "patient_id": JS001_PROFILE["patient_id"],
            "risk_score": high["fusion"]["risk_score"],
            "level": high["fusion"]["level"],
        },
    }
    report = generate_report(low["fusion"], context=ctx)
    alert_path = notify(
        report,
        low["fusion"]["level"],
        payload={"fusion": low["fusion"]},
        patient_id="MR001",
    )

    cmp = pd.DataFrame(
        [
            {
                "caso": "JS-001 (alto)",
                "video": high["video"]["score"],
                "audio": high["audio"]["score"],
                "vitals": high["vitals"]["score"],
                "fusao": high["fusion"]["risk_score"],
                "nivel": high["fusion"]["level"],
            },
            {
                "caso": "MR-001 (baixo)",
                "video": low["video"]["score"],
                "audio": low["audio"]["score"],
                "vitals": low["vitals"]["score"],
                "fusao": low["fusion"]["risk_score"],
                "nivel": low["fusion"]["level"],
            },
        ]
    )

    nb = json.loads(NB.read_text(encoding="utf-8"))
    cells = nb["cells"]

    # Remove existing 4.15 if re-run
    start = end = None
    for i, c in enumerate(cells):
        src = "".join(c.get("source", []))
        if start is None and src.lstrip().startswith("## 4.15 Caso contraste"):
            start = i
        if start is not None and src.lstrip().startswith("## 5. Stack e modelos"):
            end = i
            break
    if start is not None and end is not None:
        del cells[start:end]

    insert_at = None
    for i, c in enumerate(cells):
        if "".join(c.get("source", [])).lstrip().startswith("## 5. Stack e modelos"):
            insert_at = i
            break
    if insert_at is None:
        raise RuntimeError("§5 não encontrada")

    out_txt = (
        json.dumps(
            {"MR001": low["fusion"], "JS001_ref": high["fusion"]["risk_score"]},
            ensure_ascii=False,
            indent=2,
        )
        + f"\nAlerta/relatório salvo em: {alert_path}\n"
    )
    code_cell = {
        "cell_type": "code",
        "execution_count": 1,
        "metadata": {},
        "outputs": [
            html_df(cmp),
            stream(out_txt),
            md_out(report),
        ],
        "source": _src(CODE_415),
    }
    cells[insert_at:insert_at] = [
        {"cell_type": "markdown", "metadata": {}, "source": _src(MD_415)},
        code_cell,
    ]

    # Touch §10 with contrast line
    for i, c in enumerate(cells):
        src = "".join(c.get("source", []))
        if src.lstrip().startswith("## 10. Resultados"):
            lines = list(c.get("source", []))
            note = (
                f"\n**Contraste MR-001:** fusão={low['fusion']['risk_score']:.3f} "
                f"(**{low['fusion']['level']}**) vs JS-001={high['fusion']['risk_score']:.3f} "
                f"(**{high['fusion']['level']}**). Ver §4.15 e `alerta_MR001.md`.\n"
            )
            # remove previous contrast note if any
            lines = [ln for ln in lines if not ln.startswith("**Contraste MR-001:**")]
            lines.append(note)
            cells[i]["source"] = lines
            break

    NB.write_text(json.dumps(nb, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Updated {NB}")
    print(cmp.to_string(index=False))
    print(f"MR-001 level={low['fusion']['level']} score={low['fusion']['risk_score']:.4f}")


if __name__ == "__main__":
    main()
