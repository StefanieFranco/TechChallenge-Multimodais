"""Insere §§4.8.1–4.14 (classificador, Whisper, RX, IF-ECG, E4, E5) e atualiza §3/§8–§10."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks" / "Relatorio.ipynb"
DOCS = ROOT / "docs" / "relatorio_tecnico.md"
ROTEIRO = ROOT / "docs" / "roteiro_video_demo.md"


def md(text: str) -> dict:
    lines = text.strip("\n").split("\n")
    source = [ln + "\n" for ln in lines[:-1]] + ([lines[-1] + "\n"] if lines else [])
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def code(text: str) -> dict:
    lines = text.strip("\n").split("\n")
    source = [ln + "\n" for ln in lines[:-1]] + ([lines[-1] + "\n"] if lines else [])
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source,
    }


SECTION3 = """## 3. Requisitos do enunciado × solução adotada

O PDF da Fase 4 cita **Azure Cognitive Services**. Optamos por **stack gratuita local**, com equivalência funcional documentada abaixo (aceitável academicamente desde que justificado).

| Requisito (PDF) | Solução local | Status |
|---|---|---|
| Análise de vídeo (fisio/cirurgia) | MediaPipe Pose + assimetria L/R | Executado (`src/video`) |
| OpenPose / postura | **MediaPipe Pose** (alternativa ao OpenPose) | Executado (assimetria + alertas de forma) |
| YOLOv8 (objetos / áreas) | Ultralytics YOLOv8 | Planejado (MediaPipe cobre postura no MVP) |
| Relatório de desvios no procedimento | `anomaly_report.py` (heurística + overlays) | Executado (MVP) |
| Áudio de consultas | Whisper (STT) | Executado (`src/audio/transcription.py`) |
| Azure Speech to Text | **Whisper** | Equivalente local |
| Azure Text Analytics (termos/sentimento) | Léxico de termos críticos + RF Parkinson | Executado |
| Fadiga / disartria | Features UCI Parkinson + `voice_risk_score` | Executado (tabular) |
| Anomalias em vitais | Isolation Forest (sintético + ECG PhysioNet) | Executado |
| Evolução de prescrições | `prescription_check.py` (alvos vs série) | Executado |
| Alertas à equipe | Fusão + Ollama (prompt clínico) + `notifier.py` | Executado |
| Serviços em nuvem Azure | Stack local + Ollama; LoRA médico no HF (extensão) | Substituído / justificado |

### Tabela de equivalência Azure → local

| Azure (enunciado) | Equivalente neste projeto |
|---|---|
| Speech to Text | Whisper (`openai-whisper`) |
| Text Analytics | Regras de termos críticos + classificador RF UCI |
| Resumo / inteligência | Ollama `llama3.2` com prompt clínico SBAR (+ fallback template) |
"""

CELLS_NEW = [
    md(
        """### 4.8.1 Classificador supervisionado (RandomForest — UCI Parkinson)

Além do `voice_risk_score` heurístico (usado na fusão), treinamos um **RandomForest** saudável vs PD
para evidenciar métricas de teste no Relatorio (`src/audio/parkinson_classifier.py`).

> Educacional — não é diagnóstico clínico de Parkinson."""
    ),
    code(
        """from pathlib import Path
import sys
import json

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display

ROOT = Path("..").resolve()
if not (ROOT / "src").exists():
    ROOT = Path(".").resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.audio.parkinson_classifier import run_parkinson_classifier

sns.set_theme(style="whitegrid", context="notebook")
rf = run_parkinson_classifier()
print("Métricas teste:", rf["metrics"])
print("Modelo:", rf["model_path"])
display(pd.DataFrame([rf["metrics"]]))
print(json.dumps(rf["classification_report"], ensure_ascii=False, indent=2)[:1200])

cm = rf["confusion_matrix"]
fig, ax = plt.subplots(figsize=(4, 3.5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["saudavel", "PD"], yticklabels=["saudavel", "PD"], ax=ax)
ax.set_xlabel("Predito"); ax.set_ylabel("Real")
ax.set_title("Matriz de confusão — RF Parkinson (teste)")
plt.tight_layout(); plt.show()
"""
    ),
    md(
        """## 4.10 Whisper + termos críticos (check-in J.S.)

Áudio: `data/raw/audio/js001_checkin.wav` (TTS). STT com Whisper; Text Analytics local via léxico
em `src/audio/speech_analysis.py` (falta de ar, tontura, remédio, dor no peito, etc.)."""
    ),
    code(
        """from pathlib import Path
import sys

from IPython.display import Markdown, display

ROOT = Path("..").resolve()
if not (ROOT / "src").exists():
    ROOT = Path(".").resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.audio.transcription import transcribe
from src.audio.speech_analysis import analyze_speech, speech_risk_from_text

wav = ROOT / "data" / "raw" / "audio" / "js001_checkin.wav"
txt_ref = ROOT / "data" / "raw" / "audio" / "js001_checkin.txt"
try:
    tr = transcribe(wav, model_size="base")
    speech = analyze_speech(wav, transcript=tr)
except Exception as exc:
    print("Whisper indisponível:", type(exc).__name__, exc)
    text = txt_ref.read_text(encoding="utf-8") if txt_ref.exists() else ""
    tr = {"text": text, "segments": [], "language": "pt", "model_size": "fallback_text"}
    speech = speech_risk_from_text(text)
    speech["aviso"] = "Fallback: texto de referência (Whisper falhou)."

print("Transcrição:", tr.get("text"))
print("Termos críticos:", speech.get("hits"))
print("speech_risk:", speech.get("score"))
display(Markdown(f"**Termos:** {', '.join(speech.get('hits') or []) or '(nenhum)'}"))
"""
    ),
    md(
        """## 4.11 Prescrição JS-001

Checagem educacional de alvos (SpO₂, FC, PA) vs série sintética da noite
(`src/vitals/prescription_check.py`)."""
    ),
    code(
        """from pathlib import Path
import sys
import json

from IPython.display import display
import pandas as pd

ROOT = Path("..").resolve()
if not (ROOT / "src").exists():
    ROOT = Path(".").resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.vitals.synthetic_vitals import load_or_create_synthetic
from src.vitals.prescription_check import check_prescription, DEFAULT_JS001_PRESCRIPTION

df, _ = load_or_create_synthetic(path=ROOT / "data" / "raw" / "vitals" / "synthetic" / "js001_noite.csv")
rx = check_prescription(df, DEFAULT_JS001_PRESCRIPTION)
print(json.dumps({k: rx[k] for k in ("patient_id", "targets", "score", "counts", "achados")}, ensure_ascii=False, indent=2))
display(pd.DataFrame(rx["medications"]))
"""
    ),
    md(
        """## 4.12 Isolation Forest no ECG real (`arrhythmia_train.parquet`)

Treino hold-out mitdb+nsrdb → métricas vs rótulo `abnormal`; stress test no corpus
**ECG Fragment High-Risk** (`src/vitals/ecg_anomaly.py`)."""
    ),
    code(
        """from pathlib import Path
import sys
import json

import pandas as pd
from IPython.display import display

ROOT = Path("..").resolve()
if not (ROOT / "src").exists():
    ROOT = Path(".").resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.vitals.ecg_anomaly import run_ecg_if_pipeline

ecg = run_ecg_if_pipeline()
print("metrics_train:", ecg["metrics_train"])
print("metrics_test :", ecg["metrics_test"])
print("high_risk    :", ecg["high_risk"])
print("model_path   :", ecg["model_path"])
display(pd.DataFrame([
    {"split": "train", **ecg["metrics_train"]},
    {"split": "test", **ecg["metrics_test"]},
]))
"""
    ),
    md(
        """## 4.13 E4 — Fusão multimodal dos 3 scores

`src/fusion/risk_fusion.py` combina vídeo (clip incorreto + alertas), áudio (`voice_risk_score` PD)
e vitais (IF sintético JS-001) em risco global + breakdown."""
    ),
    code(
        """from pathlib import Path
import sys
import json

import pandas as pd
from IPython.display import display

ROOT = Path("..").resolve()
if not (ROOT / "src").exists():
    ROOT = Path(".").resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.fusion.risk_fusion import build_js_scenario_scores

scenario = build_js_scenario_scores(ROOT)
fusion = scenario["fusion"]
print(json.dumps({
    "patient_id": scenario["patient_id"],
    "video": {k: scenario["video"][k] for k in ("score", "veredito", "form_alerts", "n_alert_frames")},
    "audio": scenario["audio"],
    "vitals": {k: scenario["vitals"][k] for k in ("score", "n_anomalies")},
    "fusion": fusion,
}, ensure_ascii=False, indent=2))
display(pd.DataFrame([
    {"modalidade": "video", "score": scenario["video"]["score"]},
    {"modalidade": "audio", "score": scenario["audio"]["score"]},
    {"modalidade": "vitals", "score": scenario["vitals"]["score"]},
    {"modalidade": "FUSÃO", "score": fusion["risk_score"], "level": fusion["level"]},
]))
"""
    ),
    md(
        """## 4.14 E5 — Alerta LLM clínico (Ollama + notifier)

Prompt SBAR em linguagem médico-clínica (`src/llm/ollama_report.py`). Se Ollama estiver off,
usa **fallback template** clínico. Notificação: `data/processed/alerts/alerta_JS001.md`."""
    ),
    code(
        """from pathlib import Path
import sys

from IPython.display import Markdown, display

ROOT = Path("..").resolve()
if not (ROOT / "src").exists():
    ROOT = Path(".").resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.fusion.risk_fusion import build_js_scenario_scores
from src.llm.ollama_report import generate_report
from src.alerts.notifier import notify
from src.vitals.synthetic_vitals import load_or_create_synthetic
from src.vitals.prescription_check import check_prescription
from src.audio.speech_analysis import speech_risk_from_text

scenario = build_js_scenario_scores(ROOT)
df, _ = load_or_create_synthetic(path=ROOT / "data" / "raw" / "vitals" / "synthetic" / "js001_noite.csv")
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
}
report = generate_report(scenario["fusion"], context=context)
path = notify(report, scenario["fusion"]["level"], payload={"fusion": scenario["fusion"]}, patient_id="JS001")
print("Alerta salvo em:", path)
display(Markdown(report))
"""
    ),
]

SECTION8_MD = """## 8. Próximos experimentos (roadmap)

Experimentos principais do MVP (**E1–E5**) concluídos no Relatorio (§4.7–§4.14).

| Notebook / seção | Conteúdo |
|---|---|
| `01_vitals_sinteticos.ipynb` / §4.9 | Vitais sintéticos + Isolation Forest |
| §4.8 / §4.8.1 | Parkinson heurístico + RF supervisionado |
| §4.7 | Pose MediaPipe + alertas de forma |
| §4.10–4.14 | Whisper, prescrição, IF-ECG, fusão E4, alerta E5 |

### Extensões futuras (fora do MVP)
- YOLOv8 para objetos/áreas
- Adapter LoRA médico HF carregado em inferência local (download: `python -m src.fine_tuning.download_local_llama`)
- Gravacão e publicação do vídeo demo (roteiro em [`docs/roteiro_video_demo.md`](../docs/roteiro_video_demo.md))
"""

SECTION8_CODE = """# Checklist de experimentos
experimentos = [
    {"id": "E1", "nome": "Vitais sintéticos + Isolation Forest", "feito": True},
    {"id": "E2", "nome": "Features vocais Parkinson (UCI)", "feito": True},
    {"id": "E3", "nome": "Pose MediaPipe em vídeo próprio", "feito": True},
    {"id": "E4", "nome": "Fusão dos 3 scores", "feito": True},
    {"id": "E5", "nome": "Alerta LLM (Ollama / prompt clínico)", "feito": True},
]

for e in experimentos:
    mark = "[x]" if e["feito"] else "[ ]"
    print(f"{mark} {e['id']} — {e['nome']}")
"""

SECTION9 = """## 9. Checklist de entrega (itens fáceis de esquecer)

Conforme o PDF da Fase 4 (atividade obrigatória, ~90% da nota das disciplinas da fase):

### Repositório Git
- [x] Código-fonte completo da solução
- [x] README com setup e arquitetura
- [x] Dados de exemplo (ou script de geração) em `data/`
- [x] Este relatório técnico evoluído com **resultados e exemplos de anomalias**

### Relatório técnico (conteúdo mínimo)
- [x] Descrição do **fluxo multimodal**
- [x] **Modelos** aplicados em cada tipo de dado
- [x] **Resultados** obtidos e exemplos de anomalias detectadas (§10)
- [x] Justificativa da substituição Azure → stack local (§3)
- [x] Limitações (vídeo simulado, dados sintéticos, aviso ético)

### Vídeo demo (≤ 15 min · YouTube/Vimeo)
- [ ] Análise prática de **áudio e vídeo** — *gravar* (roteiro: [`docs/roteiro_video_demo.md`](../docs/roteiro_video_demo.md))
- [ ] Detecção e resposta a **anomalias**
- [ ] Explicação da “integração cloud” **via equivalência local**
- [ ] Fluxo final do **alerta à equipe médica**

### Detalhes técnicos frequentemente esquecidos
- [x] Modalidade **texto** (prescrição / evolução clínica) — §4.11
- [x] Padrões de **movimentação** na internação (vídeo + features) — §4.7
- [x] Definição operacional de **tempo real** (janelas) — pipeline §7
- [x] Aviso de que o LLM médico é educacional — §4.14
- [x] Evidências (prints/gráficos) dos três scores e do score fusionado — §4.13–§4.14
"""

SECTION10_PLACEHOLDER = """## 10. Resultados

_Preenchido automaticamente pelo runner `scripts/run_fusion_pipeline_into_relatorio.py` após E1–E5._

| Modalidade | Métrica / evidência | Exemplo de anomalia |
|---|---|---|
| Vídeo | ver §4.7 / §4.13 | exercício INCORRETO + alertas de forma |
| Áudio | RF + voice_risk + Whisper §4.8–4.10 | termos críticos no check-in |
| Vitais | IF sintético + IF-ECG §4.9 / §4.12 | SpO₂/HR anômalos; janelas abnormal |
| Fusão + alerta | §4.13–§4.14 | risco global + SBAR clínico |
"""

ROTEIRO_MD = """# Roteiro do vídeo demo (≤ 15 min)

Publicar em YouTube/Vimeo após gravar. Siga a ordem abaixo.

## 0. Setup (30 s)
- Abrir `notebooks/Relatorio.ipynb` e pasta `data/processed/alerts/`.
- Mencionar: stack **local gratuita** (sem Azure pago).

## 1. Problema e equivalência Azure → local (2 min)
- Objetivo: monitoramento multimodal J.S. (pós-AVC).
- Tabela §3: Whisper ≈ Speech; léxico/RF ≈ Text Analytics; Ollama ≈ resumo cognitivo.

## 2. Vídeo / fisioterapia (3 min)
- Mostrar clip CORRETO vs INCORRETO e overlays MediaPipe (§4.7).
- Destacar alertas (joelho / braço) e veredito GT.

## 3. Áudio (3 min)
- UCI Parkinson + RF (§4.8 / 4.8.1) — métricas de teste.
- Check-in WAV + Whisper + termos críticos (§4.10).

## 4. Vitais + prescrição (3 min)
- Isolation Forest sintético (§4.9) e ECG PhysioNet (§4.12).
- Prescrição JS-001 fora de alvo (§4.11).

## 5. Fusão + alerta à equipe (3 min)
- Tabela dos 3 scores + risco fusionado (§4.13).
- Ler trecho do alerta SBAR em `alerta_JS001.md` (§4.14).
- Reforçar disclaimer educacional.

## 6. Encerramento (1 min)
- Limitações (dados sintéticos / proxy).
- Link do repositório e do Relatorio.
"""


def _replace_section(nb: dict, start_prefix: str, new_cells: list[dict], end_prefixes: list[str]) -> None:
    cells = nb["cells"]
    start = None
    end = None
    for i, c in enumerate(cells):
        src = "".join(c.get("source", []))
        if start is None and src.lstrip().startswith(start_prefix):
            start = i
            continue
        if start is not None and end is None:
            for p in end_prefixes:
                if src.lstrip().startswith(p):
                    end = i
                    break
    if start is None:
        raise RuntimeError(f"Seção não encontrada: {start_prefix}")
    if end is None:
        end = start + 1
    nb["cells"][start:end] = new_cells


def main() -> None:
    nb = json.loads(NB.read_text(encoding="utf-8"))

    # §3
    for i, c in enumerate(nb["cells"]):
        src = "".join(c.get("source", []))
        if src.lstrip().startswith("## 3. Requisitos do enunciado"):
            nb["cells"][i] = md(SECTION3)
            break

    # Remover bloco novo se reexecutar (marcador)
    marker = "## 4.10 Whisper + termos críticos"
    cells = nb["cells"]
    # Remove from 4.8.1 or 4.10 until ## 5
    start_rm = None
    end_rm = None
    for i, c in enumerate(cells):
        src = "".join(c.get("source", []))
        if start_rm is None and (
            src.lstrip().startswith("### 4.8.1")
            or src.lstrip().startswith(marker)
            or src.lstrip().startswith("## 4.10 Whisper")
        ):
            start_rm = i
        if start_rm is not None and src.lstrip().startswith("## 5. Stack e modelos"):
            end_rm = i
            break
    if start_rm is not None and end_rm is not None:
        del cells[start_rm:end_rm]

    # Inserir antes da §5
    insert_at = None
    for i, c in enumerate(nb["cells"]):
        src = "".join(c.get("source", []))
        if src.lstrip().startswith("## 5. Stack e modelos"):
            insert_at = i
            break
    if insert_at is None:
        raise RuntimeError("§5 não encontrada")
    nb["cells"][insert_at:insert_at] = CELLS_NEW

    # §8
    _replace_section(
        nb,
        "## 8. Próximos experimentos",
        [md(SECTION8_MD), code(SECTION8_CODE)],
        ["## 9. Checklist"],
    )

    # §9
    _replace_section(
        nb,
        "## 9. Checklist de entrega",
        [md(SECTION9)],
        ["## 10. Resultados"],
    )

    # §10
    for i, c in enumerate(nb["cells"]):
        src = "".join(c.get("source", []))
        if src.lstrip().startswith("## 10. Resultados"):
            nb["cells"][i] = md(SECTION10_PLACEHOLDER)
            break

    # Atualizar checklist E4/E5 se ainda houver célula antiga residual
    for c in nb["cells"]:
        src = "".join(c.get("source", []))
        if '"id": "E4"' in src and "feito\": False" in src:
            c["source"] = [ln + "\n" for ln in SECTION8_CODE.strip("\n").split("\n")[:-1]] + [
                SECTION8_CODE.strip("\n").split("\n")[-1] + "\n"
            ]

    NB.write_text(json.dumps(nb, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Updated {NB}")

    ROTEIRO.write_text(ROTEIRO_MD, encoding="utf-8")
    print(f"Wrote {ROTEIRO}")

    docs = DOCS.read_text(encoding="utf-8")
    docs = docs.replace(
        "| Vitais / texto | `src/vitals` | Isolation Forest v1 (sintético); prescrição planejada |",
        "| Vitais / texto | `src/vitals` | IF sintético + IF-ECG; `prescription_check` |",
    )
    docs = docs.replace(
        "| Áudio | `src/audio` | Features UCI Parkinson; Whisper (próximo) |",
        "| Áudio | `src/audio` | Parkinson + RF; Whisper STT + termos críticos |",
    )
    docs = docs.replace(
        "| Fusão | `src/fusion` | Score ponderado dos 3 riscos |",
        "| Fusão | `src/fusion` | `fuse_risk_scores` / cenário J.S. (E4) |",
    )
    docs = docs.replace(
        "| LLM / alertas | `src/llm`, `src/alerts` | Ollama + adapter médico |",
        "| LLM / alertas | `src/llm`, `src/alerts` | Ollama SBAR clínico + notifier Markdown |",
    )
    results_block = """
## 7. Resultados

Ver tabela completa em `notebooks/Relatorio.ipynb` §10 (preenchida pelo pipeline E1–E5):
fusão multimodal, métricas RF Parkinson, IF-ECG, Whisper/termos críticos e alerta SBAR em
`data/processed/alerts/alerta_JS001.md`.

## 8. Limitações e próximos passos

- Vídeo e parte dos áudios são simulados/próprios; UCI Parkinson é proxy tabular.
- Azure substituído por equivalentes locais (justificar no vídeo de entrega).
- Extensões: YOLOv8; carregar LoRA médico HF; publicar vídeo demo (`docs/roteiro_video_demo.md`).
"""
    if "## 7. Resultados" in docs:
        # replace from ## 7 to end
        idx = docs.find("## 7. Resultados")
        docs = docs[:idx] + results_block.lstrip("\n")
    else:
        docs += "\n" + results_block
    DOCS.write_text(docs, encoding="utf-8")
    print(f"Updated {DOCS}")


if __name__ == "__main__":
    main()
