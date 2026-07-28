"""Insere §4.8 análise Parkinson (áudio tabular) no Relatorio.ipynb e atualiza docs."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks" / "Relatorio.ipynb"
DOCS = ROOT / "docs" / "relatorio_tecnico.md"


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


INTRO = """## 4.8 Análise de áudio — features vocais (UCI Parkinson)

Fonte: **Oxford Parkinson's Disease Detection Dataset** (UCI), em `data/raw/parkinsons/parkinsons.data`.
São **medidas biomédicas de voz já extraídas** (jitter, shimmer, HNR, PPE, etc.) — **não** arquivos WAV.
Complemento: Telemonitoring UPDRS em `data/raw/parkinsons/telemonitoring/`.

**Papel no projeto:** proxy educacional de alteração vocal (análogo a fadiga/disartria no check-in do paciente J.S.). Whisper/STT permanece planejado para clips `.wav` futuros.

**Score de risco vocal** ∈ [0, 1]: média min-max de `PPE`, `spread1`, `MDVP:Jitter(%)`, `MDVP:Shimmer` e `(1 − HNR_norm)`.

> **Aviso:** heurística acadêmica — **não** constitui diagnóstico clínico.
"""

LOAD_CODE = """from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from IPython.display import display

ROOT = Path("..").resolve()
if not (ROOT / "src").exists():
    ROOT = Path(".").resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.audio.parkinsons_analysis import analyze_parkinsons_dir

sns.set_theme(style="whitegrid", context="notebook")

AUDIO = ROOT / "data" / "raw" / "parkinsons"
result = analyze_parkinsons_dir(AUDIO)
df = result["df"]
tele = result["tele"]

print(f"Pasta: {result['root']}")
print(f"Detection: {result['detection_path']}")
print(f"Telemonitoring: {result['telemonitoring_path']}")
print("-" * 60)
print("Resumo:", result["summary"])
if result["tele_summary"]:
    print("Telemonitoring:", result["tele_summary"])
print("-" * 60)
display(df[["name", "subject", "label", "status", "MDVP:Jitter(%)", "MDVP:Shimmer", "HNR", "PPE", "voice_risk_score"]].head(10))
"""

MD_LOAD = """**Inventário:** 195 gravações / ~32 sujeitos; classe `PD` é majoritária. A coluna `voice_risk_score` já resume a heurística por registro para alimentar a fusão multimodal.
"""

PLOT1 = """fig, ax = plt.subplots(figsize=(6, 4))
order = ["saudavel", "PD"]
counts = df["label"].value_counts().reindex(order)
counts.plot(kind="bar", color=["#457b9d", "#e63946"], ax=ax)
ax.set_title("Contagem de gravações por classe (status)")
ax.set_xlabel("Classe")
ax.set_ylabel("N gravações")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()
print(counts.to_dict())
"""

MD1 = """**Gráfico 1 — classes.** Distribuição saudável vs PD no corpus de detecção. O desbalanceamento é esperado (mais amostras PD) e deve ser considerado ao interpretar acurácia futura.
"""

PLOT2 = """fig, axes = plt.subplots(2, 2, figsize=(10, 8))
feats = ["MDVP:Jitter(%)", "MDVP:Shimmer", "HNR", "PPE"]
for ax, feat in zip(axes.ravel(), feats):
    sns.boxplot(data=df, x="label", y=feat, order=["saudavel", "PD"], ax=ax)
    ax.set_title(feat)
plt.suptitle("Features vocais por classe", y=1.02)
plt.tight_layout()
plt.show()
"""

MD2 = """**Gráfico 2 — boxplots.** PD tende a jitter/shimmer/PPE mais altos e HNR mais baixo — padrão clássico de maior instabilidade fonatória, útil como proxy de risco de fala no MVP.
"""

PLOT3 = """fig, ax = plt.subplots(figsize=(7, 4))
sns.histplot(data=df, x="PPE", hue="label", hue_order=["saudavel", "PD"], kde=True, element="step", ax=ax)
ax.set_title("Distribuição de PPE por classe")
plt.tight_layout()
plt.show()
"""

MD3 = """**Gráfico 3 — PPE.** A pitch period entropy separa bem os grupos; valores altos associam-se a maior risco na heurística `score_voice_risk`.
"""

PLOT4 = """corr_cols = result["corr_features"]
corr = df[corr_cols].corr()
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax)
ax.set_title("Correlação — features vocais + status")
plt.tight_layout()
plt.show()
"""

MD4 = """**Gráfico 4 — heatmap.** `status` correlaciona positivamente com PPE/jitter/shimmer e negativamente com HNR, sustentando o peso dessas variáveis no score.
"""

PLOT5 = """if tele is not None and "total_UPDRS" in tele.columns:
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(data=tele, y="total_UPDRS", color="#a8dadc", ax=ax)
    ax.set_title("Telemonitoring — distribuição total_UPDRS")
    plt.tight_layout()
    plt.show()
    print(tele["total_UPDRS"].describe().round(2))
else:
    print("Telemonitoring indisponível.")
"""

MD5 = """**Gráfico 5 — UPDRS (telemonitoring).** Severidade clínica agregada no corpus longitudinal; complementar à classificação binária do detection set (não misturado no score do MVP).
"""

TOP_CODE = """print("Top sujeitos por mediana do voice_risk_score:")
display(result["top_subjects"])
print("-" * 60)
print(result["aviso"])
"""

MD_TOP = """**Leitura multimodal:** o `voice_risk_score` (mediana por sujeito ou por gravação) pode entrar na fusão (`src/fusion/risk_fusion.py`) como `audio_score`, junto com o score de assimetria de vídeo e anomalias de vitais.
"""


def patch_status_markdown(text: str) -> str:
    reps = [
        (
            "| Fadiga / disartria | Features acústicas (`speech_analysis.py`) | Planejado |",
            "| Fadiga / disartria | Features UCI Parkinson (`parkinsons_analysis.py`) | Executado (tabular) |",
        ),
        (
            "| Áudio | Amostra curta própria ou clip público pequeno | Coswara / Saarbrücken / Parkinson (PhysioNet, UCI) |",
            "| Áudio | **UCI Parkinson** (features em `data/raw/parkinsons/`) | Coswara / Saarbrücken / Whisper em WAV |",
        ),
    ]
    for old, new in reps:
        text = text.replace(old, new)
    return text


def patch_checklist(text: str) -> str:
    return text.replace(
        '{"id": "E2", "nome": "Whisper + análise de fala", "feito": False}',
        '{"id": "E2", "nome": "Features vocais Parkinson (UCI)", "feito": True}',
    )


def main() -> None:
    nb = json.loads(NB.read_text(encoding="utf-8"))

    for cell in nb["cells"]:
        src = "".join(cell.get("source", []))
        if cell["cell_type"] == "markdown" and "Requisitos do enunciado" in src:
            cell["source"] = [patch_status_markdown(src)]
        if cell["cell_type"] == "markdown" and "Estratégia do MVP" in src:
            cell["source"] = [patch_status_markdown(src)]
        if cell["cell_type"] == "code" and "Whisper + análise de fala" in src:
            cell["source"] = [patch_checklist(src)]

    already = any(
        "4.8 Análise de áudio" in "".join(c.get("source", [])) for c in nb["cells"]
    )
    if not already:
        insert_at = None
        for i, cell in enumerate(nb["cells"]):
            src = "".join(cell.get("source", []))
            if src.lstrip().startswith("## 5. Stack e modelos"):
                insert_at = i
                break
        if insert_at is None:
            raise RuntimeError("Seção 5 não encontrada")
        new_cells = [
            md(INTRO),
            code(LOAD_CODE),
            md(MD_LOAD),
            code(PLOT1),
            md(MD1),
            code(PLOT2),
            md(MD2),
            code(PLOT3),
            md(MD3),
            code(PLOT4),
            md(MD4),
            code(PLOT5),
            md(MD5),
            code(TOP_CODE),
            md(MD_TOP),
        ]
        nb["cells"][insert_at:insert_at] = new_cells

    NB.write_text(json.dumps(nb, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Updated {NB}")

    docs = DOCS.read_text(encoding="utf-8")
    old = (
        "| Áudio | Clip curto próprio / amostra pública | Coswara, Parkinson (PhysioNet/UCI) | `data/raw/audio/` |"
    )
    new = (
        "| Áudio | **UCI Parkinson** (features tabulares) | Coswara / WAV + Whisper | `data/raw/parkinsons/` |"
    )
    if old in docs:
        docs = docs.replace(old, new)
    note = (
        "\n### Áudio — features vocais Parkinson\n\n"
        "Corpus UCI Oxford em `data/raw/parkinsons/` (`parkinsons.data` + telemonitoring UPDRS). "
        "Análise e `voice_risk_score` em `src/audio/parkinsons_analysis.py` "
        "(ver `notebooks/Relatorio.ipynb` §4.8). Whisper/STT permanece para clips `.wav`.\n"
    )
    if "### Áudio — features vocais Parkinson" not in docs:
        anchor = "### Vídeo — fisioterapia (assimetria)"
        if anchor in docs:
            # insert before video section or after it
            idx = docs.find(anchor)
            # after video block (until next ##)
            next_h = docs.find("\n## ", idx + 1)
            if next_h != -1:
                docs = docs[:next_h] + note + docs[next_h:]
            else:
                docs += note
        else:
            docs += note
    DOCS.write_text(docs, encoding="utf-8")
    print(f"Updated {DOCS}")


if __name__ == "__main__":
    main()
