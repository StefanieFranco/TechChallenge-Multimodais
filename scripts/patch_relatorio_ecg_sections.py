"""Insere seções 4.1–4.6 de ECG no Relatorio.ipynb."""

from __future__ import annotations

import json
from pathlib import Path


def code_cell(text: str) -> dict:
    lines = text.splitlines(keepends=True)
    if lines and not lines[-1].endswith("\n"):
        lines[-1] += "\n"
    if not lines:
        lines = ["\n"]
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": lines,
    }


def md_cell(text: str) -> dict:
    lines = text.splitlines(keepends=True)
    if lines and not lines[-1].endswith("\n"):
        lines[-1] += "\n"
    return {"cell_type": "markdown", "metadata": {}, "source": lines}


def main() -> None:
    nb_path = Path("notebooks/Relatorio.ipynb")
    nb = json.loads(nb_path.read_text(encoding="utf-8"))

    # Evita duplicar se o script rodar de novo
    already = any(
        "4.1 Fontes PhysioNet de ECG" in "".join(c.get("source", []))
        for c in nb["cells"]
    )
    if already:
        print("Seções 4.1+ já existem — abortando inserção.")
        return

    new_cells = [
        md_cell(
            """### 4.1 Fontes PhysioNet de ECG (vitais)

Para o bloco de **sinais vitais / anomalias**, usamos três bases PhysioNet sob `data/raw/vitals/`:

| Pasta | Dataset | Papel |
|---|---|---|
| `mitdb/` | [MIT-BIH Arrhythmia](https://physionet.org/content/mitdb/1.0.0/) | Treino (normal + arritmia anotada) |
| `nsrdb/` | [MIT-BIH Normal Sinus Rhythm](https://physionet.org/content/nsrdb/1.0.0/) | Treino (baseline de ritmo normal) |
| `ecg-fragment-high-risk/` | [ECG Fragment High-Risk](https://physionet.org/content/ecg-fragment-high-risk-label/1.0.0/) | Experimento extra de **sensibilidade** a eventos críticos |

**União de treino:** apenas `mitdb` + `nsrdb` → `data/processed/vitals/arrhythmia_train.parquet`.  
O dataset de fragmentos de alto risco **não** entra nessa união (fica reservado para teste de sensibilidade).
"""
        ),
        md_cell(
            """### 4.2 Download dos datasets (`wfdb`)

Equivalente Windows-friendly aos comandos `wget -r -N -c -np` do PhysioNet. O download é **idempotente**: se já existirem arquivos `.hea`, a célula não baixa de novo.
"""
        ),
        code_cell(
            """import sys
from pathlib import Path

import pandas as pd

ROOT = Path("..").resolve()
if not (ROOT / "requirements.txt").exists():
    ROOT = Path(".").resolve()
sys.path.insert(0, str(ROOT))

from src.vitals.ecg_preprocess import (
    DATASET_SPECS,
    download_all_vitals_ecg,
    inventory,
    vitals_raw_dir,
    vitals_processed_dir,
)

RAW = vitals_raw_dir(ROOT)
PROC = vitals_processed_dir(ROOT)
print("RAW :", RAW)
print("PROC:", PROC)
display(pd.DataFrame([
    {"key": k, "db": v["db_name"], "role": v["role"], "url": v["url"]}
    for k, v in DATASET_SPECS.items()
]))
"""
        ),
        code_cell(
            """# Download: mitdb + nsrdb (treino) + ecg-fragment-high-risk (sensibilidade)
# Equivale a:
#   wfdb.dl_database("mitdb", dl_dir="data/raw/vitals/mitdb")
#   wfdb.dl_database("nsrdb", dl_dir="data/raw/vitals/nsrdb")
#   wfdb.dl_database("ecg-fragment-high-risk-label", dl_dir="data/raw/vitals/ecg-fragment-high-risk")

paths = download_all_vitals_ecg(raw_root=RAW, force=False)
inv = inventory(RAW)
display(inv)
"""
        ),
        md_cell(
            """**O que acabou de acontecer:** os três corpora ECG foram materializados em subpastas de `data/raw/vitals/`. A coluna `ready=True` indica presença de headers `.hea`. Se o fragment high-risk falhar no `wfdb`, use o fallback `wget` documentado no erro (URL em `DATASET_SPECS`).
"""
        ),
        md_cell(
            """### 4.3 Limpeza dos dados

Regras aplicadas em `src/vitals/ecg_preprocess.py`:

1. Ler canal 0 (ou primeiro disponível) via WFDB  
2. Descartar records ilegíveis / sinal quase constante / excesso de NaN  
3. Interpolar buracos curtos e clip de amplitude (percentis 0,5–99,5)  
4. **mitdb:** rótulo por janela com anotações `.atr` (`N/L/R/e/j` → normal; demais batimentos → abnormal)  
5. **nsrdb:** rótulo `normal` (ritmo sinusal) + R-peaks simples para HR/RR  
6. Janelas de **10 s**, até 20 por record (NSRDB truncado para caber em memória)
"""
        ),
        code_cell(
            """from src.vitals.ecg_preprocess import list_records, count_hea_files

print("mitdb records :", len(list_records(RAW / "mitdb")), "| .hea =", count_hea_files(RAW / "mitdb"))
print("nsrdb records :", len(list_records(RAW / "nsrdb")), "| .hea =", count_hea_files(RAW / "nsrdb"))
print("high-risk .hea:", count_hea_files(RAW / "ecg-fragment-high-risk"))
"""
        ),
        md_cell(
            """### 4.4 Unificação mitdb + nsrdb (conjunto de treino)

Geramos um único parquet com features por janela (`hr_mean`, `hr_std`, `rr_*`, estatísticas do sinal, `label`, `source`). O high-risk **fica de fora**.
"""
        ),
        code_cell(
            """from src.vitals.ecg_preprocess import build_unified_training_frame, save_unified_training

df_train, meta = build_unified_training_frame(
    raw_root=RAW,
    window_sec=10.0,
    max_windows_per_record=20,
    max_samples_per_record=650_000,
)
parquet_path, meta_path = save_unified_training(df_train, meta, processed_dir=PROC)

print("Salvo:", parquet_path)
print("Meta :", meta_path)
print("Linhas:", len(df_train))
display(df_train.head())
display(pd.Series(meta.get("label_counts", {}), name="label_counts"))
display(pd.Series(meta.get("source_counts", {}), name="source_counts"))
"""
        ),
        md_cell(
            """**Unificação:** cada linha do parquet é uma janela de 10 s proveniente de `mitdb` ou `nsrdb`, com rótulo binário e features de frequência cardíaca derivadas de picos/anotações. Esse arquivo (`arrhythmia_train.parquet`) será a base do Isolation Forest / PyOD nas próximas etapas — sem misturar o corpus de fragmentos críticos.
"""
        ),
        md_cell(
            """### 4.5 Análise Exploratória de Dados (EDA)

Abaixo, gráficos do conjunto **unificado de treino**, cada um seguido de interpretação em markdown.
"""
        ),
        code_cell(
            """import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", context="notebook")
df = df_train.copy()
print(df.shape)
df.describe(include="all").T.head(20)
"""
        ),
        md_cell(
            """A tabela `describe` resume dispersão das features numéricas (HR/RR e estatísticas do sinal). Valores NaN em HR/RR ocorrem quando a janela não tem picos suficientes após o filtro fisiológico de RR (0,3–2,0 s).
"""
        ),
        code_cell(
            """fig, ax = plt.subplots(figsize=(6, 4))
df.groupby("source")["record_id"].nunique().plot(kind="bar", color=["#2a9d8f", "#264653"], ax=ax)
ax.set_title("Records únicos por fonte (treino)")
ax.set_xlabel("Fonte")
ax.set_ylabel("Nº de records")
plt.tight_layout()
plt.show()
"""
        ),
        md_cell(
            """**Gráfico 1 — records por fonte.** Mostra quantos exames distintos entraram de `mitdb` (arritmia) e `nsrdb` (sinusal). Desbalanceamento de records é esperado: o Arrhythmia tem mais arquivos curtos; o NSRDB tem poucos sujeitos, porém registros longos (por isso limitamos janelas/amostras).
"""
        ),
        code_cell(
            """fig, ax = plt.subplots(figsize=(6, 4))
df["label"].value_counts().plot(kind="bar", color=["#457b9d", "#e63946"], ax=ax)
ax.set_title("Distribuição de classes (janelas)")
ax.set_xlabel("Label")
ax.set_ylabel("Nº de janelas")
plt.tight_layout()
plt.show()
print(df["label"].value_counts(normalize=True).round(3))
"""
        ),
        md_cell(
            """**Gráfico 2 — classes.** Contagem de janelas `normal` vs `abnormal` após a união. A classe normal tende a dominar (NSRDB inteiro + batimentos N do MIT-BIH). Isso importa para o detector de anomalias: avaliaremos com foco em recall da classe anormal / score de outlier.
"""
        ),
        code_cell(
            """fig, ax = plt.subplots(figsize=(7, 4))
sns.histplot(data=df.dropna(subset=["hr_mean"]), x="hr_mean", hue="label", bins=30, element="step", ax=ax)
ax.set_title("Distribuição de HR média (bpm) por classe")
ax.set_xlabel("hr_mean (bpm)")
plt.tight_layout()
plt.show()
"""
        ),
        md_cell(
            """**Gráfico 3 — histograma de HR.** Compara a frequência cardíaca média das janelas. Janelas anormais podem concentrar HR mais dispersa ou extremos; janelas normais tendem a se agrupar em faixas fisiológicas típicas de repouso/ambulatorial.
"""
        ),
        code_cell(
            """from src.vitals.ecg_preprocess import load_waveform_snippet

normal_row = df[df["label"] == "normal"].iloc[0]
abn = df[df["label"] == "abnormal"]
abnormal_row = abn.iloc[0] if len(abn) else None

snip_n = load_waveform_snippet(
    RAW / normal_row["source"],
    normal_row["record_id"],
    start=int(normal_row["start_sample"]),
    n_samples=int(normal_row["fs"] * 5),
)

fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=False)
if snip_n:
    sig_n, fs_n = snip_n
    axes[0].plot(sig_n, color="#2a9d8f", lw=0.8)
    axes[0].set_title(f"Trecho NORMAL — {normal_row['source']}/{normal_row['record_id']}")
    axes[0].set_ylabel("Amplitude")

if abnormal_row is not None:
    snip_a = load_waveform_snippet(
        RAW / abnormal_row["source"],
        abnormal_row["record_id"],
        start=int(abnormal_row["start_sample"]),
        n_samples=int(abnormal_row["fs"] * 5),
    )
    if snip_a:
        sig_a, fs_a = snip_a
        axes[1].plot(sig_a, color="#e63946", lw=0.8)
        axes[1].set_title(
            f"Trecho ABNORMAL — {abnormal_row['source']}/{abnormal_row['record_id']}"
        )
        axes[1].set_ylabel("Amplitude")
else:
    axes[1].text(0.5, 0.5, "Sem janelas abnormal no subset atual", ha="center")

axes[1].set_xlabel("Amostra")
plt.tight_layout()
plt.show()
"""
        ),
        md_cell(
            """**Gráfico 4 — formas de onda.** Visualização qualitativa de ~5 s de um trecho rotulado como normal versus um anormal. Serve para o professor ver, visualmente, que a união preserva morfologias distintas — base para o módulo de anomalias em vitais.
"""
        ),
        code_cell(
            """fig, ax = plt.subplots(figsize=(6, 4))
sns.boxplot(data=df.dropna(subset=["hr_std"]), x="label", y="hr_std", ax=ax)
ax.set_title("Variabilidade de HR (hr_std) por label")
ax.set_xlabel("Label")
ax.set_ylabel("hr_std (bpm)")
plt.tight_layout()
plt.show()
"""
        ),
        md_cell(
            """**Gráfico 5 — boxplot `hr_std`.** A variabilidade batimento-a-batimento dentro da janela tende a ser maior em trechos anormais (arritmia), o que motiva usar `hr_std` (e correlatas) como feature no Isolation Forest.
"""
        ),
        code_cell(
            """resumo = df.groupby(["source", "label"]).agg(
    n_janelas=("record_id", "size"),
    n_records=("record_id", "nunique"),
    hr_mean_mediana=("hr_mean", "median"),
    hr_std_mediana=("hr_std", "median"),
).reset_index()
display(resumo)
display(df.describe().T)
"""
        ),
        md_cell(
            """**Tabela resumo:** cruza fonte × label com medianas de HR. Confirma que `nsrdb` contribui só para `normal`, enquanto `mitdb` alimenta ambas as classes via anotações `.atr`.
"""
        ),
        md_cell(
            """### 4.6 Experimento extra — ECG Fragment High-Risk

O corpus `ecg-fragment-high-risk` permanece em `data/raw/vitals/ecg-fragment-high-risk/` e **não** foi mesclado em `arrhythmia_train.parquet`.

**Uso planejado:** após treinar Isolation Forest / PyOD no conjunto mitdb+nsrdb, avaliar **sensibilidade a eventos críticos** (fragmentos de arritmia perigosa) como teste hold-out de stress — sem contaminar o treino.
"""
        ),
    ]

    insert_at = 5
    nb["cells"] = nb["cells"][:insert_at] + new_cells + nb["cells"][insert_at:]

    cell4 = "".join(nb["cells"][4]["source"])
    old = "| Vitais | Séries **sintéticas** com anomalias injetadas | PhysioNet (MIT-BIH, MIMIC Waveform) |"
    new = "| Vitais | **MIT-BIH Arrhythmia + NSRDB** (treino) + high-risk fragments (extra) | MIMIC Waveform / vitais sintéticos auxiliares |"
    if old in cell4:
        cell4 = cell4.replace(old, new)
        lines = cell4.splitlines(keepends=True)
        if lines and not lines[-1].endswith("\n"):
            lines[-1] += "\n"
        nb["cells"][4]["source"] = lines

    nb_path.write_text(json.dumps(nb, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"OK: {len(nb['cells'])} cells; inserted {len(new_cells)} at index {insert_at}")


if __name__ == "__main__":
    main()
