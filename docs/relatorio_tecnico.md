# Relatório Técnico — TechChallenge Multimodais

**Tema:** Monitoramento multimodal de pacientes em reabilitação/UTI com detecção de anomalias em tempo real

**Curso:** Pós FIAP — 8IADT · Fase 4

> Aviso educacional: o sistema e o LLM médico são educacionais e não substituem avaliação profissional de saúde.

Versão narrativa completa e evolutiva: [`notebooks/Relatorio.ipynb`](../notebooks/Relatorio.ipynb).

## 1. Objetivo

Analisar e fusionar vídeo (fisioterapia), áudio (consultas/check-ins) e sinais vitais/prescrição de um paciente fictício em reabilitação/UTI, detectando anomalias e gerando alertas para a equipe médica.

## 2. Arquitetura

```text
Vídeo → MediaPipe/YOLOv8 ─┐
Áudio → Whisper/Transformers ─┼→ Fusão de scores → LLM (Ollama/LoRA) → Equipe médica
Vitais → Isolation Forest/PyOD ─┘
```

| Camada | Pacote | Modelos |
|---|---|---|
| Vídeo | `src/video` | MediaPipe Pose (assimetria L/R); YOLOv8 opcional |
| Áudio | `src/audio` | Parkinson + RF; Whisper STT + termos críticos |
| Vitais / texto | `src/vitals` | IF sintético + IF-ECG; `prescription_check` |
| Fusão | `src/fusion` | `fuse_risk_scores` / cenário J.S. (E4); pesos clínicos 0.25/0.20/0.55 |
| LLM / alertas | `src/llm`, `src/alerts` | Ollama SBAR clínico + notifier Markdown |

## 3. Equivalência Azure → stack local

O enunciado sugere Azure Cognitive Services. A solução usa stack gratuita local:

| Azure (enunciado) | Equivalente local |
|---|---|
| Speech to Text | Whisper |
| Text Analytics | Transformers + léxico de termos críticos |
| Resumo / inteligência gerenciada | Ollama (`llama3.2`) + LoRA médico (HF) |

## 4. Datasets

| Modalidade | MVP | Próximo passo | Local |
|---|---|---|---|
| Vitais (ECG) | **MIT-BIH Arrhythmia (`mitdb`) + Normal Sinus Rhythm (`nsrdb`)** unificados para treino; **ECG Fragment High-Risk** como experimento extra de sensibilidade | MIMIC Waveform / SpO₂–PA sintéticos | `data/raw/vitals/` |
| Áudio | **UCI Parkinson** (features tabulares) | Coswara / WAV + Whisper | `data/raw/parkinsons/` |
| Vídeo | Clipe próprio + MediaPipe assimetria L/R (pós-AVC) | UCF101 / NTU; YOLOv8 opcional | `data/raw/videos/` |

### ECG PhysioNet (detalhe)

| Pasta | Fonte | Papel |
|---|---|---|
| `data/raw/vitals/mitdb/` | https://physionet.org/content/mitdb/1.0.0/ | Treino (normal + arritmia) |
| `data/raw/vitals/nsrdb/` | https://physionet.org/content/nsrdb/1.0.0/ | Treino (baseline sinusal) |
| `data/raw/vitals/ecg-fragment-high-risk/` | https://physionet.org/content/ecg-fragment-high-risk-label/1.0.0/ | Sensibilidade (fora da união de treino) |

**Importação:** baixar os ZIPs no site PhysioNet, copiar para `data/raw/vitals/` e extrair com `prepare_datasets_from_zips` (ver [`notebooks/Relatorio.ipynb`](../notebooks/Relatorio.ipynb) §§4.1–4.2). Não usamos `wfdb.dl_database` neste fluxo.

**Processamento:** limpeza + união mitdb+nsrdb → [`data/processed/vitals/arrhythmia_train.parquet`](../data/processed/vitals/arrhythmia_train.parquet) via [`src/vitals/ecg_preprocess.py`](../src/vitals/ecg_preprocess.py). Figuras EDA em `data/processed/vitals/figures/`.

Justificativa: datasets clínicos abertos de fisioterapia são raros; para vitais usamos ECG público anotado (proxy cardíaco de UTI/reabilitação) e documentamos limitações (ausência nativa de SpO₂/PA no MIT-BIH).

### Vídeo — fisioterapia (assimetria)

Clipes em `data/raw/videos/`. Pose via MediaPipe Pose Landmarker; heurística educacional de assimetria L/R (ombro/quadril/joelho/tronco) em `src/video/` → veredito CORRETO / ATENCAO / INCORRETO (ver `notebooks/Relatorio.ipynb` §4.7).

### Áudio — features vocais Parkinson

Corpus UCI Oxford em `data/raw/parkinsons/` (`parkinsons.data` + telemonitoring UPDRS). Análise e `voice_risk_score` em `src/audio/parkinsons_analysis.py` (ver `notebooks/Relatorio.ipynb` §4.8). Whisper/STT permanece para clips `.wav`.

### E1 — Vitais sintéticos + Isolation Forest

Série sintética em `data/raw/vitals/synthetic/js001_noite.csv`. Detector: `src/vitals/anomaly_detection.py` → `data/processed/vitals/isolation_forest_vitals.joblib` + `_meta.json`. Acompanhar treino/métricas em `notebooks/Relatorio.ipynb` §4.9 (experimento detalhado: `01_vitals_sinteticos.ipynb`).

## 5. Pipeline

1. Ingestão em `data/raw/`
2. Features em `data/processed/`
3. Inferência por modalidade
4. Fusão de risco (`src/fusion`)
5. Relatório LLM e alertas (`src/llm`, `src/alerts`)

Tempo real no MVP: simulado por janelas deslizantes.

## 6. Paciente fictício (fio condutor)

**J.S., 68 anos, pós-AVC** — assimetria na fisioterapia (vídeo), fadiga/disartria no check-in (áudio), SpO₂/taquicardia + desvio de prescrição (vitais).

## 7. Resultados

Ver tabela completa em `notebooks/Relatorio.ipynb` §10 (preenchida pelo pipeline E1–E5):
fusão multimodal, métricas RF Parkinson, IF-ECG, Whisper/termos críticos e alerta SBAR em
`data/processed/alerts/alerta_JS001.md`.

## 8. Limitações e próximos passos

- Vídeo e parte dos áudios são simulados/próprios; UCI Parkinson é proxy tabular.
- Azure substituído por equivalentes locais (justificar no vídeo de entrega).
- Extensões: YOLOv8; carregar LoRA médico HF; publicar vídeo demo (`docs/roteiro_video_demo.md`).
