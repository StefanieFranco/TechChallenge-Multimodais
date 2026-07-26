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
| Vídeo | `src/video` | MediaPipe Pose, YOLOv8 |
| Áudio | `src/audio` | Whisper + features de fala |
| Vitais / texto | `src/vitals` | PyOD / Isolation Forest + checagem de prescrição |
| Fusão | `src/fusion` | Score ponderado dos 3 riscos |
| LLM / alertas | `src/llm`, `src/alerts` | Ollama + adapter médico |

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
| Vitais | Séries sintéticas com anomalias injetadas | PhysioNet (MIT-BIH, MIMIC Waveform) | `data/raw/vitals/` |
| Áudio | Clip curto próprio / amostra pública | Coswara, Parkinson (PhysioNet/UCI) | `data/raw/audio/` |
| Vídeo | Webcam própria (exercício simulado) | UCF101 / NTU adaptados | `data/raw/video/` |

Justificativa: datasets clínicos abertos de fisioterapia são raros; dados controlados facilitam validação e são documentados como limitação acadêmica.

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

_Preencher após experimentos (ver checklist em `notebooks/Relatorio.ipynb`)._

## 8. Limitações e próximos passos

- Vídeo e parte dos áudios são simulados/próprios por limitação de datasets clínicos públicos.
- Azure substituído por equivalentes locais (justificar no vídeo de entrega).
- Próximos notebooks: vitais sintéticos, áudio, pose, fusão/alertas.
