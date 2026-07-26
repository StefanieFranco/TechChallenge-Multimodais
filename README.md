# TechChallenge Multimodais

Trabalho final da **Fase 4** (pós FIAP · 8IADT).

**Tema:** Monitoramento multimodal de pacientes em reabilitação/UTI com detecção de anomalias em tempo real.

Pipeline local (stack gratuita) com vídeo, áudio e sinais vitais → fusão de risco → alertas via LLM.

> Aviso: uso educacional. Não substitui avaliação clínica profissional.

## Relatório

Comece pelo notebook: [`notebooks/Relatorio.ipynb`](notebooks/Relatorio.ipynb)  
Espelho em Markdown: [`docs/relatorio_tecnico.md`](docs/relatorio_tecnico.md)

## Arquitetura

```text
Vídeo (fisioterapia)  → MediaPipe / YOLOv8  ─┐
Áudio (consulta)      → Whisper / Transformers─┼→ Fusão → LLM (Ollama / LoRA médico) → Equipe
Sinais vitais         → Isolation Forest/PyOD ─┘
```

### Equivalência Azure → local

| Enunciado (Azure) | Este projeto |
|---|---|
| Speech to Text | Whisper |
| Text Analytics | Transformers + termos críticos |
| Serviços cognitivos / resumo | Ollama + adapter médico HF |

## Estrutura

```text
techchallenge-multimodais/
├── README.md
├── requirements.txt
├── .env.example
├── data/
│   ├── raw/              # datasets baixados (vitais, áudio, vídeo)
│   └── processed/
├── src/
│   ├── video/            # MediaPipe / YOLOv8
│   ├── audio/            # Whisper + fala
│   ├── vitals/           # PyOD / Isolation Forest + prescrição
│   ├── fusion/           # combina os 3 scores
│   ├── llm/              # Ollama / relatório
│   ├── alerts/
│   └── fine_tuning/      # download Llama + adapter médico
├── notebooks/
│   └── Relatorio.ipynb   # relatório técnico (startup)
├── tests/
└── docs/
    └── relatorio_tecnico.md
```

## Setup

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Variáveis de modelo (`.env`):

```env
LOCAL_LLAMA_MODEL_PATH=./models/base/Meta-Llama-3-8B-Instruct
MEDICAL_ADAPTER_PATH=./models/llama3-8b-bnb-4bit-medical/adapter
```

```powershell
python -m src.fine_tuning.download_local_llama
ollama pull llama3.2
```

## Entregáveis da Fase 4

- Repositório com código + relatório (fluxo, modelos, resultados)
- Vídeo ≤ 15 min (YouTube/Vimeo) demonstrando áudio, vídeo, anomalias e alerta à equipe
