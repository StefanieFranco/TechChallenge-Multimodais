# Roteiro do vídeo demo (≤ 15 min)

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
