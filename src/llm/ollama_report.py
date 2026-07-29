"""Gera resumo/alerta clínico via Ollama (com fallback educacional)."""

from __future__ import annotations

import json
from typing import Any

import requests

CLINICAL_SYSTEM_PROMPT = """Você é um assistente clínico educacional em português do Brasil.
Produza um alerta multiprofissional no formato SBAR (Situação, Background, Avaliação, Recomendação),
com linguagem médico-clínica objetiva (SpO₂, FC, assimetria motora, disartria/fadiga vocal,
aderência terapêutica, fisioterapia).

Regras:
- Não invente exames ou valores ausentes no contexto.
- Deixe claro que o sistema é educacional e NÃO substitui avaliação profissional.
- Inclua conduta sugerida priorizada (imediata / curto prazo).
- Seja conciso (máx. ~350 palavras).
"""


def _context_block(context: dict[str, Any] | None) -> str:
    if not context:
        return "(sem contexto adicional)"
    # Evita dumps enormes / não serializáveis
    safe: dict[str, Any] = {}
    for k, v in context.items():
        if k in {"video_report", "vitals_result", "parkinsons", "points", "df", "model"}:
            continue
        try:
            json.dumps(v, ensure_ascii=False)
            safe[k] = v
        except TypeError:
            safe[k] = str(v)
    return json.dumps(safe, ensure_ascii=False, indent=2)


def _fallback_report(fusion_result: dict[str, Any], context: dict[str, Any] | None) -> str:
    level = str(fusion_result.get("level", "moderado")).upper()
    score = float(fusion_result.get("risk_score", 0.0))
    br = fusion_result.get("breakdown") or {}
    rec = fusion_result.get("recomendacao") or ""
    ctx = context or {}

    video = ctx.get("video") or {}
    audio = ctx.get("audio") or {}
    vitals = ctx.get("vitals") or {}
    speech = ctx.get("speech") or {}
    rx = ctx.get("prescription") or {}
    transcript = ctx.get("transcript_text") or speech.get("transcript") or ""

    violations = rx.get("violations") or rx.get("achados") or []
    terms = speech.get("critical_terms") or speech.get("hits") or []

    lines = [
        f"# Alerta clínico educacional — risco {level} (score={score:.3f})",
        "",
        "> Aviso: conteúdo gerado por template local (Ollama indisponível). "
        "Uso estritamente educacional — não substitui avaliação profissional.",
        "",
        "## S — Situação",
        f"Paciente fictício JS-001 (pós-AVC / reabilitação). Risco multimodal **{level}** "
        f"(vídeo={float(br.get('video', 0)):.3f}, áudio={float(br.get('audio', 0)):.3f}, "
        f"vitais={float(br.get('vitals', 0)):.3f}).",
        "",
        "## B — Background",
        f"- Motor/fisioterapia: veredito={video.get('veredito', 'n/d')}; "
        f"alertas={video.get('form_alerts') or []}.",
        f"- Voz (proxy UCI Parkinson): score={float(audio.get('score', br.get('audio', 0))):.3f} "
        f"(sujeito={audio.get('subject_proxy', 'n/d')}).",
        f"- Vitais sintéticos (noite): anomalias preditas={vitals.get('n_anomalies', 'n/d')}; "
        f"risk={float(vitals.get('score', br.get('vitals', 0))):.3f}.",
    ]
    if transcript:
        lines += ["", "### Check-in (transcrição)", f"> {transcript}"]
    if terms:
        lines += ["", f"**Termos críticos detectados:** {', '.join(map(str, terms))}"]
    if violations:
        lines += ["", "**Prescrição / alvos:**"] + [f"- {v}" for v in violations]

    lines += [
        "",
        "## A — Avaliação",
        "Há convergência de sinais de risco motor (técnica incorreta / assimetria), "
        "alteração vocal proxy e desvios em sinais vitais e/ou alvos prescritos. "
        "O padrão sugere necessidade de reassessment clínico-funcional na janela atual.",
        "",
        "## R — Recomendação",
        f"1. {rec}",
        "2. Reavaliar SpO₂ e FC; se hipoxemia ou taquicardia persistir, acionar plantão.",
        "3. Revisar técnica do exercício com fisioterapia e reforçar aderência medicamentosa.",
        "4. Documentar evolução e repetir fusão multimodal na próxima janela.",
        "",
        "— Fim do alerta educacional —",
    ]
    return "\n".join(lines)


def generate_report(
    fusion_result: dict[str, Any],
    context: dict[str, Any] | None = None,
    model: str = "llama3.2",
    host: str = "http://localhost:11434",
    *,
    timeout: float = 90.0,
    temperature: float = 0.2,
) -> str:
    """Chama o Ollama para produzir resumo e alerta textual clínico.

    Se Ollama estiver indisponível, retorna template clínico determinístico.
    """
    user_prompt = (
        "Com base na fusão de risco e no contexto abaixo, redija o alerta SBAR "
        "para a equipe multiprofissional.\n\n"
        f"### Fusão\n{json.dumps(fusion_result, ensure_ascii=False, indent=2)}\n\n"
        f"### Contexto\n{_context_block(context)}\n"
    )

    url = host.rstrip("/") + "/api/generate"
    payload = {
        "model": model,
        "prompt": user_prompt,
        "system": CLINICAL_SYSTEM_PROMPT,
        "stream": False,
        "options": {"temperature": temperature},
    }

    try:
        resp = requests.post(url, json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        text = (data.get("response") or "").strip()
        if not text:
            raise RuntimeError("Resposta vazia do Ollama.")
        disclaimer = (
            "\n\n---\n*Aviso educacional: saída do modelo local Ollama "
            f"(`{model}`). Não substitui avaliação clínica profissional.*"
        )
        return text + disclaimer
    except Exception as exc:  # noqa: BLE001 — MVP: qualquer falha → fallback
        fb = _fallback_report(fusion_result, context)
        return fb + f"\n\n<!-- fallback_reason: {type(exc).__name__}: {exc} -->\n"
