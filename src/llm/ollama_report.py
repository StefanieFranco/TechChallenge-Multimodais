"""Gera resumo/alerta clínico via Ollama (com fallback educacional)."""

from __future__ import annotations

import json
from typing import Any

import requests

# Perfil educacional — caso ALTO risco (fio condutor).
JS001_PROFILE: dict[str, Any] = {
    "patient_id": "JS-001",
    "nome_ficticio": "J.S.",
    "idade": 68,
    "sexo": "masculino",
    "condicao_principal": "AVC isquêmico recente (pós-AVC em reabilitação)",
    "contexto": (
        "Paciente em programa de reabilitação/UTI-stepdown com monitoramento multimodal "
        "(vídeo de fisioterapia, check-in de voz e sinais vitais)."
    ),
    "sequelas_esperadas": [
        "hemiparesia / assimetria motora",
        "risco de queda e técnica inadequada no exercício",
        "disartria ou fadiga vocal",
        "instabilidade de SpO₂/FC sob esforço",
        "necessidade de aderência a antiplaquetário / estatina / anti-hipertensivo",
    ],
    "risco_esperado": "alto",
}

# Perfil educacional — caso BAIXO risco (contraste).
MR001_PROFILE: dict[str, Any] = {
    "patient_id": "MR-001",
    "nome_ficticio": "M.R.",
    "idade": 45,
    "sexo": "feminino",
    "condicao_principal": "Fisioterapia preventiva / condicionamento (sem evento agudo)",
    "contexto": (
        "Check-in ambulatorial com exercício supervisionado, voz estável e sinais vitais "
        "dentro da faixa alvo — contraste educacional ao caso JS-001."
    ),
    "sequelas_esperadas": [],
    "risco_esperado": "baixo",
}

CLINICAL_SYSTEM_PROMPT_HIGH = """Você é um assistente clínico educacional em português do Brasil.
Produza um alerta multiprofissional no formato SBAR (Situação, Background, Avaliação, Recomendação),
com linguagem médico-clínica objetiva (SpO₂, FC, assimetria motora, disartria/fadiga vocal,
aderência terapêutica, fisioterapia).

O paciente fictício J.S. (JS-001) está em reabilitação pós-AVC. Além do SBAR ligado aos scores,
inclua obrigatoriamente uma seção curta **"Sobre o AVC (contexto educacional)"** explicando:
1) o que é o AVC (em 2–4 frases simples);
2) sequelas/problemas comuns que a pessoa pode apresentar;
3) como os achados multimodais deste caso se conectam a essas sequelas —
   sem inventar exames ou valores ausentes no contexto.

Regras:
- Não invente exames, imagem ou valores ausentes no contexto.
- Use o perfil do paciente e o breakdown de risco fornecidos.
- Deixe claro que o sistema é educacional e NÃO substitui avaliação profissional.
- Inclua conduta sugerida priorizada (imediata / curto prazo).
- Extensão alvo: ~450–550 palavras.
"""

CLINICAL_SYSTEM_PROMPT_LOW = """Você é um assistente clínico educacional em português do Brasil.
Produza um relatório multiprofissional no formato SBAR para um caso de **risco baixo/estável**,
com linguagem médico-clínica objetiva.

O paciente fictício M.R. (MR-001) está em fisioterapia preventiva, sem evento agudo.
Além do SBAR, inclua uma seção curta **"Por que o risco não está elevado"** explicando:
1) que exercício correto + voz estável + vitais na meta reduzem o score fusionado;
2) que o sistema também serve para documentar evolução favorável (não só alertas críticos);
3) contraste breve com um cenário de alto risco (ex.: pós-AVC com falha técnica e SpO₂/FC alterados),
   sem inventar dados ausentes.

Regras:
- Não invente exames ou valores ausentes.
- Se o check-in negar sintomas (ex.: "sem falta de ar"), NÃO afirme o contrário.
- Enfatize manutenção do plano e reassessment de rotina.
- Aviso educacional obrigatório.
- Extensão alvo: ~300–400 palavras.
"""

# Compat: nome antigo usado por imports externos
CLINICAL_SYSTEM_PROMPT = CLINICAL_SYSTEM_PROMPT_HIGH


def _context_block(context: dict[str, Any] | None, default_profile: dict[str, Any]) -> str:
    if not context:
        return "(sem contexto adicional)"
    safe: dict[str, Any] = {}
    for k, v in context.items():
        if k in {"video_report", "vitals_result", "parkinsons", "points", "df", "model"}:
            continue
        try:
            json.dumps(v, ensure_ascii=False)
            safe[k] = v
        except TypeError:
            safe[k] = str(v)
    if "patient_profile" not in safe:
        safe["patient_profile"] = default_profile
    return json.dumps(safe, ensure_ascii=False, indent=2)


def _is_low_risk_case(fusion_result: dict[str, Any], profile: dict[str, Any]) -> bool:
    if str(profile.get("risco_esperado", "")).lower() == "baixo":
        return True
    if str(profile.get("patient_id", "")).upper().startswith("MR-"):
        return True
    return str(fusion_result.get("level", "")).lower() == "baixo"


def _avc_educational_block() -> list[str]:
    return [
        "## Sobre o AVC (contexto educacional)",
        "",
        "**O que é:** o acidente vascular cerebral (AVC) ocorre quando o fluxo sanguíneo para "
        "uma área do cérebro é interrompido (isquêmico, por oclusão) ou há sangramento "
        "(hemorrágico). Sem oxigênio e nutrientes, neurônios sofrem lesão em minutos.",
        "",
        "**Problemas frequentes na reabilitação:** hemiparesia; risco de queda; disartria; "
        "disfagia; oscilações de SpO₂/FC; necessidade de aderência medicamentosa.",
        "",
        "**Ligação com o caso J.S.:** o monitoramento multimodal busca captar desvios nessas "
        "dimensões (motor, voz/sintomas, vitais/prescrição).",
    ]


def _fallback_report(fusion_result: dict[str, Any], context: dict[str, Any] | None) -> str:
    level = str(fusion_result.get("level", "moderado")).upper()
    score = float(fusion_result.get("risk_score", 0.0))
    br = fusion_result.get("breakdown") or {}
    rec = fusion_result.get("recomendacao") or ""
    ctx = context or {}
    profile = ctx.get("patient_profile") or JS001_PROFILE
    low = _is_low_risk_case(fusion_result, profile)

    video = ctx.get("video") or {}
    audio = ctx.get("audio") or {}
    vitals = ctx.get("vitals") or {}
    speech = ctx.get("speech") or {}
    rx = ctx.get("prescription") or {}
    transcript = ctx.get("transcript_text") or speech.get("transcript") or ""
    violations = rx.get("violations") or rx.get("achados") or []
    terms = speech.get("critical_terms") or speech.get("hits") or []
    sequelas = profile.get("sequelas_esperadas") or []

    lines = [
        f"# Relatório clínico educacional — risco {level} (score={score:.3f})",
        "",
        "> Aviso: template local (Ollama indisponível). Uso educacional — "
        "não substitui avaliação profissional.",
        "",
        "## S — Situação",
        f"Paciente fictício **{profile.get('nome_ficticio')}** "
        f"({profile.get('patient_id')}), {profile.get('idade')} anos — "
        f"{profile.get('condicao_principal')}. Risco multimodal **{level}** "
        f"(vídeo={float(br.get('video', 0)):.3f}, áudio={float(br.get('audio', 0)):.3f}, "
        f"vitais={float(br.get('vitals', 0)):.3f}).",
        "",
        "## B — Background",
        f"- Contexto: {profile.get('contexto')}.",
    ]
    if sequelas:
        lines.append(f"- Sequelas sob vigilância: {', '.join(map(str, sequelas))}.")
    lines += [
        f"- Motor: veredito={video.get('veredito', 'n/d')}; alertas={video.get('form_alerts') or []}.",
        f"- Áudio proxy: score={float(audio.get('score', br.get('audio', 0))):.3f} "
        f"(sujeito={audio.get('subject_proxy', 'n/d')}).",
        f"- Vitais: risk={float(vitals.get('score', br.get('vitals', 0))):.3f}; "
        f"anomalias preditas={vitals.get('n_anomalies', 'n/d')}.",
    ]
    if transcript:
        lines += ["", "### Check-in", f"> {transcript}"]
    if terms:
        lines += ["", f"**Termos críticos:** {', '.join(map(str, terms))}"]
    if violations:
        lines += ["", "**Prescrição / alvos:**"] + [f"- {v}" for v in violations]

    if low:
        lines += [
            "",
            "## A — Avaliação",
            "Os três eixos estão favoráveis: técnica correta (ou sem alerta relevante), "
            "proxy vocal de baixo risco e vitais estáveis. Isso reduz o score fusionado "
            "e indica evolução/janela segura para manter o plano.",
            "",
            "## Por que o risco não está elevado",
            "Exercício adequado + voz estável + SpO₂/FC na meta mantêm o risco **baixo**. "
            "O mesmo pipeline que alerta o caso JS-001 (pós-AVC com falha técnica e vitais "
            "anômalos) também documenta estabilidade — útil para evitar alarmes desnecessários.",
            "",
            "## R — Recomendação",
            f"1. {rec}",
            "2. Manter sessão de fisioterapia conforme protocolo.",
            "3. Reavaliar na próxima janela de monitoramento de rotina.",
            "",
            "— Fim do relatório educacional —",
        ]
    else:
        lines += [
            "",
            "## A — Avaliação",
            "Há convergência de sinais de risco motor, alteração vocal/sintomas e/ou "
            "desvios em vitais/prescrição — reassessment clínico-funcional indicado.",
            "",
            *_avc_educational_block(),
            "",
            "## R — Recomendação",
            f"1. {rec}",
            "2. Reavaliar SpO₂ e FC; acionar plantão se hipoxemia/taquicardia persistir.",
            "3. Revisar técnica com fisioterapia e aderência medicamentosa.",
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
    timeout: float = 120.0,
    temperature: float = 0.2,
) -> str:
    """Chama o Ollama para produzir resumo/alerta textual clínico.

    Se Ollama estiver indisponível, retorna template clínico determinístico.
    """
    ctx = dict(context or {})
    profile = ctx.get("patient_profile") or JS001_PROFILE
    ctx["patient_profile"] = profile
    low = _is_low_risk_case(fusion_result, profile)

    if low:
        system = CLINICAL_SYSTEM_PROMPT_LOW
        user_prompt = (
            "Redija o SBAR para o paciente M.R. (risco baixo/estável) e inclua a seção "
            "**Por que o risco não está elevado**, com contraste breve ao cenário de alto risco.\n\n"
            f"### Perfil\n{json.dumps(profile, ensure_ascii=False, indent=2)}\n\n"
            f"### Fusão\n{json.dumps(fusion_result, ensure_ascii=False, indent=2)}\n\n"
            f"### Contexto\n{_context_block(ctx, profile)}\n"
        )
    else:
        system = CLINICAL_SYSTEM_PROMPT_HIGH
        user_prompt = (
            "Redija o alerta SBAR para o paciente J.S. (pós-AVC) e inclua "
            "**Sobre o AVC (contexto educacional)**.\n\n"
            f"### Perfil\n{json.dumps(profile, ensure_ascii=False, indent=2)}\n\n"
            f"### Fusão\n{json.dumps(fusion_result, ensure_ascii=False, indent=2)}\n\n"
            f"### Contexto\n{_context_block(ctx, profile)}\n"
        )

    url = host.rstrip("/") + "/api/generate"
    payload = {
        "model": model,
        "prompt": user_prompt,
        "system": system,
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
    except Exception as exc:  # noqa: BLE001
        fb = _fallback_report(fusion_result, ctx)
        return fb + f"\n\n<!-- fallback_reason: {type(exc).__name__}: {exc} -->\n"
