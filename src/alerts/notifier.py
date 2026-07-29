"""Envio de alertas a partir do risco fusionado / relatório LLM."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _project_root() -> Path:
    start = Path.cwd().resolve()
    for candidate in [start, *start.parents]:
        if (candidate / "requirements.txt").exists() and (candidate / "src").exists():
            return candidate
    return start


def default_alert_path(root: Path | None = None, patient_id: str = "JS001") -> Path:
    root = root or _project_root()
    return root / "data" / "processed" / "alerts" / f"alerta_{patient_id}.md"


def notify(
    message: str,
    risk_level: str,
    payload: dict[str, Any] | None = None,
    *,
    path: str | Path | None = None,
    patient_id: str = "JS001",
) -> Path:
    """Dispara notificação de alerta (print + arquivo Markdown).

    Args:
        message: Texto do alerta.
        risk_level: Nível de risco (ex.: baixo, médio/moderado, alto).
        payload: Dados estruturados opcionais.
        path: Destino do arquivo; padrão data/processed/alerts/alerta_JS001.md.
        patient_id: Sufixo do arquivo padrão.

    Returns:
        Caminho do arquivo gravado.
    """
    out = Path(path) if path else default_alert_path(patient_id=patient_id)
    out.parent.mkdir(parents=True, exist_ok=True)

    ts = datetime.now(timezone.utc).isoformat()
    header = (
        f"---\n"
        f"patient_id: {patient_id}\n"
        f"risk_level: {risk_level}\n"
        f"timestamp_utc: {ts}\n"
        f"---\n\n"
    )
    body = message.strip() + "\n"
    if payload:
        body += "\n## Payload estruturado\n\n```json\n"
        import json

        try:
            body += json.dumps(payload, ensure_ascii=False, indent=2, default=str)
        except TypeError:
            body += str(payload)
        body += "\n```\n"

    out.write_text(header + body, encoding="utf-8")
    print(f"[ALERTA {risk_level.upper()}] gravado em {out}")
    print(message[:500] + ("..." if len(message) > 500 else ""))
    return out
