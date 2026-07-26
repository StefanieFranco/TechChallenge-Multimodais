"""Baixa o Llama 3 base e o adaptador médico LoRA para uso local.

Uso:
    python -m src.fine_tuning.download_local_llama

Requer autenticação Hugging Face (`hf auth login`) e aceite dos termos
de meta-llama/Meta-Llama-3-8B-Instruct.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import snapshot_download

BASE_REPO_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
ADAPTER_REPO_ID = "StefanieFranco/llama3-medical-fine-tuning"

DEFAULT_BASE_PATH = "./models/base/Meta-Llama-3-8B-Instruct"
DEFAULT_ADAPTER_PATH = "./models/llama3-8b-bnb-4bit-medical/adapter"


def _resolve_path(raw: str) -> Path:
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    return path.resolve()


def download_models(
    base_path: str | Path | None = None,
    adapter_path: str | Path | None = None,
) -> tuple[Path, Path]:
    """Baixa modelo base e adaptador médico para os caminhos do .env."""
    load_dotenv()

    base_dest = _resolve_path(
        str(base_path or os.getenv("LOCAL_LLAMA_MODEL_PATH", DEFAULT_BASE_PATH))
    )
    adapter_dest = _resolve_path(
        str(adapter_path or os.getenv("MEDICAL_ADAPTER_PATH", DEFAULT_ADAPTER_PATH))
    )

    print(f"[1/2] Baixando base {BASE_REPO_ID}")
    print(f"      -> {base_dest}")
    snapshot_download(
        repo_id=BASE_REPO_ID,
        local_dir=str(base_dest),
    )

    print(f"[2/2] Baixando adapter {ADAPTER_REPO_ID}")
    print(f"      -> {adapter_dest}")
    snapshot_download(
        repo_id=ADAPTER_REPO_ID,
        local_dir=str(adapter_dest),
    )

    print("Download concluído.")
    print(f"LOCAL_LLAMA_MODEL_PATH = {base_dest}")
    print(f"MEDICAL_ADAPTER_PATH   = {adapter_dest}")
    return base_dest, adapter_dest


def main() -> None:
    download_models()


if __name__ == "__main__":
    main()
