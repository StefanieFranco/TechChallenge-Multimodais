"""Atualiza células 4.2 do Relatorio para fluxo via ZIP (sem wfdb.dl_database)."""

from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    nb_path = Path("notebooks/Relatorio.ipynb")
    nb = json.loads(nb_path.read_text(encoding="utf-8"))

    replacements = {
        "### 4.2 Download dos datasets (`wfdb`)\n\nEquivalente Windows-friendly aos comandos `wget -r -N -c -np` do PhysioNet. O download é **idempotente**: se já existirem arquivos `.hea`, a célula não baixa de novo.\n":
        "### 4.2 Importação via ZIP (sem `wfdb.dl_database`)\n\nOs datasets PhysioNet são baixados manualmente no navegador e os **ZIPs** são colocados em `data/raw/vitals/`. Em seguida extraímos para as subpastas `mitdb/`, `nsrdb/` e `ecg-fragment-high-risk/`. A preparação é **idempotente**: se os `.hea` já estiverem prontos, não reextrai.\n\nZIPs esperados (nomes típicos PhysioNet):\n- `mit-bih-arrhythmia-database-1.0.0.zip` → `mitdb/`\n- `mit-bih-normal-sinus-rhythm-database-1.0.0.zip` → `nsrdb/`\n- ZIP do *ECG Fragment High-Risk* → `ecg-fragment-high-risk/` (opcional nesta etapa)\n",

        "**O que acabou de acontecer:** os três corpora ECG foram materializados em subpastas de `data/raw/vitals/`. A coluna `ready=True` indica presença de headers `.hea`. Se o fragment high-risk falhar no `wfdb`, use o fallback `wget` documentado no erro (URL em `DATASET_SPECS`).\n":
        "**O que acabou de acontecer:** as pastas em `data/raw/vitals/` foram preparadas a partir dos ZIPs (ou de dados já extraídos). `ready=True` indica que o mínimo de `.hea` foi atingido; `n_records_ok` conta apenas records com `.dat` válido. O high-risk pode ficar ausente até o ZIP correspondente ser colocado na pasta.\n",
    }

    code_old_imports = '''from src.vitals.ecg_preprocess import (
    DATASET_SPECS,
    download_all_vitals_ecg,
    inventory,
    vitals_raw_dir,
    vitals_processed_dir,
)'''

    code_new_imports = '''from src.vitals.ecg_preprocess import (
    DATASET_SPECS,
    prepare_datasets_from_zips,
    inventory,
    vitals_raw_dir,
    vitals_processed_dir,
)'''

    code_old_dl = '''# Download: mitdb + nsrdb (treino) + ecg-fragment-high-risk (sensibilidade)
# Equivale a:
#   wfdb.dl_database("mitdb", dl_dir="data/raw/vitals/mitdb")
#   wfdb.dl_database("nsrdb", dl_dir="data/raw/vitals/nsrdb")
#   wfdb.dl_database("ecg-fragment-high-risk-label", dl_dir="data/raw/vitals/ecg-fragment-high-risk")

paths = download_all_vitals_ecg(raw_root=RAW, force=False)
inv = inventory(RAW)
display(inv)
'''

    code_new_dl = '''# Importação local via ZIP (sem wfdb.dl_database)
# 1) Baixe os ZIPs no site PhysioNet
# 2) Copie para data/raw/vitals/
# 3) Rode esta célula para extrair / reutilizar pastas já prontas

paths = prepare_datasets_from_zips(raw_root=RAW, force=False)
inv = inventory(RAW)
display(inv)
print("ZIPs detectados / pastas:", {k: str(v) for k, v in paths.items()})
'''

    changed = 0
    for cell in nb["cells"]:
        src = "".join(cell.get("source", []))
        original = src
        for old, new in replacements.items():
            src = src.replace(old, new)
        src = src.replace(code_old_imports, code_new_imports)
        src = src.replace(code_old_dl, code_new_dl)
        if src != original:
            lines = src.splitlines(keepends=True)
            if lines and not lines[-1].endswith("\n"):
                lines[-1] += "\n"
            cell["source"] = lines
            changed += 1

    nb_path.write_text(json.dumps(nb, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"updated cells: {changed}")


if __name__ == "__main__":
    main()
