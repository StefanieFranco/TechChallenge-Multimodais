"""Pré-processamento de ECG PhysioNet (MIT-BIH + NSRDB) para o pipeline de vitais."""

from __future__ import annotations

import json
import re
import shutil
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import wfdb

# Anotações MIT-BIH consideradas batimento normal (resto → abnormal).
NORMAL_BEAT_SYMBOLS = {"N", "L", "R", "e", "j"}

DATASET_SPECS: dict[str, dict[str, str]] = {
    "mitdb": {
        "db_name": "mitdb",
        "role": "treino",
        "description": "MIT-BIH Arrhythmia Database",
        "url": "https://physionet.org/content/mitdb/1.0.0/",
        "min_hea": "48",
        "zip_patterns": "mit-bih-arrhythmia|mitdb",
    },
    "nsrdb": {
        "db_name": "nsrdb",
        "role": "treino",
        "description": "MIT-BIH Normal Sinus Rhythm Database",
        "url": "https://physionet.org/content/nsrdb/1.0.0/",
        "min_hea": "18",
        "zip_patterns": "normal-sinus|nsrdb",
    },
    "ecg-fragment-high-risk": {
        "db_name": "ecg-fragment-high-risk-label",
        "role": "sensibilidade",
        "description": "ECG Fragment Database (dangerous arrhythmia)",
        "url": "https://physionet.org/content/ecg-fragment-high-risk-label/1.0.0/",
        "wget": "https://physionet.org/files/ecg-fragment-high-risk-label/1.0.0/",
        "min_hea": "1",
        "zip_patterns": "ecg-fragment|high-risk|dangerous-arrhythmia",
    },
}


def project_root_from(start: Path | None = None) -> Path:
    """Resolve a raiz do repositório a partir do CWD ou de um caminho dado."""
    start = (start or Path.cwd()).resolve()
    for candidate in [start, *start.parents]:
        if (candidate / "requirements.txt").exists() and (candidate / "src").exists():
            return candidate
    return start


def vitals_raw_dir(root: Path | None = None) -> Path:
    root = root or project_root_from()
    return root / "data" / "raw" / "vitals"


def vitals_processed_dir(root: Path | None = None) -> Path:
    root = root or project_root_from()
    return root / "data" / "processed" / "vitals"


def has_hea_files(directory: Path) -> bool:
    return directory.exists() and any(directory.rglob("*.hea"))


def count_hea_files(directory: Path) -> int:
    if not directory.exists():
        return 0
    return len(list(directory.rglob("*.hea")))


def is_download_complete(key: str, directory: Path) -> bool:
    """True se a pasta atingiu o mínimo de .hea esperado para o dataset."""
    if key not in DATASET_SPECS:
        return has_hea_files(directory)
    min_hea = int(DATASET_SPECS[key].get("min_hea", "1"))
    return count_hea_files(directory) >= min_hea


def find_zip_for_dataset(key: str, raw_root: Path | None = None) -> Path | None:
    """Localiza ZIP do dataset em data/raw/vitals/ (ou um nível acima em data/raw/)."""
    raw_root = raw_root or vitals_raw_dir()
    patterns = DATASET_SPECS[key].get("zip_patterns", key).split("|")
    search_dirs = [raw_root, raw_root.parent]
    candidates: list[Path] = []
    for folder in search_dirs:
        if not folder.exists():
            continue
        for zip_path in folder.glob("*.zip"):
            name = zip_path.name.lower()
            if any(re.search(pat, name) for pat in patterns):
                candidates.append(zip_path)
    if not candidates:
        return None
    # Preferir o maior arquivo (download completo)
    return max(candidates, key=lambda p: p.stat().st_size)


def _flatten_hea_tree(dest: Path) -> None:
    """Se o ZIP extraiu numa subpasta, move *.hea/*.dat/*.atr para dest."""
    heas = list(dest.rglob("*.hea"))
    top_heas = list(dest.glob("*.hea"))
    if top_heas or not heas:
        return
    # todos os .hea estão em subpastas → promover arquivos WFDB
    for hea in heas:
        parent = hea.parent
        for ext in (".hea", ".dat", ".atr", ".xws"):
            src = parent / f"{hea.stem}{ext}"
            if src.exists():
                target = dest / src.name
                if not target.exists():
                    shutil.move(str(src), str(target))


def extract_dataset_zip(
    key: str,
    raw_root: Path | None = None,
    zip_path: Path | None = None,
    force: bool = False,
) -> Path:
    """Extrai ZIP PhysioNet para data/raw/vitals/<key>/."""
    if key not in DATASET_SPECS:
        raise KeyError(f"Dataset desconhecido: {key}. Opções: {list(DATASET_SPECS)}")

    raw_root = raw_root or vitals_raw_dir()
    dest = raw_root / key
    dest.mkdir(parents=True, exist_ok=True)

    if is_download_complete(key, dest) and not force:
        print(f"[skip] {key}: já prontos {count_hea_files(dest)} .hea em {dest}")
        return dest

    zip_path = zip_path or find_zip_for_dataset(key, raw_root=raw_root)
    if zip_path is None:
        raise FileNotFoundError(
            f"ZIP não encontrado para '{key}'. Coloque o arquivo em {raw_root} "
            f"(padrões: {DATASET_SPECS[key].get('zip_patterns')})."
        )

    print(f"[extract] {zip_path.name} -> {dest}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest)
    _flatten_hea_tree(dest)
    print(f"[ok] {key}: {count_hea_files(dest)} records (.hea)")
    return dest


def prepare_datasets_from_zips(
    raw_root: Path | None = None,
    force: bool = False,
    required: tuple[str, ...] = ("mitdb", "nsrdb"),
) -> dict[str, Path]:
    """Prepara pastas a partir de ZIPs (ou reutiliza dados já extraídos).

    `ecg-fragment-high-risk` é opcional: se o ZIP não existir, apenas avisa.
    """
    raw_root = raw_root or vitals_raw_dir()
    paths: dict[str, Path] = {}
    for key in ("mitdb", "nsrdb", "ecg-fragment-high-risk"):
        dest = raw_root / key
        try:
            if is_download_complete(key, dest) and not force:
                print(f"[skip] {key}: {count_hea_files(dest)} .hea")
                paths[key] = dest
                continue
            if find_zip_for_dataset(key, raw_root=raw_root) is not None:
                paths[key] = extract_dataset_zip(key, raw_root=raw_root, force=force)
            elif has_hea_files(dest):
                print(
                    f"[partial] {key}: {count_hea_files(dest)} .hea "
                    "(abaixo do mínimo, sem ZIP novo — seguimos com o disponível)"
                )
                paths[key] = dest
            elif key in required:
                raise FileNotFoundError(
                    f"Dataset obrigatório '{key}' ausente. "
                    f"Coloque o ZIP correspondente em {raw_root}."
                )
            else:
                print(f"[warn] {key}: ZIP ausente — reservado para experimento extra")
                paths[key] = dest
        except Exception as exc:  # noqa: BLE001
            if key in required:
                raise
            print(f"[warn] {key}: {exc}")
            paths[key] = dest
    return paths


def download_dataset(key: str, raw_root: Path | None = None, force: bool = False) -> Path:
    """Compat: prefere ZIP local; não usa mais wfdb.dl_database por padrão."""
    return extract_dataset_zip(key, raw_root=raw_root, force=force)


def download_all_vitals_ecg(raw_root: Path | None = None, force: bool = False) -> dict[str, Path]:
    """Compat: alias de prepare_datasets_from_zips."""
    return prepare_datasets_from_zips(raw_root=raw_root, force=force)


def list_records(db_dir: Path) -> list[str]:
    """Lista record names que possuem .hea e .dat legíveis."""
    if not db_dir.exists():
        return []
    names: set[str] = set()
    for hea in db_dir.rglob("*.hea"):
        dat = hea.with_suffix(".dat")
        if dat.exists() and dat.stat().st_size > 0:
            names.add(hea.stem)
    return sorted(names)


def inventory(raw_root: Path | None = None) -> pd.DataFrame:
    """Resumo de arquivos por dataset."""
    raw_root = raw_root or vitals_raw_dir()
    rows = []
    for key, spec in DATASET_SPECS.items():
        d = raw_root / key
        n_hea = count_hea_files(d) if d.exists() else 0
        zip_found = find_zip_for_dataset(key, raw_root=raw_root)
        rows.append(
            {
                "dataset": key,
                "role": spec["role"],
                "path": str(d),
                "n_hea": n_hea,
                "n_records_ok": len(list_records(d)) if d.exists() else 0,
                "min_hea": int(spec.get("min_hea", "1")),
                "ready": is_download_complete(key, d) if d.exists() else False,
                "zip": zip_found.name if zip_found else None,
            }
        )
    return pd.DataFrame(rows)

def _clean_signal(sig: np.ndarray) -> np.ndarray | None:
    """Remove NaN, rejeita sinal constante e aplica clip por percentis."""
    x = np.asarray(sig, dtype=float).ravel()
    if x.size < 100:
        return None
    if np.all(~np.isfinite(x)):
        return None
    finite = np.isfinite(x)
    if finite.mean() < 0.95:
        return None
    # interpola buracos curtos
    if not finite.all():
        idx = np.arange(len(x))
        x = np.interp(idx, idx[finite], x[finite])
    if np.nanstd(x) < 1e-6:
        return None
    lo, hi = np.percentile(x, [0.5, 99.5])
    if hi > lo:
        x = np.clip(x, lo, hi)
    return x


def _rr_hr_from_peaks(peak_samples: np.ndarray, fs: float) -> dict[str, float]:
    if peak_samples is None or len(peak_samples) < 3 or fs <= 0:
        return {"hr_mean": np.nan, "hr_std": np.nan, "rr_mean": np.nan, "rr_std": np.nan}
    rr = np.diff(peak_samples.astype(float)) / fs
    rr = rr[(rr > 0.3) & (rr < 2.0)]  # filtra RR fisiológicos ~30–200 bpm
    if len(rr) < 2:
        return {"hr_mean": np.nan, "hr_std": np.nan, "rr_mean": np.nan, "rr_std": np.nan}
    hr = 60.0 / rr
    return {
        "hr_mean": float(np.mean(hr)),
        "hr_std": float(np.std(hr)),
        "rr_mean": float(np.mean(rr)),
        "rr_std": float(np.std(rr)),
    }


def _simple_rpeaks(sig: np.ndarray, fs: float) -> np.ndarray:
    """Detector R-peak leve (máximos locais) quando anotação não existe."""
    if fs <= 0 or len(sig) < int(fs):
        return np.array([], dtype=int)
    # janela ~0.3s
    win = max(1, int(0.3 * fs))
    thr = np.percentile(sig, 75)
    peaks: list[int] = []
    i = win
    n = len(sig) - win
    while i < n:
        seg = sig[i - win : i + win]
        j = int(np.argmax(seg)) + (i - win)
        if sig[j] >= thr and (not peaks or j - peaks[-1] >= win):
            peaks.append(j)
            i = j + win
        else:
            i += win
    return np.asarray(peaks, dtype=int)


def extract_record_features(
    db_dir: Path,
    record_id: str,
    source: str,
    window_sec: float = 10.0,
    max_windows: int = 30,
    max_samples: int | None = None,
) -> list[dict[str, Any]]:
    """Extrai features por janela de um record WFDB."""
    record_path = db_dir / record_id
    # wfdb aceita path sem extensão; se estiver em subpasta, tenta rglob
    hea = db_dir / f"{record_id}.hea"
    if not hea.exists():
        matches = list(db_dir.rglob(f"{record_id}.hea"))
        if not matches:
            return []
        record_path = matches[0].with_suffix("")

    try:
        rec = wfdb.rdrecord(str(record_path), channels=[0])
    except Exception:
        try:
            rec = wfdb.rdrecord(str(record_path))
        except Exception:
            return []

    if rec.p_signal is None:
        return []
    sig = _clean_signal(rec.p_signal[:, 0])
    if sig is None:
        return []

    fs = float(rec.fs)
    if max_samples is not None and len(sig) > max_samples:
        sig = sig[:max_samples]

    # anotações (mitdb)
    ann_symbols: list[str] | None = None
    ann_samples: np.ndarray | None = None
    try:
        ann = wfdb.rdann(str(record_path), "atr")
        ann_symbols = list(ann.symbol)
        ann_samples = np.asarray(ann.sample, dtype=int)
    except Exception:
        ann_symbols = None
        ann_samples = None

    win = int(window_sec * fs)
    if win < 50:
        return []

    step = win  # janelas não sobrepostas
    rows: list[dict[str, Any]] = []
    n_windows = 0
    for start in range(0, len(sig) - win + 1, step):
        if n_windows >= max_windows:
            break
        segment = sig[start : start + win]
        if np.std(segment) < 1e-6:
            continue

        label = "normal"
        peaks: np.ndarray
        if source == "mitdb" and ann_samples is not None and ann_symbols is not None:
            mask = (ann_samples >= start) & (ann_samples < start + win)
            syms = [s for s, m in zip(ann_symbols, mask, strict=False) if m]
            # ignora não-batimentos comuns
            beat_syms = [s for s in syms if s not in {"+", "~", "|", "[", "]", "!", "x", "(", ")", "u"}]
            if beat_syms:
                abnormal = any(s not in NORMAL_BEAT_SYMBOLS for s in beat_syms)
                label = "abnormal" if abnormal else "normal"
            peaks = ann_samples[mask] - start
            # só picos que parecem batimentos
            if beat_syms:
                peaks = peaks[: len(beat_syms)]
        else:
            # nsrdb ou sem atr → normal + R-peaks simples
            label = "normal"
            peaks = _simple_rpeaks(segment, fs)

        hr_stats = _rr_hr_from_peaks(peaks, fs)
        rows.append(
            {
                "source": source,
                "record_id": record_id,
                "window_idx": n_windows,
                "start_sample": start,
                "fs": fs,
                "n_samples": win,
                "sig_mean": float(np.mean(segment)),
                "sig_std": float(np.std(segment)),
                "label": label,
                **hr_stats,
            }
        )
        n_windows += 1

    return rows


def build_unified_training_frame(
    raw_root: Path | None = None,
    window_sec: float = 10.0,
    max_windows_per_record: int = 20,
    max_samples_per_record: int | None = 650000,
    mitdb_records: list[str] | None = None,
    nsrdb_records: list[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Limpa e unifica mitdb + nsrdb em um DataFrame de features por janela."""
    raw_root = raw_root or vitals_raw_dir()
    mit_dir = raw_root / "mitdb"
    nsr_dir = raw_root / "nsrdb"

    mit_ids = mitdb_records or list_records(mit_dir)
    nsr_ids = nsrdb_records or list_records(nsr_dir)

    rows: list[dict[str, Any]] = []
    skipped = {"mitdb": 0, "nsrdb": 0}

    for rid in mit_ids:
        feat = extract_record_features(
            mit_dir,
            rid,
            source="mitdb",
            window_sec=window_sec,
            max_windows=max_windows_per_record,
            max_samples=max_samples_per_record,
        )
        if not feat:
            skipped["mitdb"] += 1
        rows.extend(feat)

    for rid in nsr_ids:
        feat = extract_record_features(
            nsr_dir,
            rid,
            source="nsrdb",
            window_sec=window_sec,
            max_windows=max_windows_per_record,
            # NSRDB é longo: limita ~5 min @ 128 Hz
            max_samples=max_samples_per_record or int(128 * 60 * 5),
        )
        if not feat:
            skipped["nsrdb"] += 1
        rows.extend(feat)

    df = pd.DataFrame(rows)
    meta: dict[str, Any] = {
        "sources_train": ["mitdb", "nsrdb"],
        "excluded_from_train": ["ecg-fragment-high-risk"],
        "window_sec": window_sec,
        "max_windows_per_record": max_windows_per_record,
        "n_rows": int(len(df)),
        "n_mitdb_records": len(mit_ids),
        "n_nsrdb_records": len(nsr_ids),
        "skipped_records": skipped,
        "label_counts": df["label"].value_counts().to_dict() if len(df) else {},
        "source_counts": df["source"].value_counts().to_dict() if len(df) else {},
        "normal_beat_symbols": sorted(NORMAL_BEAT_SYMBOLS),
    }
    return df, meta


def save_unified_training(
    df: pd.DataFrame,
    meta: dict[str, Any],
    processed_dir: Path | None = None,
) -> tuple[Path, Path]:
    processed_dir = processed_dir or vitals_processed_dir()
    processed_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = processed_dir / "arrhythmia_train.parquet"
    meta_path = processed_dir / "arrhythmia_train_meta.json"
    df.to_parquet(parquet_path, index=False)
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return parquet_path, meta_path


def load_waveform_snippet(
    db_dir: Path,
    record_id: str,
    start: int = 0,
    n_samples: int = 2000,
    channel: int = 0,
) -> tuple[np.ndarray, float] | None:
    """Carrega um trecho limpo de ECG para plot."""
    record_path = db_dir / record_id
    if not (db_dir / f"{record_id}.hea").exists():
        matches = list(db_dir.rglob(f"{record_id}.hea"))
        if not matches:
            return None
        record_path = matches[0].with_suffix("")
    try:
        rec = wfdb.rdrecord(
            str(record_path),
            sampfrom=start,
            sampto=start + n_samples,
            channels=[channel],
        )
    except Exception:
        try:
            rec = wfdb.rdrecord(str(record_path), sampfrom=start, sampto=start + n_samples)
        except Exception:
            return None
    if rec.p_signal is None:
        return None
    sig = _clean_signal(rec.p_signal[:, 0])
    if sig is None:
        return None
    return sig, float(rec.fs)
