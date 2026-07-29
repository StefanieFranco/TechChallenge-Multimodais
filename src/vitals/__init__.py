"""Módulo de sinais vitais e prescrição."""

from src.vitals.anomaly_detection import detect_anomalies, run_training_pipeline
from src.vitals.prescription_check import check_prescription
from src.vitals.synthetic_vitals import generate_synthetic_vitals, load_or_create_synthetic

__all__ = [
    "detect_anomalies",
    "run_training_pipeline",
    "generate_synthetic_vitals",
    "load_or_create_synthetic",
    "check_prescription",
]
