"""Módulo de fusão multimodal de risco."""

from src.fusion.risk_fusion import build_js_scenario_scores, fuse_risk_scores, risk_level

__all__ = ["fuse_risk_scores", "build_js_scenario_scores", "risk_level"]
