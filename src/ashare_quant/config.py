from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import yaml


def _normalize_market_timing_config(config: dict[str, Any]) -> dict[str, Any]:
    market_timing = config.get("market_timing")
    if not isinstance(market_timing, dict):
        return config

    benchmark_symbol = market_timing.get("benchmark_symbol")
    if isinstance(benchmark_symbol, int):
        market_timing["benchmark_symbol"] = f"{benchmark_symbol:06d}"
    elif isinstance(benchmark_symbol, str):
        cleaned = benchmark_symbol.strip()
        if cleaned.isdigit() and len(cleaned) < 6:
            market_timing["benchmark_symbol"] = cleaned.zfill(6)
        else:
            market_timing["benchmark_symbol"] = cleaned
    return config


def load_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as file:
        loaded = yaml.safe_load(file)
    if not isinstance(loaded, dict):
        raise ValueError(f"invalid config structure in {path}")
    return _normalize_market_timing_config(loaded)


def compute_config_fingerprint(config_path: str | Path) -> str:
    path = Path(config_path)
    content = path.read_text(encoding="utf-8")
    return hashlib.sha256(content.encode("utf-8")).hexdigest()
