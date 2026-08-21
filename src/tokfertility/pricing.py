"""Loads a dated pricing table and turns token counts into estimated cost."""

import json
from pathlib import Path

# Packaged rate cards live next to this module. Resolved from __file__ rather
# than importlib.resources: the data is installed as real files (setuptools
# package-data), so a plain path works for both editable and regular installs.
DATA_DIR = Path(__file__).resolve().parent / "data"
DEFAULT_PRICING_FILE = "pricing_2026-07-30.json"


def default_pricing_path() -> Path:
    """Path to the bundled default rate card."""
    return DATA_DIR / DEFAULT_PRICING_FILE


def load_pricing(path) -> dict:
    with open(path, encoding="utf-8") as f:
        table = json.load(f)
    return {row["model_id"]: row for row in table["models"]}


def estimate_input_cost(token_count: int, price_row: dict):
    """Estimated USD cost to send this many tokens as input. None if this
    model has no pricing entry (e.g. self-hosted open-weight models)."""
    price = price_row.get("input_usd_per_1m_tokens")
    if price is None:
        return None
    return token_count / 1_000_000 * price


def cost_per_1m_chars(token_count: int, chars: int, price_row: dict):
    """Cost to send 1M characters of this language through this model -
    normalizes across languages/sentences of different lengths."""
    price = price_row.get("input_usd_per_1m_tokens")
    if price is None or chars == 0:
        return None
    tokens_per_char = token_count / chars
    return tokens_per_char * price
