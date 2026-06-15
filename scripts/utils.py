"""Shared helpers for CLI scripts."""

import yaml
import json
import logging


from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)


def load_config(section: str, config_path: str = "config.yaml") -> dict:
    """Load a non-empty configuration section from YAML."""
    path = Path(config_path)
    try:
        with open(path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
    except FileNotFoundError:
        logger.error("Config file not found: %s", path)
        raise
    except yaml.YAMLError as exc:
        logger.error("Error parsing config %s: %s", path, exc)
        raise

    if section not in config:
        raise KeyError(f"Section '{section}' not found in {path}")

    section_config = config[section]
    if not section_config:
        raise ValueError(f"Section '{section}' is empty in {path}")

    return section_config


def drop_keys(config: dict, *keys: str) -> dict:
    """Return a copy of config without the given keys."""
    exclude = set(keys)
    return {key: value for key, value in config.items() if key not in exclude}


def require_file(path_value: str | Path, description: str = "File") -> Path:
    """Ensure a configured path exists on disk."""
    path = Path(path_value)
    if not path.exists():
        raise FileNotFoundError(f"{description} not found: {path}")
    return path


def to_json_serializable(obj: Any) -> Any:
    """Convert NumPy scalars/arrays and nested structures to JSON-safe values."""
    import numpy as np

    if isinstance(obj, dict):
        return {key: to_json_serializable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_json_serializable(value) for value in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def save_json(data: dict, output_path: Path) -> None:
    """Write a dictionary to JSON with NumPy-safe serialization."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(to_json_serializable(data), f, indent=2)
