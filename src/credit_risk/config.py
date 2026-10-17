"""Project configuration: loads configs/params.yaml and resolves paths against the project root."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PARAMS_PATH = PROJECT_ROOT / "configs" / "params.yaml"

TOP_LEVEL_KEYS = ("seed", "target", "paths", "data", "costs", "split")
PATH_KEYS = ("raw", "interim", "processed", "external", "models", "reports")
DATA_KEYS = ("training_file",)
COST_KEYS = ("false_negative", "false_positive")
SPLIT_KEYS = ("test_size", "valid_size")


class ConfigError(ValueError):
    """Raised when params.yaml is missing or invalid."""


@dataclass(frozen=True)
class Paths:
    raw: Path
    interim: Path
    processed: Path
    external: Path
    models: Path
    reports: Path


@dataclass(frozen=True)
class Costs:
    false_negative: float
    false_positive: float


@dataclass(frozen=True)
class Split:
    test_size: float
    valid_size: float


@dataclass(frozen=True)
class Config:
    seed: int
    target: str
    training_file: Path
    paths: Paths
    costs: Costs
    split: Split


def _require(section: Any, keys: tuple[str, ...], where: str) -> dict[str, Any]:
    if not isinstance(section, dict):
        raise ConfigError(f"{where} is empty or not a mapping")
    missing = [key for key in keys if key not in section]
    if missing:
        raise ConfigError(f"Missing keys in {where}: {', '.join(missing)}")
    return section


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _positive_float(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"costs.{name} must be a number, got {value!r}") from exc
    if number <= 0:
        raise ConfigError(f"costs.{name} must be positive, got {number}")
    return number


def _fraction(value: Any, name: str) -> float:
    message = f"split.{name} must be a number between 0 and 1 (exclusive), got {value!r}"
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(message) from exc
    if not 0 < number < 1:
        raise ConfigError(message)
    return number


def load_config(path: Path | str = DEFAULT_PARAMS_PATH) -> Config:
    """Load and validate params.yaml. Relative paths resolve against PROJECT_ROOT, not the cwd."""
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"Config file not found: {path}")

    raw = _require(yaml.safe_load(path.read_text(encoding="utf-8")), TOP_LEVEL_KEYS, path.name)
    path_section = _require(raw["paths"], PATH_KEYS, f"{path.name} [paths]")
    data_section = _require(raw["data"], DATA_KEYS, f"{path.name} [data]")
    cost_section = _require(raw["costs"], COST_KEYS, f"{path.name} [costs]")
    split_section = _require(raw["split"], SPLIT_KEYS, f"{path.name} [split]")

    paths = Paths(**{key: _resolve(path_section[key]) for key in PATH_KEYS})
    costs = Costs(**{key: _positive_float(cost_section[key], key) for key in COST_KEYS})
    split = Split(**{key: _fraction(split_section[key], key) for key in SPLIT_KEYS})

    return Config(
        seed=int(raw["seed"]),
        target=str(raw["target"]),
        training_file=paths.raw / data_section["training_file"],
        paths=paths,
        costs=costs,
        split=split,
    )
