from pathlib import Path

import pytest

from credit_risk.config import PROJECT_ROOT, ConfigError, load_config

VALID_YAML = """\
seed: 7
target: SeriousDlqin2yrs
paths:
  raw: data/raw
  interim: data/interim
  processed: data/processed
  external: data/external
  models: models
  reports: reports
data:
  training_file: cs-training.csv
costs:
  false_negative: 5
  false_positive: 1
"""


def write_params(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "params.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_default_params_file_loads():
    cfg = load_config()
    assert cfg.seed == 42
    assert cfg.target == "SeriousDlqin2yrs"
    assert cfg.training_file == PROJECT_ROOT / "data" / "raw" / "cs-training.csv"
    assert cfg.costs.false_negative == 5.0
    assert cfg.costs.false_positive == 1.0


def test_relative_paths_resolve_against_project_root_not_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = load_config(write_params(tmp_path, VALID_YAML))
    assert cfg.paths.raw == PROJECT_ROOT / "data" / "raw"
    assert cfg.paths.models == PROJECT_ROOT / "models"


def test_absolute_paths_are_kept(tmp_path):
    elsewhere = (tmp_path / "elsewhere").as_posix()
    text = VALID_YAML.replace("raw: data/raw", f'raw: "{elsewhere}"')
    cfg = load_config(write_params(tmp_path, text))
    assert cfg.paths.raw == Path(elsewhere)


def test_costs_are_parsed_as_floats(tmp_path):
    cfg = load_config(write_params(tmp_path, VALID_YAML))
    assert isinstance(cfg.costs.false_negative, float)
    assert cfg.costs.false_negative == 5.0


def test_missing_file_raises_config_error(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "missing.yaml")


def test_empty_file_raises_config_error(tmp_path):
    with pytest.raises(ConfigError, match="empty"):
        load_config(write_params(tmp_path, ""))


def test_missing_top_level_key_is_named(tmp_path):
    text = VALID_YAML.replace("seed: 7\n", "")
    with pytest.raises(ConfigError, match="seed"):
        load_config(write_params(tmp_path, text))


def test_missing_path_key_is_named(tmp_path):
    text = VALID_YAML.replace("  models: models\n", "")
    with pytest.raises(ConfigError, match="models"):
        load_config(write_params(tmp_path, text))


def test_section_that_is_not_a_mapping_is_rejected(tmp_path):
    text = VALID_YAML.replace("costs:\n  false_negative: 5\n  false_positive: 1\n", "costs: null\n")
    with pytest.raises(ConfigError, match="costs"):
        load_config(write_params(tmp_path, text))


@pytest.mark.parametrize(("fn", "fp"), [(0, 1), (5, -1)])
def test_non_positive_costs_are_rejected(tmp_path, fn, fp):
    text = VALID_YAML.replace("false_negative: 5", f"false_negative: {fn}").replace(
        "false_positive: 1", f"false_positive: {fp}"
    )
    with pytest.raises(ConfigError, match="positive"):
        load_config(write_params(tmp_path, text))


def test_non_numeric_cost_is_rejected(tmp_path):
    text = VALID_YAML.replace("false_negative: 5", "false_negative: high")
    with pytest.raises(ConfigError, match="number"):
        load_config(write_params(tmp_path, text))
