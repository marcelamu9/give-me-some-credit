import importlib

import pytest

SUBPACKAGES = ["data", "features", "models", "evaluation", "decision", "monitoring", "api"]


@pytest.mark.parametrize("name", SUBPACKAGES)
def test_subpackage_is_importable(name):
    importlib.import_module(f"credit_risk.{name}")
