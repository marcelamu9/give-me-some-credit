import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from credit_risk.eda.plots import (
    plot_correlation_heatmap,
    plot_default_rate_by_decile,
    plot_distribution_by_target,
    plot_missing,
    plot_target_distribution,
    save_figure,
)
from credit_risk.eda.profile import default_rate_by_decile, missing_summary, spearman_matrix


@pytest.fixture
def frame() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "a": rng.random(60),
            "b": np.r_[np.zeros(30), rng.random(30)],
            "c": np.r_[rng.random(55), [np.nan] * 5],
            "y": np.r_[np.zeros(45), np.ones(15)].astype(int),
        }
    )


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def test_plot_target_distribution(frame):
    fig = plot_target_distribution(frame, "y")
    assert isinstance(fig, Figure) and len(fig.axes) >= 1


def test_plot_distribution_by_target(frame):
    fig = plot_distribution_by_target(frame, "a", "y")
    assert isinstance(fig, Figure) and len(fig.axes) >= 1


def test_log_distribution_with_zeros(frame):
    fig = plot_distribution_by_target(frame, "b", "y", log=True)
    assert isinstance(fig, Figure)
    assert "log" in fig.axes[0].get_xlabel()


def test_plot_distribution_ignores_missing_values(frame):
    assert isinstance(plot_distribution_by_target(frame, "c", "y"), Figure)


def test_plot_missing(frame):
    fig = plot_missing(missing_summary(frame, "y"))
    assert isinstance(fig, Figure) and len(fig.axes) >= 1


def test_heatmap_labels(frame):
    corr = spearman_matrix(frame, ["a", "b", "c"])
    fig = plot_correlation_heatmap(corr)
    labels = [t.get_text() for t in fig.axes[0].get_xticklabels()]
    assert labels == ["a", "b", "c"]


def test_plot_default_rate_by_decile(frame):
    table = default_rate_by_decile(frame, "c", "y")
    fig = plot_default_rate_by_decile(table, "c")
    assert isinstance(fig, Figure) and len(fig.axes) >= 1


def test_save_figure(tmp_path, frame):
    fig = plot_target_distribution(frame, "y")
    path = save_figure(fig, tmp_path / "sub" / "f.png")
    assert path.is_file() and path.stat().st_size > 0
    assert not plt.fignum_exists(fig.number)
