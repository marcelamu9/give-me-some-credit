"""Plotting helpers for the EDA. They return figures; they never call plt.show()."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure


def plot_target_distribution(df: pd.DataFrame, target: str) -> Figure:
    counts = df[target].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar([str(v) for v in counts.index], counts.values)
    for bar, count in zip(bars, counts.values, strict=True):
        ax.annotate(
            f"{count:,}\n({count / counts.sum():.1%})",
            (bar.get_x() + bar.get_width() / 2, count),
            ha="center",
            va="bottom",
        )
    ax.set_xlabel(target)
    ax.set_ylabel("rows")
    ax.set_title("Distribución del target")
    ax.margins(y=0.15)
    return fig


def plot_distribution_by_target(
    df: pd.DataFrame, column: str, target: str, log: bool = False
) -> Figure:
    fig, ax = plt.subplots(figsize=(6, 4))
    values = df[column]
    if log:
        values = np.log1p(values)
    values = values.where(np.isfinite(values))
    edges = np.histogram_bin_edges(values.dropna(), bins=40)
    for label, group in values.groupby(df[target]):
        ax.hist(group.dropna(), bins=edges, density=True, alpha=0.5, label=f"{target}={label}")
    ax.set_xlabel(f"log1p({column})" if log else column)
    ax.set_ylabel("density")
    ax.set_title(f"{column} por target")
    ax.legend()
    return fig


def plot_missing(missing: pd.DataFrame) -> Figure:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(missing.index, missing["missing_pct"])
    ax.invert_yaxis()
    ax.set_xlabel("% de nulos")
    ax.set_title("Nulos por variable")
    return fig


def plot_correlation_heatmap(corr: pd.DataFrame) -> Figure:
    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(corr.values, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(corr.columns)), labels=list(corr.columns), rotation=90)
    ax.set_yticks(range(len(corr.index)), labels=list(corr.index))
    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            ax.text(j, i, f"{corr.iat[i, j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(image, ax=ax)
    ax.set_title("Correlación de Spearman")
    fig.tight_layout()
    return fig


def plot_default_rate_by_decile(table: pd.DataFrame, column: str) -> Figure:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(len(table)), table["default_rate"])
    ax.set_xticks(range(len(table)), labels=list(table["bin"]), rotation=60, ha="right")
    ax.set_ylabel("tasa de mora")
    ax.set_title(f"Tasa de mora por tramo de {column}")
    fig.tight_layout()
    return fig


def save_figure(fig: Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path
