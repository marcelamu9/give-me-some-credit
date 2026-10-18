"""Markdown rendering of the training results (no external table library)."""

import numpy as np
import pandas as pd


def _cell(value, floatfmt: str) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "-"
    if isinstance(value, float | np.floating):
        return format(value, floatfmt)
    return str(value)


def markdown_table(df: pd.DataFrame, floatfmt: str = ".4f") -> str:
    """Render a DataFrame (index excluded) as a GitHub-flavoured Markdown table."""
    lines = [
        "| " + " | ".join(str(c) for c in df.columns) + " |",
        "|" + "---|" * len(df.columns),
    ]
    for record in df.itertuples(index=False):
        lines.append("| " + " | ".join(_cell(v, floatfmt) for v in record) + " |")
    return "\n".join(lines)


def _bands_table(bands: dict) -> str:
    frame = pd.DataFrame(bands).T.rename_axis("banda").reset_index()
    frame = frame.astype({"rows": int}).rename(columns={"rows": "clientes", "share": "proporción"})
    return markdown_table(frame)


def render_report(metadata: dict) -> str:
    """Build reports/model_comparison.md from the training metadata."""
    rows = []
    for name, c in metadata["candidates"].items():
        m = c["valid_metrics"]
        rows.append(
            {
                "modelo": name,
                "AUC CV (± std)": f"{c['cv_auc_mean']:.4f} ± {c['cv_auc_std']:.4f}",
                "AUC": m["roc_auc"],
                "Gini": m["gini"],
                "KS": m["ks"],
                "PR-AUC": m["pr_auc"],
                "Brier": m["brier"],
                "Brier sin calibrar": c["valid_metrics_uncalibrated"]["brier"],
                "costo esperado": c["valid_expected_cost"],
                "t*": c["valid_t_star"],
                "búsqueda (s)": round(c["search_seconds"], 1),
                "calibración (s)": round(c["calibration_seconds"], 1),
                "latencia (ms/cliente)": round(c["latency_ms"], 2),
            }
        )
    thresholds = metadata["thresholds"]
    costs = metadata["costs"]
    sections = [
        "# Comparación de modelos",
        f"Generado por `uv run poe train` el {metadata['trained_at']} "
        f"(semilla {metadata['seed']}). "
        "Cifras de validación salvo la última sección.",
        "## Candidatos (validación)",
        markdown_table(pd.DataFrame(rows)),
        f"La latencia se midió prediciendo de una fila en una sobre {metadata['latency_rows']} "
        "filas de validación (mediana de repeticiones); el costo esperado es por cliente con "
        f"FN = {costs['false_negative']:g} y FP = {costs['false_positive']:g}.",
        "## Campeón",
        f"**{metadata['champion']}**. Regla: {metadata['selection_rule']}",
        "## Umbrales y bandas (validación)",
        f"- `t*` (rechazo) = {thresholds['t_star']:.3f} "
        f"(umbral teórico FP/(FP+FN) = {thresholds['bayes_threshold']:.3f}).",
        f"- `t_low` (aprobación) = {thresholds['t_low']:.3f}, con capacidad de revisión de "
        f"{metadata['review_capacity']:.0%}.",
        _bands_table(metadata["valid_bands"]),
        "## Sensibilidad al costo (razón FN:FP)",
        markdown_table(pd.DataFrame(metadata["cost_sensitivity"])),
        "## Sensibilidad a la capacidad de revisión",
        markdown_table(pd.DataFrame(metadata["capacity_sensitivity"])),
        "## Resultado en test (única evaluación)",
        markdown_table(pd.DataFrame([metadata["test_metrics"]])),
        f"Costo esperado en test con `t*` = {thresholds['t_star']:.3f}: "
        f"{metadata['test_expected_cost']:.4f}.",
        _bands_table(metadata["test_bands"]),
    ]
    return "\n\n".join(sections) + "\n"
