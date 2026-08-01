"""Publication figures rendered directly from Batch 2 CSV artifacts."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any, cast

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from benchmark.publication.inputs import PublicationData
from benchmark.utils.filesystem import ensure_directory

LOGGER = logging.getLogger(__name__)

PALETTE = ("#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9")
DISPLAY_NAMES = {
    "gradient_boosting": "Gradient Boosting",
    "linear_regression": "Linear Regression",
    "random_forest": "Random Forest",
}


def performance_values(data: PublicationData) -> pd.DataFrame:
    """Return the exact metric columns and rows plotted in Figure 4."""
    return data.metrics.loc[:, ["Model", "Dataset", "MAE", "RMSE", "MAPE", "R²"]].copy()


def prediction_values(data: PublicationData) -> pd.DataFrame:
    """Return the exact prediction columns and rows plotted in Figure 5."""
    return data.predictions.loc[
        :, ["SampleID", "Timestamp", "Model", "Actual", "Predicted"]
    ].copy()


def generate_figures(data: PublicationData, directory: Path) -> tuple[Path, ...]:
    """Generate both publication figures, SVG variants, and caption files."""
    destination = ensure_directory(directory)
    outputs = (
        *_performance_figure(data, destination),
        *_prediction_figure(data, destination),
    )
    LOGGER.info("Generated publication figure assets count=%d", len(outputs))
    return outputs


def _performance_figure(data: PublicationData, directory: Path) -> tuple[Path, ...]:
    values = performance_values(data)
    models = tuple(values["Model"].drop_duplicates())
    datasets = tuple(values["Dataset"].drop_duplicates())
    metrics = (("MAE", "MAE"), ("RMSE", "RMSE"), ("MAPE", "MAPE"), ("R²", "R²"))
    with plt.rc_context(_publication_style()):
        figure, axes = plt.subplots(2, 2, figsize=(7.2, 5.4))
        figure.subplots_adjust(top=0.82, hspace=0.38, wspace=0.26)
        x_positions = np.arange(len(datasets), dtype=float)
        width = 0.24
        for axis, (column, label) in zip(axes.flat, metrics, strict=True):
            for index, model in enumerate(models):
                model_values = values.loc[values["Model"] == model].set_index("Dataset")
                heights = [
                    float(model_values.loc[dataset, column]) for dataset in datasets
                ]
                axis.bar(
                    x_positions + (index - (len(models) - 1) / 2) * width,
                    heights,
                    width,
                    label=DISPLAY_NAMES.get(model, model),
                    color=PALETTE[index],
                )
            axis.set_ylabel(label)
            axis.set_xticks(x_positions, [name.title() for name in datasets])
            axis.axhline(0.0, color="#333333", linewidth=0.6)
            axis.grid(axis="y", alpha=0.25)
        handles, labels = axes.flat[0].get_legend_handles_labels()
        figure.suptitle("Forecasting performance by dataset partition", y=0.98)
        figure.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.93),
            ncols=len(models),
        )
        png = directory / "Figure4_PerformanceComparison.png"
        svg = directory / "Figure4_PerformanceComparison.svg"
        _save_figure(figure, png, svg, data.paths.metrics)
        plt.close(figure)
    caption = directory / "Figure4_caption.md"
    caption.write_text(
        "**Figure 4. Forecasting performance comparison.** MAE, RMSE, MAPE, "
        "and R² for each baseline model on the training, validation, and test "
        "partitions. Values are plotted directly from "
        "`artifacts/evaluation/metrics.csv`.\n",
        encoding="utf-8",
    )
    return png, svg, caption


def _prediction_figure(data: PublicationData, directory: Path) -> tuple[Path, ...]:
    values = prediction_values(data)
    actual = values.drop_duplicates("SampleID", keep="first")
    models = tuple(values["Model"].drop_duplicates())
    x_positions = np.arange(len(actual), dtype=float)
    with plt.rc_context(_publication_style()):
        figure, axis = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
        axis.plot(
            x_positions,
            actual["Actual"].to_numpy(dtype=float),
            color="#000000",
            linewidth=2.0,
            marker="o",
            markersize=3.5,
            label="Actual",
        )
        for index, model in enumerate(models):
            model_values = values.loc[values["Model"] == model].set_index("SampleID")
            predicted = [
                float(model_values.loc[sample_id, "Predicted"])
                for sample_id in actual["SampleID"]
            ]
            axis.plot(
                x_positions,
                predicted,
                color=PALETTE[index],
                linewidth=1.4,
                label=DISPLAY_NAMES.get(model, model),
            )
        partitions = actual["SampleID"].str.split("-").str[0]
        boundaries = np.flatnonzero(
            partitions.to_numpy()[1:] != partitions.to_numpy()[:-1]
        )
        for boundary in boundaries:
            axis.axvline(boundary + 0.5, color="#777777", linestyle="--", linewidth=0.8)
        tick_step = max(1, len(actual) // 6)
        ticks = x_positions[::tick_step]
        labels = (
            pd.to_datetime(actual["Timestamp"]).dt.strftime("%H:%M").iloc[::tick_step]
        )
        axis.set_xticks(ticks, labels)
        axis.set_xlabel("Timestamp (UTC)")
        axis.set_ylabel("Load (kW)")
        axis.set_title("Actual and predicted load across benchmark partitions")
        axis.grid(alpha=0.22)
        axis.legend(ncols=2, frameon=False)
        png = directory / "Figure5_ActualVsPredicted.png"
        svg = directory / "Figure5_ActualVsPredicted.svg"
        _save_figure(figure, png, svg, data.paths.predictions)
        plt.close(figure)
    caption = directory / "Figure5_caption.md"
    caption.write_text(
        "**Figure 5. Actual versus predicted load.** Observed load and baseline "
        "predictions across the training, validation, and test partitions; dashed "
        "lines mark partition boundaries. Values are plotted directly from "
        "`artifacts/evaluation/predictions.csv`.\n",
        encoding="utf-8",
    )
    return png, svg, caption


def _save_figure(figure: Figure, png: Path, svg: Path, source: Path) -> None:
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    description = f"Source: {source.as_posix()}; SHA-256: {source_hash}"
    figure.savefig(
        png,
        dpi=300,
        bbox_inches="tight",
        metadata={"Description": description},
    )
    figure.savefig(
        svg,
        bbox_inches="tight",
        metadata={"Description": description},
    )


def _publication_style() -> Any:
    return cast(
        Any,
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "svg.fonttype": "none",
        },
    )
