"""Tests for publication figure generation and source consistency."""

from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from benchmark.publication import PublicationData
from benchmark.publication.figures import (
    generate_figures,
    performance_values,
    prediction_values,
)


def test_figure_values_exactly_match_source_frames(tmp_path: Path) -> None:
    """Figure extraction must preserve metric and prediction source values."""
    data = PublicationData.load()

    pd.testing.assert_frame_equal(
        performance_values(data),
        data.metrics[["Model", "Dataset", "MAE", "RMSE", "MAPE", "R²"]],
    )
    pd.testing.assert_frame_equal(
        prediction_values(data),
        data.predictions[["SampleID", "Timestamp", "Model", "Actual", "Predicted"]],
    )

    outputs = generate_figures(data, tmp_path)

    assert len(outputs) == 6
    assert all(path.is_file() and path.stat().st_size > 0 for path in outputs)
    with Image.open(tmp_path / "Figure4_PerformanceComparison.png") as image:
        dpi = image.info["dpi"]
    assert dpi[0] == pytest.approx(300.0, abs=0.01)
    assert dpi[1] == pytest.approx(300.0, abs=0.01)
    assert "SHA-256" in (tmp_path / "Figure4_PerformanceComparison.svg").read_text(
        encoding="utf-8"
    )
