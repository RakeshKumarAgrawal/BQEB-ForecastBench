"""Tests for model ranking and metric summary statistics."""

import pytest

from benchmark.evaluation.comparison import ModelComparison
from benchmark.evaluation.results import EvaluationResult


def _result(name: str, rmse: float, r2: float) -> EvaluationResult:
    return EvaluationResult(
        model_name=name,
        dataset="test",
        mae=rmse - 0.1,
        rmse=rmse,
        mape=rmse / 10,
        r2=r2,
        prediction_count=10,
        evaluation_timestamp="2026-08-01T00:00:00+00:00",
        configuration_hash="hash",
        repository_version="0.4.0",
        additional_metrics={},
    )


def test_comparison_ranks_minimized_and_maximized_metrics() -> None:
    """Error metrics should minimize while R-squared should maximize."""
    comparison = ModelComparison(
        [_result("forest", 0.5, 0.8), _result("linear", 0.8, 0.9)]
    )

    assert [result.model_name for result in comparison.rank("rmse")] == [
        "forest",
        "linear",
    ]
    assert comparison.best("r2").model_name == "linear"


def test_comparison_returns_summary_statistics() -> None:
    """Summary values should describe every available model metric."""
    comparison = ModelComparison(
        [_result("forest", 0.5, 0.8), _result("linear", 0.7, 0.9)]
    )

    summary = comparison.summary_statistics()

    assert summary["rmse"]["count"] == 2
    assert summary["rmse"]["mean"] == pytest.approx(0.6)
    assert summary["rmse"]["minimum"] == pytest.approx(0.5)


def test_comparison_rejects_empty_or_unavailable_metrics() -> None:
    """Comparison requires results and a metric shared by at least one model."""
    with pytest.raises(ValueError, match="at least one"):
        ModelComparison([])
    comparison = ModelComparison([_result("linear", 0.7, 0.9)])
    with pytest.raises(ValueError, match="unavailable"):
        comparison.rank("missing")
