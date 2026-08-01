"""Metric-driven comparison of model evaluation results."""

from statistics import fmean, pstdev

from benchmark.evaluation.results import EvaluationResult


class ModelComparison:
    """Rank evaluated models and summarize their metric distributions."""

    def __init__(self, results: list[EvaluationResult]) -> None:
        """Initialize a comparison from at least one evaluation result."""
        if not results:
            raise ValueError("Model comparison requires at least one result")
        self._results = tuple(results)

    @property
    def results(self) -> tuple[EvaluationResult, ...]:
        """Return the immutable result collection."""
        return self._results

    def rank(
        self, metric: str = "rmse", *, higher_is_better: bool | None = None
    ) -> tuple[EvaluationResult, ...]:
        """Return models ordered from best to worst for one metric."""
        normalized = metric.strip().lower()
        maximize = normalized == "r2" if higher_is_better is None else higher_is_better
        available = [
            (result.get_metric(normalized), result)
            for result in self._results
            if normalized in result.metrics
        ]
        if not available:
            raise ValueError(f"Metric {normalized!r} is unavailable for comparison")
        available.sort(
            key=lambda item: (
                -item[0] if maximize else item[0],
                item[1].model_name,
            )
        )
        return tuple(result for _, result in available)

    def best(
        self, metric: str = "rmse", *, higher_is_better: bool | None = None
    ) -> EvaluationResult:
        """Return the best-performing model for one metric."""
        return self.rank(metric, higher_is_better=higher_is_better)[0]

    def summary_statistics(self) -> dict[str, dict[str, float | int]]:
        """Return count, mean, spread, and extrema for every available metric."""
        names = sorted({name for result in self._results for name in result.metrics})
        summary: dict[str, dict[str, float | int]] = {}
        for name in names:
            values = [
                result.get_metric(name)
                for result in self._results
                if name in result.metrics
            ]
            summary[name] = {
                "count": len(values),
                "mean": fmean(values),
                "standard_deviation": pstdev(values),
                "minimum": min(values),
                "maximum": max(values),
            }
        return summary
