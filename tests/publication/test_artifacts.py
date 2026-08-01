"""Integration tests for complete publication artifact generation."""

from pathlib import Path

from benchmark.publication import PublicationAssetGenerator, PublicationOutputPaths


def test_generator_creates_complete_publication_asset_set(tmp_path: Path) -> None:
    """One generation call should create exactly the required Batch 3 outputs."""
    outputs = PublicationAssetGenerator(
        output_paths=PublicationOutputPaths(
            tables=tmp_path / "tables",
            figures=tmp_path / "figures",
            reports=tmp_path / "reports",
        )
    ).generate()

    assert {path.name for path in outputs} == {
        "Table5_BenchmarkResults.csv",
        "Table5_BenchmarkResults.md",
        "Figure4_PerformanceComparison.png",
        "Figure4_PerformanceComparison.svg",
        "Figure5_ActualVsPredicted.png",
        "Figure5_ActualVsPredicted.svg",
        "Figure4_caption.md",
        "Figure5_caption.md",
        "evaluation_summary.md",
        "experiment_report.md",
        "benchmark_report.md",
    }
    assert all(path.is_file() and path.stat().st_size > 0 for path in outputs)
