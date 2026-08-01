"""Tests for report sections, traceability, and source consistency."""

from pathlib import Path

from benchmark.publication import PublicationData, generate_reports
from benchmark.publication.reports import REPORT_SECTIONS


def test_reports_include_required_sections_and_manifest_traceability(
    tmp_path: Path,
) -> None:
    """Every report should cite all provenance fields and source artifacts."""
    data = PublicationData.load()

    paths = generate_reports(data, tmp_path)

    assert {path.name for path in paths} == {
        "evaluation_summary.md",
        "experiment_report.md",
        "benchmark_report.md",
    }
    for path in paths:
        report = path.read_text(encoding="utf-8")
        assert all(f"## {section}" in report for section in REPORT_SECTIONS)
        assert "artifacts/experiments/experiment_manifest.json" in report
        assert data.manifest["repository_version"] in report
        assert data.manifest["configuration_hash"] in report
        assert data.manifest["git_commit"] in report
        assert data.manifest["dataset_fingerprint"] in report
        for value in data.benchmark_results["RMSE"].astype(str):
            assert value in report
