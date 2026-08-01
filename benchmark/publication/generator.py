"""End-to-end generation of publication assets from validated Batch 2 inputs."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from benchmark.constants import PROJECT_ROOT
from benchmark.publication.figures import generate_figures
from benchmark.publication.inputs import PublicationData, PublicationInputPaths
from benchmark.publication.reports import generate_reports
from benchmark.publication.tables import generate_table5

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class PublicationOutputPaths:
    """Directories owned by the publication asset generator."""

    tables: Path
    figures: Path
    reports: Path

    @classmethod
    def repository_defaults(cls) -> PublicationOutputPaths:
        """Return repository publication output directories."""
        artifacts = PROJECT_ROOT / "artifacts"
        return cls(
            tables=artifacts / "tables",
            figures=artifacts / "figures",
            reports=artifacts / "reports",
        )


class PublicationAssetGenerator:
    """Generate tables, figures, captions, and reports from one input snapshot."""

    def __init__(
        self,
        input_paths: PublicationInputPaths | None = None,
        output_paths: PublicationOutputPaths | None = None,
    ) -> None:
        """Initialize strict source and destination paths."""
        self.input_paths = input_paths or PublicationInputPaths.repository_defaults()
        self.output_paths = output_paths or PublicationOutputPaths.repository_defaults()

    def generate(self) -> tuple[Path, ...]:
        """Load sources once and generate the complete Batch 3 asset set."""
        data = PublicationData.load(self.input_paths)
        outputs = (
            *generate_table5(data, self.output_paths.tables),
            *generate_figures(data, self.output_paths.figures),
            *generate_reports(data, self.output_paths.reports),
        )
        LOGGER.info("Generated all publication assets count=%d", len(outputs))
        return outputs
