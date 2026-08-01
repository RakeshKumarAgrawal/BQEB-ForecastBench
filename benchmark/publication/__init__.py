"""Publication asset generation from immutable benchmark artifacts."""

from benchmark.publication.figures import (
    generate_figures,
    performance_values,
    prediction_values,
)
from benchmark.publication.generator import (
    PublicationAssetGenerator,
    PublicationOutputPaths,
)
from benchmark.publication.inputs import PublicationData, PublicationInputPaths
from benchmark.publication.reports import generate_reports
from benchmark.publication.tables import generate_table5

__all__ = [
    "PublicationAssetGenerator",
    "PublicationData",
    "PublicationInputPaths",
    "PublicationOutputPaths",
    "generate_figures",
    "generate_reports",
    "generate_table5",
    "performance_values",
    "prediction_values",
]
