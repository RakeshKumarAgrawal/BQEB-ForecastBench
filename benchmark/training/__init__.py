"""Public API for model training orchestration and persistence."""

from benchmark.training.callbacks import (
    CallbackList,
    TrainingCallback,
    TrainingContext,
)
from benchmark.training.checkpoint import (
    Checkpoint,
    CheckpointError,
    latest_checkpoint,
    load_checkpoint,
    save_checkpoint,
)
from benchmark.training.history import (
    TrainingHistory,
    TrainingRecord,
    configuration_hash,
)
from benchmark.training.model_io import (
    ModelArtifact,
    ModelIOError,
    load_model,
    load_model_artifact,
    save_model,
)
from benchmark.training.trainer import (
    ModelTrainer,
    SerializationSettings,
    TrainingArtifactPaths,
    TrainingConfigurationError,
    TrainingSettings,
    load_training_settings,
)

__all__ = [
    "CallbackList",
    "Checkpoint",
    "CheckpointError",
    "ModelArtifact",
    "ModelIOError",
    "ModelTrainer",
    "SerializationSettings",
    "TrainingArtifactPaths",
    "TrainingCallback",
    "TrainingConfigurationError",
    "TrainingContext",
    "TrainingHistory",
    "TrainingRecord",
    "TrainingSettings",
    "configuration_hash",
    "latest_checkpoint",
    "load_checkpoint",
    "load_model",
    "load_model_artifact",
    "load_training_settings",
    "save_checkpoint",
    "save_model",
]
