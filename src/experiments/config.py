"""Typed settings shared by fixed-split training and cross-validation factories."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from ..evaluation.k_fold import TrainingConfig


@dataclass(frozen=True)
class DatasetSpec:
    """One normalized cohort or legacy modality with explicit annotation overrides.

    available_labels contains source-mask values, not output channel indexes.
    None reads normalized annotation metadata; legacy data requires an override.
    """

    name: str = "D1"
    layout: Literal["normalized", "legacy"] = "normalized"
    modality: str | None = None
    sub_folder: str | None = None
    available_labels: tuple[int, ...] | None = None
    sham: bool = False


@dataclass(frozen=True)
class DataConfig:
    """Dataset root and registrations; no dataset is included implicitly."""

    root: Path
    datasets: tuple[DatasetSpec, ...] = (DatasetSpec(),)
    image_name: str = "N4"
    roi_name: str = "Labels"
    brain_name: str = "brain_mask"


@dataclass(frozen=True)
class LabelConfig:
    """Source values mapped to contiguous classes, including background zero."""

    mapping: dict[int, int] = field(default_factory=lambda: {0: 0, 1: 1, 3: 2, 13: 2, 21: 3})
    names: tuple[str, ...] = ("Background", "Lesion", "Lateral ventricles", "Third ventricle")
    unknown: Literal["error", "background"] = "error"

    def validate(self) -> None:
        """Reject ambiguous class definitions before any images are processed."""
        values = [value for pair in self.mapping.items() for value in pair]
        if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
            raise ValueError("Label mapping keys and values must be integers")

        expected = set(range(len(self.names)))
        if len(self.names) < 2 or set(self.mapping.values()) != expected:
            raise ValueError("Label mapping must cover contiguous classes matching names")

        if self.mapping.get(0) != 0 or self.mapping.get(1) != 1:
            raise ValueError("Keep source labels 0 and 1 mapped to 0 and 1 for shared brain-mask preprocessing")

        if not all(self.names) or len(set(self.names)) != len(self.names):
            raise ValueError("Class names must be nonempty and unique")
        if self.unknown not in ("error", "background"):
            raise ValueError("Unknown labels must raise an error or map explicitly to background")


@dataclass(frozen=True)
class PreprocessingConfig:
    """Frozen reference and geometry, with augmentation confined to training."""

    reference: Path
    patch_size: tuple[int, int, int] = (80, 80, 80)
    spacing_mm: tuple[float, float, float] = (0.1, 0.1, 0.1)
    correct_units: bool = True
    normalization: Literal["zscore", "minmax"] = "zscore"
    augment: bool = True
    random_crop: bool = False
    rotation_range: tuple[float, float] = (-5.0, 5.0)
    scale_range: tuple[float, float] = (0.95, 1.05)
    augmentation_probability: float = 0.3
    flip_probability: float = 0.5


@dataclass(frozen=True)
class ModelConfig:
    """Existing builder options; architecture and dropout defaults stay unchanged."""

    architecture: Literal["3drnet", "unet"] = "3drnet"
    filters: tuple[int, int, int, int, int] = (16, 32, 64, 128, 256)
    residual: bool | None = None
    attention: bool | None = None
    max_pooling: bool = True
    classifier: bool = False


@dataclass(frozen=True)
class LossConfig:
    """ROI objective and historical brain-head loss parameters."""

    regions: Literal["dice_ce", "dice_ce_gradient_fix"] = "dice_ce"
    gamma: float = 0.8
    smooth: float = 1e-5
    batch_wise: bool = True
    brain_gamma: float = 0.5
    brain_alpha: float = 0.6


@dataclass(frozen=True)
class InferenceConfig:
    """Optional held-out full-volume ROI evaluation, without implicit postprocessing."""

    enabled: bool = False
    stride: int = 40
    save_predictions: bool = True


@dataclass(frozen=True)
class ExperimentConfig:
    """One experiment; split randomness is independent of training randomness."""

    data: DataConfig
    preprocessing: PreprocessingConfig
    training: TrainingConfig
    labels: LabelConfig = field(default_factory=LabelConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    learning_rate: float = 0.001
    split_seed: int = 42
    early_stopping_patience: int = 20

    def validate(self) -> None:
        """Fail early on incompatible settings or inaccessible input locations."""
        self.labels.validate()
        self.training.validate(n_subjects=2, cross_validation=False)
        if not Path(self.data.root).is_dir():
            raise FileNotFoundError(self.data.root)
        if not Path(self.preprocessing.reference).is_file():
            raise FileNotFoundError(self.preprocessing.reference)

        if not self.data.datasets:
            raise ValueError("Select at least one dataset")
        for dataset in self.data.datasets:
            if dataset.layout not in ("normalized", "legacy"):
                raise ValueError("Unsupported dataset layout")
            if dataset.layout == "legacy" and (not dataset.modality or dataset.available_labels is None):
                raise ValueError("Legacy datasets need modality and explicit available_labels")

        shape = self.preprocessing.patch_size
        invalid_shape = any(
            isinstance(size, bool) or not isinstance(size, int) or size <= 0 or size % 16
            for size in shape
        )
        if len(shape) != 3 or invalid_shape:
            raise ValueError("Patch dimensions must be positive multiples of 16")

        spacing = self.preprocessing.spacing_mm
        if len(spacing) != 3 or any(not math.isfinite(value) or value <= 0 for value in spacing):
            raise ValueError("Spacing must contain three positive finite millimetre values")

        if self.model.architecture not in ("3drnet", "unet"):
            raise ValueError("Unsupported architecture")
        invalid_filters = any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in self.model.filters
        )
        if len(self.model.filters) != 5 or invalid_filters:
            raise ValueError("Supply five positive integer filter widths")
        if self.model.classifier and self.model.architecture == "unet":
            raise ValueError("The existing U-Net builder does not include a classifier")
        if self.loss.regions not in ("dice_ce", "dice_ce_gradient_fix"):
            raise ValueError("Unsupported ROI loss")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("Learning rate must be positive and finite")
        if not 0 < self.loss.smooth < 0.5:
            raise ValueError("Loss smoothing must be between zero and 0.5")
        for value in (self.loss.gamma, self.loss.brain_gamma, self.loss.brain_alpha,
                      self.preprocessing.augmentation_probability, self.preprocessing.flip_probability):
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("Loss weights and probabilities must be between zero and one")

        seeds = (self.split_seed, self.training.seed)
        if any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in seeds):
            raise ValueError("Seeds must be nonnegative integers")
        if not isinstance(self.early_stopping_patience, int) or self.early_stopping_patience < 0:
            raise ValueError("Early stopping patience must be nonnegative")
        if self.training.infer_steps_from_ids:
            raise ValueError("Use generator cardinality; animal counts do not determine batch counts")
        stride = self.inference.stride
        if isinstance(stride, bool) or not isinstance(stride, int) or not 1 <= stride <= min(shape):
            raise ValueError("Inference stride must be between one and the smallest patch dimension")

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible settings, with absolute paths and integer mappings."""
        settings = asdict(self)
        settings["data"]["root"] = str(Path(self.data.root).resolve())
        settings["preprocessing"]["reference"] = str(Path(self.preprocessing.reference).resolve())
        return settings

    def fingerprint(self) -> str:
        """Hash the serialized configuration to identify precisely configured runs."""
        settings = self.to_dict()
        payload = json.dumps(settings, sort_keys=True, default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()
