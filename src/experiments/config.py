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
    Each available foreground ROI is fully delineated throughout the volume:
    its mask supplies both positive voxels and known negatives outside it.
    An annotated but empty ROI remains available; absence of voxels alone never
    establishes availability. Sparse slices/contours require a spatial validity
    mask and are not supported by this full-volume annotation contract.
    None reads normalized annotation metadata; legacy data requires an override.
    Zero is reserved for the task remainder, not a verified background annotation.
    """

    name: str = "D1"
    layout: Literal["normalized", "legacy"] = "normalized"
    modality: str | None = None
    sub_folder: str | None = None
    available_labels: tuple[int, ...] | None = None
    sham: bool = False
    annotation_scope: Literal["full_volume"] = "full_volume"

    def validate(self) -> None:
        """Require an explicit layout and whole-volume foreground annotation policy."""
        if self.layout not in ("normalized", "legacy"):
            raise ValueError("Unsupported dataset layout")
        if self.layout == "legacy" and (not self.modality or self.available_labels is None):
            raise ValueError("Legacy datasets need modality and explicit available_labels")
        if self.annotation_scope != "full_volume":
            raise ValueError("Only full_volume ROI annotations are supported; sparse annotations need a spatial validity mask")

        if self.available_labels is not None:
            invalid_labels = any(
                isinstance(label, bool) or not isinstance(label, int) or label < 0
                for label in self.available_labels
            )
            if invalid_labels or len(set(self.available_labels)) != len(self.available_labels):
                raise ValueError("available_labels must contain unique nonnegative integer source values")
            if 0 not in self.available_labels:
                raise ValueError("Explicit available_labels must include background zero as the task remainder")


@dataclass(frozen=True)
class FixedSplitConfig:
    """Animal fractions of the complete cohort, defaulting to train/val/test 70/20/10.

    Validation and test counts are rounded up separately; training gets the
    remainder. Saved/manual memberships take precedence over these fractions.
    This policy is for fixed splits; CV inner validation uses its own setting.
    """

    validation_fraction: float = 0.2
    test_fraction: float = 0.1

    @property
    def train_fraction(self) -> float:
        """Return the requested training proportion before integer rounding."""
        return 1 - self.validation_fraction - self.test_fraction

    def validate(self) -> None:
        """Require finite positive holdout fractions that leave training animals."""
        fractions = (self.validation_fraction, self.test_fraction)
        for fraction in fractions:
            if isinstance(fraction, bool) or not isinstance(fraction, (int, float)):
                raise ValueError("Split fractions must be finite and between zero and one")
            if not math.isfinite(fraction) or not 0 < fraction < 1:
                raise ValueError("Split fractions must be finite and between zero and one")

        if sum(fractions) >= 1:
            raise ValueError("Validation and test fractions must leave training animals")


@dataclass(frozen=True)
class DataConfig:
    """Dataset root and registrations; no dataset is included implicitly."""

    root: Path
    datasets: tuple[DatasetSpec, ...] = (DatasetSpec(),)
    image_name: str = "N4"
    roi_name: str = "Labels"
    brain_name: str = "brain_mask"
    animal_ids: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class LabelConfig:
    """Map selected source labels to classes; project extra anatomy to background.

    Set unknown="error" explicitly when every source label must be declared.
    Background projection does not exclude these voxels from loss computation.
    In partial annotations, zero may contain unavailable anatomy. Foreground
    availability and background supervision are separate loss-policy decisions.
    """

    mapping: dict[int, int] = field(default_factory=lambda: {0: 0, 1: 1, 3: 2, 13: 2, 21: 3})
    names: tuple[str, ...] = ("Background", "Lesion", "Lateral ventricles", "Third ventricle")
    unknown: Literal["error", "background"] = "background"

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
    """Frozen geometry with centre-biased random crops enabled for training.

    crop_mode selects the sampling distribution, with automatic standard deviation.
    Setting random_crop=False uses a fixed training crop. Validation/test use
    the separate, saved EvaluationPatchConfig collection.
    Flipping is off by default; D1 and rat notebooks explicitly enable anatomical LR flips.
    Intensity augmentation normally precedes resampling on the native grid.
    The resampled setting retains the earlier affine/blur/noise order for saved
    runs. Blur bounds describe sigma; noise bounds describe standard deviation.
    """

    reference: Path
    patch_size: tuple[int, int, int] = (80, 80, 80)
    spacing_mm: tuple[float, float, float] = (0.1, 0.1, 0.1)
    correct_units: bool = True
    normalization: Literal["zscore", "minmax"] = "zscore"
    augment: bool = True
    random_crop: bool = True
    crop_mode: Literal["center", "random"] = "center"
    rotation_range: tuple[float, float] = (-5.0, 5.0)
    scale_range: tuple[float, float] = (0.95, 1.05)
    augmentation_probability: float = 0.3
    flip_probability: float = 0.0
    intensity_augmentation_grid: Literal["native", "resampled"] = "native"
    blur_sigma_range: tuple[float, float] = (0.0, 0.6)
    blur_sigma_units: Literal["voxels", "mm"] = "voxels"
    noise_std_range: tuple[float, float] = (0.0, 0.05)
    noise_std_units: Literal["intensity", "image_std"] = "intensity"

    def validate(self) -> None:
        """Validate acquisition-style augmentation ranges and their declared units."""
        if self.intensity_augmentation_grid not in ("native", "resampled"):
            raise ValueError("Intensity augmentation grid must be native or resampled")
        if self.blur_sigma_units not in ("voxels", "mm"):
            raise ValueError("Blur sigma units must be voxels or mm")
        if self.noise_std_units not in ("intensity", "image_std"):
            raise ValueError("Noise standard deviation units must be intensity or image_std")

        ranges = (("Blur sigma", self.blur_sigma_range), ("Noise standard deviation", self.noise_std_range))
        for name, bounds in ranges:
            if len(bounds) != 2:
                raise ValueError(f"{name} range must contain two nonnegative finite ordered bounds")
            invalid_bounds = any(
                isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
                for value in bounds
            )
            if invalid_bounds or bounds[0] < 0 or bounds[0] > bounds[1]:
                raise ValueError(f"{name} range must contain two nonnegative finite ordered bounds")


@dataclass(frozen=True)
class ModelConfig:
    """Explicit builder flags, defaulting to the original 3DR-Net settings."""

    architecture: Literal["3drnet", "unet"] = "3drnet"
    filters: tuple[int, int, int, int, int] = (16, 32, 64, 128, 256)
    residual: bool = True
    attention: bool = True
    max_pooling: bool = True
    classifier: bool = False

    def __post_init__(self) -> None:
        """Reject non-boolean component flags rather than inferring their meaning."""
        flags = (self.residual, self.attention, self.max_pooling, self.classifier)
        if any(not isinstance(flag, bool) for flag in flags):
            raise ValueError("Model component flags must be booleans")


@dataclass(frozen=True)
class EvaluationPatchConfig:
    """Validation/test crops sampled once, independently of training randomness.

    Left-right pairs double the samples and reflect MRI and both masks together.
    Keep pairs disabled for MouseFull. Training data_multiplier is independent.
    """

    patches_per_scan: int = 3
    sampling: Literal["center_biased", "center"] = "center_biased"
    seed: int = 42
    left_right_pairs: bool = False

    def validate(self) -> None:
        """Require a reproducible, nonempty patch collection."""
        if isinstance(self.patches_per_scan, bool) or not isinstance(self.patches_per_scan, int) or self.patches_per_scan < 1:
            raise ValueError("Evaluation patches_per_scan must be a positive integer")

        if self.sampling not in ("center_biased", "center"):
            raise ValueError("Evaluation sampling must be center_biased or center")

        if self.sampling == "center" and self.patches_per_scan != 1:
            raise ValueError("A fixed center crop requires patches_per_scan=1")

        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("Evaluation patch seed must be a nonnegative integer")

        if not isinstance(self.left_right_pairs, bool):
            raise ValueError("Evaluation left_right_pairs must be boolean")


@dataclass(frozen=True)
class LossConfig:
    """ROI objective and historical brain-head loss parameters.

    gradient_fix_version=2 selects foreground-only Dice/CE with full-volume ROI
    negatives. Version 1 identifies older saved objectives for evaluation and
    initialization only; it cannot silently continue training under version 2.
    """

    regions: Literal["dice_ce", "dice_ce_gradient_fix"] = "dice_ce"
    gamma: float = 0.8
    smooth: float = 1e-5
    batch_wise: bool = True
    brain_gamma: float = 0.5
    brain_alpha: float = 0.6
    gradient_fix_version: Literal[1, 2] = 2

    def validate_for_training(self) -> None:
        """Require a new run when the saved partial-label objective is obsolete."""
        if self.regions == "dice_ce_gradient_fix" and self.gradient_fix_version != 2:
            raise ValueError("Partial-label loss objective changed; start a new version-2 run. Older weights remain usable for evaluation or initialization")


@dataclass(frozen=True)
class PostprocessingConfig:
    """Historical full-volume operations; morphology thresholds are voxel counts.

    Split ventricles only for the four-class merged D1/rat vocabulary.
    MouseFull already predicts laterality and must keep this option disabled.
    The historical default hemisphere selects the side of array origin, not
    an affine-derived anatomical side. A missing predicted third ventricle
    raises an error when a symmetry plane is needed; no side is guessed.
    """

    clean_brain_mask: bool = False
    fill_small_holes: bool = True
    remove_small_objects: bool = True
    holes_max_area: int = 20000
    object_min_area: int = 30000
    split_ventricles: bool = False
    use_centroids: bool = False
    default_hemisphere: Literal["right", "left"] = "right"
    lesion_voxels_threshold: int = 10

    def __post_init__(self) -> None:
        """Validate options before loading a checkpoint or predicting a volume."""
        flags = (self.clean_brain_mask, self.fill_small_holes,
                 self.remove_small_objects, self.split_ventricles, self.use_centroids)
        if any(not isinstance(value, bool) for value in flags):
            raise ValueError("Postprocessing flags must be booleans")

        counts = (self.holes_max_area, self.object_min_area, self.lesion_voxels_threshold)
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in counts):
            raise ValueError("Postprocessing thresholds must be nonnegative integer voxel counts")
        if self.default_hemisphere not in ("right", "left"):
            raise ValueError("Default hemisphere must be right or left")


@dataclass(frozen=True)
class InferenceConfig:
    """Full-volume evaluation settings, separate from training targets.

    Notebooks explicitly enable original IDs and historical postprocessing.
    False retains consecutive class IDs (postprocessed IDs when splitting).
    Disable postprocessing as well for a raw model-output comparison.
    """

    enabled: bool = False
    stride: int = 40
    save_predictions: bool = True
    original_label_ids: bool = False
    postprocessing: PostprocessingConfig = field(default_factory=PostprocessingConfig)


@dataclass(frozen=True)
class InitializationConfig:
    """Scratch, complete weights, or feature transfer with new output heads.

    In backbone mode, source_experiment reconstructs a source fitting graph for
    weights-only checkpoints. Without it, checkpoint must be a complete model.
    """

    mode: Literal["scratch", "weights", "backbone"] = "scratch"
    checkpoint: Path | None = None
    freeze_backbone: bool = False
    source_experiment: Path | None = None

    def validate(self) -> None:
        """Require an explicit checkpoint for pretrained initialization."""
        if self.mode not in ("scratch", "weights", "backbone"):
            raise ValueError("Unknown initialization mode")
        if not isinstance(self.freeze_backbone, bool):
            raise ValueError("freeze_backbone must be a boolean")
        if self.mode == "scratch":
            if self.checkpoint is not None or self.freeze_backbone or self.source_experiment is not None:
                raise ValueError("Scratch initialization cannot load or freeze pretrained weights")
        elif self.checkpoint is None or not Path(self.checkpoint).is_file():
            raise FileNotFoundError("Pretrained initialization requires an existing checkpoint file")

        if self.source_experiment is not None:
            if self.mode != "backbone":
                raise ValueError("source_experiment is only used for backbone transfer")
            if not Path(self.source_experiment).is_file():
                raise FileNotFoundError(self.source_experiment)

    @classmethod
    def from_run(
        cls, run_dir: Path, *, fold: int = 0, freeze_backbone: bool = False,
    ) -> InitializationConfig:
        """Select committed best weights for backbone transfer, even before fit ends.

        All source fitting epochs and label channels come from that run's saved
        experiment. This selects weights only, never its optimizer or callbacks.
        """
        if isinstance(fold, bool) or not isinstance(fold, int) or fold < 0:
            raise ValueError("Source fold must be a nonnegative integer")

        run_dir = Path(run_dir).resolve()
        source_experiment = run_dir / "experiment.json"
        fold_dir = run_dir / f"fold_{fold}"
        checkpoint = fold_dir / "best.weights.h5"
        state_file = fold_dir / "training_state.json"
        if state_file.is_file():
            state = json.loads(state_file.read_text(encoding="utf-8"))
            checkpoint = fold_dir / state["selected_checkpoint"]

        settings = cls(
            mode="backbone", checkpoint=checkpoint,
            freeze_backbone=freeze_backbone, source_experiment=source_experiment,
        )
        settings.validate()
        return settings


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
    initialization: InitializationConfig = field(default_factory=InitializationConfig)
    learning_rate: float = 0.001
    split_seed: int = 42
    early_stopping_patience: int = 20  # Legacy fallback when a fixed-split caller supplies no callback list.
    cv_validation_fraction: float = 0.2
    fixed_split: FixedSplitConfig = field(default_factory=FixedSplitConfig)
    evaluation_patches: EvaluationPatchConfig = field(default_factory=EvaluationPatchConfig)

    def validate(self) -> None:
        """Fail early on incompatible settings or inaccessible input locations."""
        self.labels.validate()
        self.initialization.validate()
        self.fixed_split.validate()
        self.preprocessing.validate()
        self.evaluation_patches.validate()
        dataset_names = {dataset.name for dataset in self.data.datasets}
        if "D3" in dataset_names and self.evaluation_patches.left_right_pairs:
            raise ValueError("MouseFull evaluation must keep left-right pairs disabled")
        if not math.isfinite(self.cv_validation_fraction) or not 0 < self.cv_validation_fraction < 1:
            raise ValueError("CV validation fraction must be between zero and one")
        self.training.validate(n_subjects=2, cross_validation=False)
        if not Path(self.data.root).is_dir():
            raise FileNotFoundError(self.data.root)
        if not Path(self.preprocessing.reference).is_file():
            raise FileNotFoundError(self.preprocessing.reference)

        if not self.data.datasets:
            raise ValueError("Select at least one dataset")
        for dataset in self.data.datasets:
            dataset.validate()

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
        version = self.loss.gradient_fix_version
        if isinstance(version, bool) or not isinstance(version, int) or version not in (1, 2):
            raise ValueError("Gradient-fix objective version must be one or two")
        if self.preprocessing.crop_mode not in ("center", "random"):
            raise ValueError("Training crop mode must be center or random")
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

        if self.inference.enabled:
            from ..evaluation.postprocessing import OutputLabels

            OutputLabels.resolve(
                self.labels, split_ventricles=self.inference.postprocessing.split_ventricles,
                original_label_ids=self.inference.original_label_ids,
            )

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible settings, with absolute paths and integer mappings."""
        settings = asdict(self)
        settings["data"]["root"] = str(Path(self.data.root).resolve())
        settings["preprocessing"]["reference"] = str(Path(self.preprocessing.reference).resolve())
        if self.initialization.checkpoint is not None:
            settings["initialization"]["checkpoint"] = str(Path(self.initialization.checkpoint).resolve())
        if self.initialization.source_experiment is not None:
            source_experiment = Path(self.initialization.source_experiment).resolve()
            settings["initialization"]["source_experiment"] = str(source_experiment)
        return settings

    def fingerprint(self) -> str:
        """Hash the serialized configuration to identify precisely configured runs."""
        settings = self.to_dict()
        payload = json.dumps(settings, sort_keys=True, default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @classmethod
    def from_dict(cls, settings: dict[str, Any]) -> ExperimentConfig:
        """Restore a saved configuration, including paths, tuples and integer labels.

        Validation remains explicit so inference can replace relocated input paths
        before checking them. Unknown settings raise rather than being discarded.
        """
        values = dict(settings)
        data = dict(values.pop("data"))
        data["root"] = Path(data["root"])
        datasets = []
        for record in data["datasets"]:
            record = dict(record)
            if record.get("available_labels") is not None:
                record["available_labels"] = tuple(record["available_labels"])
            datasets.append(DatasetSpec(**record))
        data["datasets"] = tuple(datasets)

        preprocessing = dict(values.pop("preprocessing"))
        preprocessing["reference"] = Path(preprocessing["reference"])
        # Older saved runs augmented intensities after resampling and affine.
        # Preserve that protocol when loading; new configurations default native.
        preprocessing.setdefault("intensity_augmentation_grid", "resampled")
        for name in ("patch_size", "spacing_mm", "rotation_range", "scale_range"):
            preprocessing[name] = tuple(preprocessing[name])
        for name in ("blur_sigma_range", "noise_std_range"):
            if name in preprocessing:
                preprocessing[name] = tuple(preprocessing[name])

        labels = dict(values.pop("labels"))
        labels["mapping"] = {int(source): channel for source, channel in labels["mapping"].items()}
        labels["names"] = tuple(labels["names"])
        model = dict(values.pop("model"))
        model["filters"] = tuple(model["filters"])
        initialization = dict(values.pop("initialization"))
        if initialization["checkpoint"] is not None:
            initialization["checkpoint"] = Path(initialization["checkpoint"])
        if initialization.get("source_experiment") is not None:
            initialization["source_experiment"] = Path(initialization["source_experiment"])

        inference = dict(values.pop("inference"))
        postprocessing = inference.pop("postprocessing", {})
        inference["postprocessing"] = PostprocessingConfig(**postprocessing)

        fixed_split = FixedSplitConfig(**values.pop("fixed_split", {}))
        patch_settings = values.pop("evaluation_patches", None)
        if patch_settings is None:
            # Saved runs without this field used one fixed center crop.
            patch_settings = {"patches_per_scan": 1, "sampling": "center"}
        evaluation_patches = EvaluationPatchConfig(**patch_settings)
        loss = dict(values.pop("loss"))
        if loss.get("regions") == "dice_ce_gradient_fix" and "gradient_fix_version" not in loss:
            loss["gradient_fix_version"] = 1

        return cls(
            data=DataConfig(**data), preprocessing=PreprocessingConfig(**preprocessing),
            training=TrainingConfig(**values.pop("training")), labels=LabelConfig(**labels),
            model=ModelConfig(**model), loss=LossConfig(**loss),
            inference=InferenceConfig(**inference),
            initialization=InitializationConfig(**initialization), fixed_split=fixed_split,
            evaluation_patches=evaluation_patches, **values,
        )

    @classmethod
    def load(cls, path: Path) -> ExperimentConfig:
        """Read the experiment.json written by either training runner."""
        settings = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(settings)
