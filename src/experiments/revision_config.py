"""Visible configuration for the sequential reviewer experiment suite."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Literal

from .config import (
    DataConfig, DatasetSpec, EvaluationPatchConfig, ExperimentConfig,
    InferenceConfig, LabelConfig, LossConfig, ModelConfig, PostprocessingConfig,
    PreprocessingConfig,
)
from ..evaluation.k_fold import TrainingConfig


@dataclass(frozen=True)
class DraftConfig:
    """Small real-data copies and networks; these are never manuscript results.

    Supplied split files retain all their animals and folds. Otherwise the
    draft selects a small, coverage-balanced animal subset before making folds.
    Every selected longitudinal scan is retained.
    """

    folds: int = 2
    d1_animals: int = 6
    sham_animals: int = 6
    rat_animals: int = 3
    external_animals: int = 2
    # Four downsampling stages leave a 2-cube bottleneck. A 16-cube input
    # leaves only one value per BatchNorm channel when the draft batch is one.
    patch_size: tuple[int, int, int] = (32, 32, 32)
    filters: tuple[int, int, int, int, int] = (2, 4, 8, 16, 32)
    epochs: int = 1
    steps_per_epoch: int = 1


@dataclass(frozen=True)
class RevisionSuiteConfig:
    """One reproducible suite, with explicit animals, dependencies and budgets.

    IDs are canonical animal IDs, for example D1/SUBJ_0001, not scan names.
    folds_file accepts either a complete D1 plan or a joint D1+D3 plan.
    anchor_split_file instead fixes train/validation/test in fold zero and
    generates the remaining outer folds. No saved membership is overwritten.
    All mouse fits start from scratch; only independent rat fine-tuning loads
    a D1 checkpoint. Optional MouseFull pretraining is confined to notebook 05.
    """

    data_root: Path
    reference: Path
    output_root: Path = Path("results/revision_suite")
    run_name: str = "revision_v1"
    seeds: tuple[int, ...] = (42,)
    split_seed: int = 42
    folds: int = 5
    validation_fraction: float = 0.2

    d1_ids: tuple[str, ...] | None = None
    sham_ids: tuple[str, ...] | None = None
    rat_ids: tuple[str, ...] | None = None
    external_ids: tuple[str, ...] | None = None
    animal_ids: dict[str, str] = field(default_factory=dict)
    folds_file: Path | None = None
    anchor_split_file: Path | None = None
    rat_folds_file: Path | None = None
    reference_splits: Path | None = None
    reference_fold: int = 0
    ablation_mode: Literal["fixed_split", "cross_validation"] = "fixed_split"

    d1_epochs: int = 100
    mouse_full_epochs: int = 300
    rat_epochs: int = 100
    interpolation_epochs: int = 10
    interpolation_steps: int | None = 4
    interpolation_orders: tuple[int, ...] = (0, 1, 2, 3)
    architecture_ablations: tuple[str, ...] = (
        "no_attention", "no_residual", "no_multiscale", "no_deep_supervision",
    )
    supervision_controls: tuple[str, ...] = (
        "no_classifier", "missing_as_background", "missing_as_background_no_classifier",
    )
    include_interpolation: bool = True
    include_rats: bool = True
    include_release: bool = True
    include_external: bool = True
    include_annotation_volumes: bool = True
    inter_rater_manifest: Path | None = None
    annotation_spatial_unit: str | None = None
    external_results: dict[str, Path] = field(default_factory=dict)

    filters: tuple[int, int, int, int, int] = (16, 32, 64, 128, 256)
    patch_size: tuple[int, int, int] = (80, 80, 80)
    rat_patch_size: tuple[int, int, int] = (112, 112, 112)
    spacing_mm: tuple[float, float, float] = (0.1, 0.1, 0.1)
    batch_size: int = 8
    rat_batch_size: int = 4
    data_multiplier: int = 3
    learning_rate: float = 0.001
    mixed_precision: bool = False
    save_predictions: bool = True
    bootstrap_samples: int = 2000
    release_seed: int = 42
    rat_source_fold: int = 0
    draft: DraftConfig = field(default_factory=DraftConfig)

    def validate(self) -> None:
        """Reject conflicting split sources and invalid budgets before writing."""
        import re

        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", self.run_name):
            raise ValueError("run_name must be a safe directory name")
        if self.folds_file is not None and self.anchor_split_file is not None:
            raise ValueError("Choose folds_file or anchor_split_file, not both")
        if self.ablation_mode not in ("fixed_split", "cross_validation"):
            raise ValueError("Ablation mode must be fixed_split or cross_validation")
        if type(self.reference_fold) is not int or self.reference_fold < 0:
            raise ValueError("reference_fold must be a nonnegative integer")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("Provide distinct training seeds")
        if any(type(seed) is not int or seed < 0 for seed in self.seeds):
            raise ValueError("Training seeds must be nonnegative integers")
        if self.release_seed not in self.seeds:
            raise ValueError("release_seed must be one of the configured training seeds")
        if self.include_external and not self.include_release:
            raise ValueError("External evaluation requires the final release stage")
        if not 0 < self.validation_fraction < 1:
            raise ValueError("validation_fraction must be between zero and one")
        counts = (
            self.folds, self.d1_epochs, self.mouse_full_epochs, self.rat_epochs,
            self.interpolation_epochs, self.batch_size, self.rat_batch_size,
            self.data_multiplier, self.draft.folds, self.draft.epochs,
            self.draft.steps_per_epoch, self.draft.d1_animals, self.draft.sham_animals,
            self.draft.rat_animals, self.draft.external_animals,
        )
        if any(type(value) is not int or value < 1 for value in counts):
            raise ValueError("Epoch, batch, fold and draft counts must be positive integers")
        if self.folds < 2 or self.draft.folds < 2:
            raise ValueError("Cross-validation requires at least two folds")
        draft_shape = self.draft.patch_size
        if len(draft_shape) != 3 or any(type(size) is not int or size < 16 or size % 16 for size in draft_shape):
            raise ValueError("Draft patch dimensions must be positive multiples of 16")
        if all(size == 16 for size in draft_shape):
            raise ValueError("Draft BatchNorm needs more than one bottleneck voxel; use patch_size=(32, 32, 32)")
        if self.bootstrap_samples < 100:
            raise ValueError("bootstrap_samples must be at least 100")
        if not self.interpolation_orders or len(set(self.interpolation_orders)) != len(self.interpolation_orders):
            raise ValueError("Provide distinct interpolation orders")
        if any(type(order) is not int or order not in (0, 1, 2, 3) for order in self.interpolation_orders):
            raise ValueError("The interpolation study supports orders 0, 1, 2 and 3")
        supported_architecture = {
            "no_attention", "no_residual", "no_multiscale", "no_deep_supervision",
            "strided_downsampling",
        }
        supported_supervision = {
            "no_classifier", "missing_as_background", "missing_as_background_no_classifier",
        }
        for selected, supported in (
            (self.architecture_ablations, supported_architecture),
            (self.supervision_controls, supported_supervision),
        ):
            if len(selected) != len(set(selected)) or not set(selected).issubset(supported):
                raise ValueError("Unknown or duplicated suite variant")
        for ids in (self.d1_ids, self.sham_ids, self.rat_ids, self.external_ids):
            if ids is not None and (not ids or len(ids) != len(set(ids))):
                raise ValueError("Animal selections must be nonempty and unique")
        for name in self.external_results:
            parts = name.split("/")
            if len(parts) != 2 or parts[0] not in ("d1", "mouse_full", "rats"):
                raise ValueError("External result keys must be task/variant, for example d1/nnunet")
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", parts[1]):
                raise ValueError("External comparator names must be simple variant names")
            reserved = {"reference", *self.architecture_ablations, *self.supervision_controls}
            if parts[1] in reserved:
                raise ValueError("External comparator names must not duplicate suite variants")
        for path in (self.folds_file, self.anchor_split_file, self.rat_folds_file, self.inter_rater_manifest):
            if path is not None and not Path(path).is_file():
                raise FileNotFoundError(path)
        if self.folds_file is None and self.anchor_split_file is None and self.reference_splits is not None:
            if not Path(self.reference_splits).is_file():
                raise FileNotFoundError(self.reference_splits)


def experiment_configuration(
    settings: RevisionSuiteConfig, task: Literal["d1", "mouse_full", "rats"],
    *, output_dir: Path, mode: Literal["full", "draft"] = "full",
) -> ExperimentConfig:
    """Create the same task defaults used by the training notebooks."""
    is_mouse_full: bool = task == "mouse_full"
    is_rat: bool = task == "rats"
    dataset_names: tuple[str, ...] = ("D1",)
    if is_mouse_full:
        dataset_names = ("D1", "D3")
    elif is_rat:
        dataset_names = ("D2",)
    datasets = tuple(DatasetSpec(name=name) for name in dataset_names)
    overrides = {
        key: value for key, value in settings.animal_ids.items()
        if key.split("/", 1)[0] in dataset_names
    }
    data = DataConfig(root=settings.data_root, datasets=datasets, animal_ids=overrides)

    epochs: int = settings.d1_epochs
    patch_size = settings.patch_size
    batch_size: int = settings.batch_size
    flip_probability: float = 0.5
    if is_mouse_full:
        epochs = settings.mouse_full_epochs
        flip_probability = 0.0
    elif is_rat:
        epochs = settings.rat_epochs
        patch_size = settings.rat_patch_size
        batch_size = settings.rat_batch_size

    preprocessing = PreprocessingConfig(
        reference=settings.reference, patch_size=patch_size, spacing_mm=settings.spacing_mm,
        flip_probability=flip_probability,
    )
    model = ModelConfig(filters=settings.filters, classifier=is_mouse_full)
    labels = LabelConfig()
    loss = LossConfig(regions="dice_ce", gradient_fix_version=1)
    if is_mouse_full:
        # Explicit source IDs preserve the normalized dataset's original vocabulary.
        sources = (0, 1, 2, 3, 5, 6, 12, 13, 15, 16, 21)
        mapping = {source: channel for channel, source in enumerate(sources)}
        names = (
            "Background", "Lesion", "Contra CC", "Contra ventricle", "Contra hippocampus",
            "Contra cortex", "Ipsi CC", "Ipsi ventricle", "Ipsi hippocampus", "Ipsi cortex",
            "Third ventricle",
        )
        labels = LabelConfig(mapping=mapping, names=names)
        loss = LossConfig(regions="dice_ce_gradient_fix", gradient_fix_version=1)

    postprocessing = PostprocessingConfig(
        clean_brain_mask=True, split_ventricles=not is_mouse_full, use_centroids=True,
    )
    training = TrainingConfig(
        model_name=task, k=settings.folds, epochs=epochs, batch_size=batch_size,
        data_multiplier=settings.data_multiplier, default_callbacks=False,
        output_dir=str(output_dir), fit_kwargs={"verbose": 1, "workers": 0},
    )
    evaluation_patches = EvaluationPatchConfig(patches_per_scan=3)
    if is_rat:
        evaluation_patches = replace(evaluation_patches, patches_per_scan=5)
    stride: int = patch_size[0] // 2

    if mode == "draft":
        patch_size = settings.draft.patch_size
        preprocessing = replace(preprocessing, patch_size=patch_size, correct_units=False)
        model = replace(model, filters=settings.draft.filters)
        training = replace(
            training, k=settings.draft.folds, epochs=settings.draft.epochs,
            batch_size=1, data_multiplier=1, steps_per_epoch=settings.draft.steps_per_epoch,
            fit_kwargs={"verbose": 0, "workers": 0},
        )
        evaluation_patches = EvaluationPatchConfig(patches_per_scan=1, sampling="center")
        postprocessing = replace(
            postprocessing, split_ventricles=False, holes_max_area=0, object_min_area=0,
        )
        stride = patch_size[0]

    original_label_ids: bool = True
    if mode == "draft" and not is_mouse_full:
        # A one-step model may have no third-ventricle prediction. Do not invent
        # a splitting plane merely to make a smoke check run. Merged draft
        # outputs therefore retain their explicitly saved training-channel IDs.
        original_label_ids = False
    inference = InferenceConfig(
        enabled=True, stride=stride, save_predictions=settings.save_predictions,
        original_label_ids=original_label_ids, postprocessing=postprocessing,
    )
    return ExperimentConfig(
        data=data, preprocessing=preprocessing, training=training, labels=labels,
        model=model, loss=loss, inference=inference, evaluation_patches=evaluation_patches,
        learning_rate=settings.learning_rate, split_seed=settings.split_seed,
        cv_validation_fraction=settings.validation_fraction,
    )
