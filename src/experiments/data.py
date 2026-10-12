"""Animal-level dataset preparation and stage-specific preprocessing factories."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import nibabel as nib
import numpy as np
import pandas as pd

from .config import ExperimentConfig
from ..evaluation.k_fold import SubjectIndex
from ..utils.loader import Rodent, RodentDatasets

if TYPE_CHECKING:
    from ..preprocessing.generator import MultiTaskGenerator
    from ..preprocessing.preprocessor import Preprocessor


# Normalized schema v3 uses these anatomical abbreviations, not network indexes.
ANNOTATED_VALUES: dict[str, int] = {
    "Les": 1, "cl-CC": 2, "il-CC": 12, "cl-V": 3, "il-V": 13,
    "cl-Hipp": 5, "il-Hipp": 15, "cl-Ctx": 6, "il-Ctx": 16, "tV": 21,
}


@dataclass(frozen=True)
class SubjectSplit:
    """Disjoint animals used for fitting, checkpoint selection, and final testing."""

    train: tuple[str, ...]
    validation: tuple[str, ...]
    test: tuple[str, ...]

    def validate(self, index: SubjectIndex) -> None:
        """Require exact coverage of selected animals, without duplicated scans."""
        groups = (self.train, self.validation, self.test)
        if any(not group for group in groups):
            raise ValueError("Training, validation and test must each contain animals")

        selected = [subject for group in groups for subject in group]
        if len(selected) != len(set(selected)):
            raise ValueError("Split contains duplicated or overlapping animals")
        if set(selected) != set(index.subject_ids):
            raise ValueError("Split must cover exactly the registered animals")

        index.scan_keys(selected)

    def save(self, path: Path) -> None:
        """Persist identities, independent of generator shuffling or sampling."""
        payload = {"train": self.train, "validation": self.validation, "test": self.test}
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path, index: SubjectIndex) -> SubjectSplit:
        """Read and validate a previously selected animal split."""
        payload = json.loads(path.read_text(encoding="utf-8"))
        split = cls(
            train=tuple(payload["train"]),
            validation=tuple(payload["validation"]),
            test=tuple(payload["test"]),
        )
        split.validate(index)
        return split


def make_split(
    index: SubjectIndex, *, seed: int, validation_fraction: float = 0.2,
    test_fraction: float = 0.1,
) -> SubjectSplit:
    """Split animals using fractions of the complete cohort, without stratification.

    For longitudinal/small cohorts or matched manuscript experiments, load a
    scientifically selected split instead. The local RNG does not alter training.
    """
    fractions = (validation_fraction, test_fraction)
    if any(not np.isfinite(value) or not 0 < value < 1 for value in fractions):
        raise ValueError("Split fractions must be finite and between zero and one")
    if sum(fractions) >= 1:
        raise ValueError("Validation and test fractions must leave training animals")

    ids = np.asarray(index.subject_ids)
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(ids)
    count: int = len(ids)
    n_validation: int = max(1, int(np.ceil(count * validation_fraction)))
    n_test: int = max(1, int(np.ceil(count * test_fraction)))
    if n_validation + n_test >= count:
        raise ValueError("Too few animals for the requested three-way split")

    test = tuple(sorted(shuffled[:n_test].tolist()))
    validation = tuple(sorted(shuffled[n_test:n_test + n_validation].tolist()))
    train = tuple(sorted(shuffled[n_test + n_validation:].tolist()))
    split = SubjectSplit(train=train, validation=validation, test=test)
    split.validate(index)
    return split


def annotation_values(scan: Rodent) -> set[int]:
    """Resolve schema-v3 annotation names; never infer availability from voxels.

    label_mapping describes a vocabulary and is not an annotation declaration.
    In particular SHAM metadata may include Lesion in that vocabulary while
    has_lesion_annotation is false.
    """
    names = scan.metadata.get("annotated_regions")
    if not isinstance(names, list) or any(not isinstance(name, str) for name in names):
        raise ValueError(f"{scan.id}: supply available_labels or valid annotated_regions")

    unknown = set(names) - ANNOTATED_VALUES.keys()
    if unknown:
        raise ValueError(f"{scan.id}: unknown annotation names {sorted(unknown)}; provide an override")

    available: set[int] = {0}
    for name in names:
        available.add(ANNOTATED_VALUES[name])

    lesion_annotation = scan.metadata.get("has_lesion_annotation")
    if lesion_annotation is True:
        available.add(1)
    elif lesion_annotation is False:
        available.discard(1)

    return available


@dataclass
class PreparedData:
    """Loaded scan proxies and animal identities reusable by every runner."""

    config: ExperimentConfig
    loader: RodentDatasets
    subjects: SubjectIndex

    def split_summary(self, split: SubjectSplit) -> pd.DataFrame:
        """Show animal and scan counts separately, before starting training."""
        split.validate(self.subjects)
        rows: list[dict[str, str | int]] = []
        for role, ids in (("train", split.train), ("validation", split.validation), ("test", split.test)):
            keys = self.subjects.scan_keys(ids)
            rows.append({"Role": role, "Animals": len(ids), "Scans": len(keys)})
        return pd.DataFrame(rows)

    def make_generator(
        self, ids: list[str] | tuple[str, ...], *, training: bool,
        batch_size: int | None = None,
    ) -> MultiTaskGenerator:
        """Expand animals to scans; any repetition applies exclusively to training."""
        from ..preprocessing.generator import MultiTaskGenerator

        config = self.config
        keys = self.subjects.scan_keys(ids)
        stage: Literal["training", "validation", "inference"] = "validation"
        if training:
            stage = "training"
            keys = keys * config.training.data_multiplier

        pipeline = make_preprocessor(config, stage=stage)
        settings = {
            "dataset_path": str(config.data.root),
            "labels": list(config.labels.mapping),
            "label_mapping": config.labels.mapping,
            "input_shape": config.preprocessing.patch_size,
            "in_channels": 1,
            "num_classes": len(config.labels.names),
            "batch_size": config.training.batch_size,
        }
        partial_labels: bool = config.loss.regions == "dice_ce_gradient_fix"
        return MultiTaskGenerator(
            ids=keys, loader=self.loader, config=settings, shuffle=training,
            preprocessor=pipeline, drop_last=False, partial_labels=partial_labels,
            with_classifier=config.model.classifier, batch_size=batch_size,
        )


def prepare_data(config: ExperimentConfig) -> PreparedData:
    """Register explicit datasets and resolve missing source labels per scan."""
    config.validate()
    labels = list(config.labels.mapping)
    loader = RodentDatasets(labels=labels)

    for dataset in config.data.datasets:
        previous = set(loader.get_subjects_list())
        if dataset.layout == "normalized":
            loader.add_normalized_dataset(
                dataset_path=config.data.root, dataset=dataset.name,
                modality=dataset.modality, image_name=config.data.image_name,
                roi_mask_name=config.data.roi_name, brain_mask_name=config.data.brain_name,
            )
        else:
            loader.add_dataset(
                dataset_path=config.data.root, modality=dataset.modality,
                sub_folder=dataset.sub_folder, image_suffix=config.data.image_name,
                roi_mask_suffix=config.data.roi_name, brain_mask_suffix=config.data.brain_name,
                sham=dataset.sham,
            )

        added = set(loader.get_subjects_list()) - previous
        if not added:
            raise ValueError(f"No scans selected from {dataset.name}")

        for key in added:
            scan = loader.get_subject(key)
            if dataset.available_labels is None:
                available = annotation_values(scan)
            else:
                available = set(dataset.available_labels)
                if 0 not in available:
                    raise ValueError("Explicit available_labels must include background zero")

            scan.unavailable_labels = sorted(set(labels) - available)
            if scan.unavailable_labels and config.loss.regions != "dice_ce_gradient_fix":
                raise ValueError(f"{scan.id}: partial annotations require dice_ce_gradient_fix")

    subjects = SubjectIndex(loader)
    return PreparedData(config=config, loader=loader, subjects=subjects)


def make_preprocessor(
    config: ExperimentConfig, *, stage: Literal["training", "validation", "inference"],
) -> Preprocessor:
    """Reorient first; use linear intensities, nearest masks, and deterministic validation."""
    from ..preprocessing.preprocessor import (
        CorrectX10, Flip, GaussianBlur, MapLabels, Noise, Normalize, PadderCutter,
        Preprocessor, PreprocessingStep, RandomAffine, RandomCropping, Reorient, Resample,
    )

    if stage not in ("training", "validation", "inference"):
        raise ValueError("Unknown preprocessing stage")

    settings = config.preprocessing
    reference = nib.load(str(settings.reference))
    steps: list[PreprocessingStep] = [Reorient(ref_img=reference)]
    if settings.correct_units:
        steps.append(CorrectX10(reference_resolution=settings.spacing_mm))

    steps.append(MapLabels(mapping=config.labels.mapping, unknown=config.labels.unknown))
    resampling = Resample(
        target_resolution=settings.spacing_mm, interpolation=1, orthogonal=True,
    )
    steps.append(resampling)

    if stage == "training" and settings.augment:
        affine = RandomAffine(
            scale_range=settings.scale_range, rotation_range=settings.rotation_range,
            probability=settings.augmentation_probability,
        )
        steps.append(affine)
        steps.append(GaussianBlur(gaussian_blur_range=(0.0, 0.6), probability=settings.augmentation_probability))
        steps.append(Noise(noise_range=(0.0, 0.05), probability=settings.augmentation_probability))

    steps.append(Normalize(mode=settings.normalization))

    if stage != "inference":
        if stage == "training" and settings.random_crop:
            crop = RandomCropping(dim=settings.patch_size, mode="random", padding=True)
        else:
            crop = PadderCutter(img_size=settings.patch_size)
        steps.append(crop)

    if stage == "training" and settings.augment:
        steps.append(Flip(axis_list=[0], probability=settings.flip_probability))

    return Preprocessor(steps=steps)
