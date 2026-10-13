"""Animal-level dataset preparation and stage-specific preprocessing factories."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Mapping, Sequence

import nibabel as nib
import numpy as np
import pandas as pd

from .config import ExperimentConfig, FixedSplitConfig
from ..evaluation.k_fold import FoldSplit, SubjectIndex
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
    """Disjoint animals used for fitting, checkpoint selection, and final testing.

    All registered scans of an animal inherit its role. Saved memberships are
    authoritative, even when their proportions differ from generation defaults.
    """

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
        """Read exact memberships without regenerating or enforcing default fractions."""
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
    settings = FixedSplitConfig(
        validation_fraction=validation_fraction, test_fraction=test_fraction,
    )
    settings.validate()
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("Split seed must be a nonnegative integer")

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
    Declared foreground regions are complete masks, including known negatives
    and annotated empty regions. Zero is included for target encoding only;
    it does not establish verified background supervision.
    """
    names = scan.metadata.get("annotated_regions")
    if not isinstance(names, list) or any(not isinstance(name, str) for name in names):
        raise ValueError(f"{scan.id}: supply available_labels or valid annotated_regions")

    if len(names) != len(set(names)):
        raise ValueError(f"{scan.id}: annotated_regions must not contain duplicates")

    unknown = set(names) - ANNOTATED_VALUES.keys()
    if unknown:
        raise ValueError(f"{scan.id}: unknown annotation names {sorted(unknown)}; provide an override")

    available: set[int] = {0}
    for name in names:
        available.add(ANNOTATED_VALUES[name])

    lesion_annotation = scan.metadata.get("has_lesion_annotation")
    if lesion_annotation is not None and not isinstance(lesion_annotation, bool):
        raise ValueError(f"{scan.id}: has_lesion_annotation must be a boolean when declared")
    if lesion_annotation is True:
        available.add(1)
    elif lesion_annotation is False:
        available.discard(1)

    return available


def make_cv_splits(
    subjects: list[str], *, k: int, seed: int, validation_fraction: float = 0.2,
    strata: Mapping[str, str] | None = None,
) -> list[FoldSplit]:
    """Create outer holdouts with optional stratification of biological animals.

    strata maps each selected animal to one group. Stratified folds distribute
    each group's available animals approximately evenly; they do not force a
    50/50 group ratio or oversample animals. Inner validation remains random.
    """
    from sklearn.model_selection import KFold, StratifiedKFold

    if len(subjects) != len(set(subjects)):
        raise ValueError("CV animals must be unique")
    if not np.isfinite(validation_fraction) or not 0 < validation_fraction < 1:
        raise ValueError("Validation fraction must be between zero and one")
    ids = np.asarray(sorted(subjects))
    if strata is None:
        splitter = KFold(n_splits=k, shuffle=True, random_state=seed)
        partitions = splitter.split(ids)
    else:
        if set(strata) != set(subjects):
            raise ValueError("CV strata must cover exactly the selected animals")
        if any(not isinstance(group, str) or not group.strip() for group in strata.values()):
            raise ValueError("CV strata must contain nonempty group names")
        groups = np.asarray([strata[subject] for subject in ids])
        splitter = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
        partitions = splitter.split(ids, groups)

    folds: list[FoldSplit] = []
    for index, (pool_indexes, test_indexes) in enumerate(partitions):
        pool = ids[pool_indexes]
        rng = np.random.default_rng(seed + index)
        shuffled = rng.permutation(pool)
        validation_count = max(1, int(np.ceil(len(pool) * validation_fraction)))
        if validation_count >= len(pool):
            raise ValueError("Too few animals for separate training and checkpoint validation")
        fold = FoldSplit(
            index=index, train_ids=sorted(shuffled[validation_count:].tolist()),
            val_ids=sorted(shuffled[:validation_count].tolist()), test_ids=sorted(ids[test_indexes].tolist()),
        )
        fold.validate(set(subjects))
        folds.append(fold)
    return folds


def make_leave_one_animal_out_splits(subjects: list[str]) -> list[FoldSplit]:
    """Match historical rat CV: select/evaluate one animal, train on all others.

    These two-role folds use the omitted animal for checkpoint selection and
    full-volume validation. They do not provide an independent outer test set.
    Scan expansion remains the responsibility of SubjectIndex and the factories.
    """
    if len(subjects) < 2:
        raise ValueError("Leave-one-animal-out requires at least two animals")
    if any(not isinstance(subject, str) or not subject.strip() for subject in subjects):
        raise ValueError("Animal IDs must be nonempty strings")
    if len(subjects) != len(set(subjects)):
        raise ValueError("CV animals must be unique")

    ids: list[str] = sorted(subjects)
    folds: list[FoldSplit] = []
    for index, animal in enumerate(ids):
        training_ids = [subject for subject in ids if subject != animal]
        fold = FoldSplit(index=index, train_ids=training_ids, val_ids=[animal])
        fold.validate(set(ids))
        folds.append(fold)

    return folds


def fold_summary(folds: list[FoldSplit], subjects: SubjectIndex) -> pd.DataFrame:
    """Show animals and scans separately for all training/validation/holdout roles."""
    rows = []
    for fold in folds:
        for role, ids in (("train", fold.train_ids), ("validation", fold.val_ids), ("holdout", fold.test_ids)):
            rows.append({"Fold": fold.index, "Role": role, "Animals": len(ids),
                         "Scans": len(subjects.scan_keys(ids))})
    return pd.DataFrame(rows)


def fold_annotation_summary(
    folds: list[FoldSplit], data: PreparedData, *, include_lesion: bool = False,
) -> pd.DataFrame:
    """Audit complete/partial animal counts for each split role and outer fold.

    Anatomy-only coverage is the default. A complete animal has every required
    source label declared available in all its registered scans.
    """
    groups = data.annotation_groups(include_lesion=include_lesion)
    rows: list[dict[str, str | int]] = []
    for fold in folds:
        roles = (("train", fold.train_ids), ("validation", fold.val_ids), ("holdout", fold.test_ids))
        for role, ids in roles:
            data.subjects.scan_keys(ids)
            complete = sum(groups[animal] == "complete" for animal in ids)
            rows.append({"Fold": fold.index, "Role": role, "Complete": complete,
                         "Partial": len(ids) - complete})
    return pd.DataFrame(rows)


@dataclass
class PreparedData:
    """Loaded scan proxies and animal identities reusable by every runner."""

    config: ExperimentConfig
    loader: RodentDatasets
    subjects: SubjectIndex

    def make_split(self) -> SubjectSplit:
        """Generate new animal memberships from the experiment's fixed-split policy.

        Load SubjectSplit instead for saved or manuscript memberships. Fractions
        apply to the complete cohort; integer rounding may change realized ratios.
        """
        settings = self.config.fixed_split
        seed = self.config.split_seed
        validation_fraction = settings.validation_fraction
        test_fraction = settings.test_fraction

        return make_split(
            self.subjects, seed=seed,
            validation_fraction=validation_fraction, test_fraction=test_fraction,
        )

    def annotation_groups(
        self, subject_ids: Sequence[str] | None = None, *, include_lesion: bool = False,
    ) -> dict[str, str]:
        """Classify animals by declared coverage, independent of lesion presence.

        By default require selected anatomical classes, excluding background and
        lesion. include_lesion=True also requires lesion annotation. An animal
        with any required label unavailable in any scan is partial. Merged
        classes require all contributing source labels, matching target masking.
        """
        if subject_ids is None:
            subject_ids = self.subjects.subject_ids
        self.subjects.scan_keys(subject_ids)
        excluded_channels = {0}
        if not include_lesion:
            excluded_channels.add(1)
        required_sources = {
            source for source, channel in self.config.labels.mapping.items()
            if channel not in excluded_channels
        }
        if not required_sources:
            raise ValueError("Annotation balancing requires at least one selected region")

        groups: dict[str, str] = {}
        for subject_id in subject_ids:
            group = "complete"
            for key in self.subjects.scan_keys([subject_id]):
                scan = self.loader.get_subject(key)
                unavailable = set(scan.unavailable_labels or ())
                if required_sources & unavailable:
                    group = "partial"
                    break
            groups[subject_id] = group
        return groups

    def contract(self) -> list[dict[str, Any]]:
        """Snapshot identities, annotation availability, geometry and source-file stamps."""
        records = []
        for key in sorted(self.loader.get_subjects_list()):
            scan = self.loader.get_subject(key)
            files = []
            for source in (scan.image_path, scan.labels_path, scan.brain_mask_path):
                if source is not None:
                    path = Path(source).resolve()
                    stamp = path.stat()
                    files.append({"path": str(path), "size": stamp.st_size, "mtime_ns": stamp.st_mtime_ns})
            records.append({
                "scan": list(key), "animal": self.subjects.subject_for_scan(key),
                "metadata": scan.metadata, "unavailable_labels": scan.unavailable_labels,
                "shape": list(scan.image.shape), "affine": scan.image.affine.tolist(), "files": files,
            })
        return records

    def split_summary(self, split: SubjectSplit) -> pd.DataFrame:
        """Show realized animal fractions and scan counts, including saved splits."""
        split.validate(self.subjects)
        rows: list[dict[str, str | int | float]] = []
        total_animals = len(self.subjects.subject_ids)
        for role, ids in (("train", split.train), ("validation", split.validation), ("test", split.test)):
            keys = self.subjects.scan_keys(ids)
            animal_fraction = len(ids) / total_animals
            rows.append({"Role": role, "Animals": len(ids), "AnimalFraction": animal_fraction,
                         "Scans": len(keys)})
        return pd.DataFrame(rows)

    def annotation_summary(self) -> pd.DataFrame:
        """Count declared annotation coverage per dataset and output class.

        Foreground availability means full-volume delineation, including empty
        annotated masks. Background rows describe target encoding, not verified
        background annotation. Merged classes require all contributing sources.
        """
        mapping = self.config.labels.mapping
        rows: list[dict[str, Any]] = []
        keys = self.loader.get_subjects_list()
        scans = [self.loader.get_subject(key) for key in keys]
        datasets = sorted({scan.subject_group[0] for scan in scans})
        for dataset in datasets:
            selected = [scan for scan in scans if scan.subject_group[0] == dataset]
            for channel, name in enumerate(self.config.labels.names):
                sources = [source for source, target in mapping.items() if target == channel]
                unavailable = 0
                for scan in selected:
                    if set(sources) & set(scan.unavailable_labels or ()):
                        unavailable += 1

                meaning = "full-volume ROI, including known negatives"
                if channel == 0:
                    meaning = "task remainder; background supervision is loss-specific"

                rows.append({
                    "Dataset": dataset, "ClassIndex": channel, "Class": name,
                    "AnnotationMeaning": meaning,
                    "SourceLabels": sources, "AnnotatedScans": len(selected) - unavailable,
                    "UnannotatedScans": unavailable, "TotalScans": len(selected),
                })

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


def prepare_data(config: ExperimentConfig, *, require_training_loss: bool = True) -> PreparedData:
    """Register datasets and resolve missing source labels per scan.

    Training requires a partial-label-aware loss for incomplete annotations.
    Offline evaluation can disable that loss check because it does not fit; all
    vocabulary and annotation checks still apply, including missing-label masks.
    """
    config.validate()
    if require_training_loss:
        config.loss.validate_for_training()
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
            declared_scope = scan.metadata.get("annotation_scope", dataset.annotation_scope)
            if declared_scope != dataset.annotation_scope:
                raise ValueError(f"{scan.id}: annotation scope must be full_volume; sparse annotations need a spatial validity mask")
            if dataset.available_labels is None:
                available = annotation_values(scan)
            else:
                available = set(dataset.available_labels)
                if 0 not in available:
                    raise ValueError("Explicit available_labels must include background zero")

            scan.unavailable_labels = sorted(set(labels) - available)
            if require_training_loss and scan.unavailable_labels and config.loss.regions != "dice_ce_gradient_fix":
                raise ValueError(f"{scan.id}: partial annotations require dice_ce_gradient_fix")

    subjects = SubjectIndex(loader)
    if config.data.animal_ids:
        unknown_animals = set(config.data.animal_ids) - set(subjects.subject_ids)
        if unknown_animals:
            raise ValueError(f"Animal identity overrides contain unknown IDs: {sorted(unknown_animals)}")
        overrides = {}
        for original, animal in config.data.animal_ids.items():
            for key in subjects.scan_keys([original]):
                overrides[key] = animal
        subjects = SubjectIndex(loader, subject_overrides=overrides)
    return PreparedData(config=config, loader=loader, subjects=subjects)


def make_preprocessor(
    config: ExperimentConfig, *, stage: Literal["training", "validation", "inference"],
    label_mapping: dict[int, int] | None = None,
) -> Preprocessor:
    """Build shared geometry; optionally retain output label IDs in evaluation GT.

    Training uses the configured channel mapping. Full-volume evaluation can
    supply a separate mapping so merged training classes do not erase GT sides.
    """
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

    mapping = config.labels.mapping
    if label_mapping is not None:
        if stage != "inference":
            raise ValueError("Label mapping overrides are only supported for full-volume inference")
        mapping = label_mapping

    steps.append(MapLabels(mapping=mapping, unknown=config.labels.unknown))
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
            crop = RandomCropping(
                dim=settings.patch_size, mode=settings.crop_mode, std=None, padding=True,
            )
        else:
            crop = PadderCutter(img_size=settings.patch_size)
        steps.append(crop)

    if stage == "training" and settings.augment and settings.flip_probability > 0:
        flip = Flip(anatomical_axis="LR", probability=settings.flip_probability)
        steps.append(flip)

    return Preprocessor(steps=steps)
