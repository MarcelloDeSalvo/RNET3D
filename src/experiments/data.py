"""Animal-level dataset preparation and stage-specific preprocessing factories."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Mapping, Sequence

import nibabel as nib
import numpy as np
import pandas as pd

from .config import ExperimentConfig, FixedSplitConfig, InitializationConfig
from ..evaluation.k_fold import FoldSplit, SubjectIndex
from ..utils.loader import Rodent, RodentDatasets

if TYPE_CHECKING:
    from .patches import EvaluationPatches
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


@dataclass(frozen=True)
class TrainingSelection:
    """A fixed split or complete CV plan, with first-fold preview identities."""

    split: SubjectSplit | None
    folds: list[FoldSplit] | None
    summary: pd.DataFrame

    def __post_init__(self) -> None:
        """Require exactly one nonempty selection mode."""
        if (self.split is None) == (self.folds is None) or self.folds == []:
            raise ValueError("Provide either a fixed split or nonempty CV folds")

    @property
    def train_ids(self) -> tuple[str, ...]:
        """Return fixed training IDs or the first fold for a generator preview."""
        if self.split is not None:
            return self.split.train
        assert self.folds is not None
        return tuple(self.folds[0].train_ids)

    @property
    def validation_ids(self) -> tuple[str, ...]:
        """Return fixed validation IDs or the first fold for a generator preview."""
        if self.split is not None:
            return self.split.validation
        assert self.folds is not None
        return tuple(self.folds[0].val_ids)

    @property
    def evaluation_role(self) -> Literal["test", "holdout"]:
        """Choose independent final-test or outer-holdout result displays."""
        if self.split is not None:
            return "test"
        return "holdout"


def resolve_training_selection(
    data: PreparedData, *, k_fold: bool = False,
    resume_run: Path | None = None, split_file: Path | None = None, fold_file: Path | None = None,
    reference_file: Path | None = None, reference_fold: int = 0,
    strata: Mapping[str, str] | None = None,
    initializations: Sequence[InitializationConfig] | None = None,
    initializations_by_fold: Mapping[int, Sequence[InitializationConfig]] | None = None,
) -> TrainingSelection:
    """Select fixed training or all compatible folds from one explicit switch.

    Saved memberships are authoritative. CV has no additional permanent test
    split: every selected animal contributes one independent outer holdout.
    reference_file supplies a common plan; fixed mode selects reference_fold.
    Explicit split files and resumed memberships override this generation default.
    Source memberships and initialization ancestors retain their existing guards.
    """
    if not isinstance(k_fold, bool):
        raise ValueError("k_fold must be boolean")
    if not k_fold:
        if fold_file is not None:
            raise ValueError("FOLD_FILE requires K_FOLD_COMPATIBLE=True")
        split = resolve_training_split(
            data, resume_run=resume_run, split_file=split_file, initializations=initializations,
            reference_file=reference_file, reference_fold=reference_fold,
        )
        summary = data.split_summary(split)
        return TrainingSelection(split=split, folds=None, summary=summary)

    if split_file is not None:
        raise ValueError("Use FOLD_FILE for CV; SPLIT_FILE belongs to fixed-split mode")
    from ..evaluation.k_fold import load_fold_splits

    animals = data.subjects.subject_ids
    k = data.config.training.k
    if resume_run is not None:
        saved = Path(resume_run) / "splits.json"
        folds = load_fold_splits(saved, animals, k)
        if fold_file is not None:
            requested = load_fold_splits(fold_file, animals, k)
            if requested != folds:
                raise ValueError("FOLD_FILE differs from the resumed run's folds")
    elif fold_file is not None:
        folds = load_fold_splits(fold_file, animals, k)
    else:
        folds = make_pretrained_cv_splits(
            data, strata=strata, initializations_by_fold=initializations_by_fold,
            reference_file=reference_file,
        )
    if any(not fold.test_ids for fold in folds):
        raise ValueError("K-Fold-compatible mode requires independent outer holdouts")
    summary = fold_summary(folds, data.subjects)
    return TrainingSelection(split=None, folds=folds, summary=summary)


def make_split(
    index: SubjectIndex, *, seed: int, validation_fraction: float = 0.2,
    test_fraction: float = 0.1, strata: Mapping[str, str] | None = None,
) -> SubjectSplit:
    """Split whole animals proportionally by cohort/modality or supplied strata.

    For longitudinal/small cohorts or matched manuscript experiments, load a
    scientifically selected split instead. The local RNG does not alter training.
    """
    settings = FixedSplitConfig(
        validation_fraction=validation_fraction, test_fraction=test_fraction,
    )
    settings.validate()
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("Split seed must be a nonnegative integer")

    from .splits import stratified_validation_ids

    ids = index.subject_ids
    count: int = len(ids)
    n_validation: int = max(1, int(np.ceil(count * validation_fraction)))
    n_test: int = max(1, int(np.ceil(count * test_fraction)))
    if n_validation + n_test >= count:
        raise ValueError("Too few animals for the requested three-way split")

    if strata is None:
        manifest = index.manifest(ids)
        strata = {}
        for animal, scans in manifest.groupby("Subject"):
            datasets = ",".join(sorted(scans["Dataset"].unique()))
            modalities = ",".join(sorted(scans["Modality"].unique()))
            strata[animal] = f"{datasets}|{modalities}"
    test = tuple(stratified_validation_ids(ids, strata, n_test, seed))
    remaining = sorted(set(ids) - set(test))
    validation = tuple(stratified_validation_ids(remaining, strata, n_validation, seed + 1))
    train = tuple(sorted(set(remaining) - set(validation)))
    split = SubjectSplit(train=train, validation=validation, test=test)
    split.validate(index)
    return split


def resolve_training_split(
    data: PreparedData, *, resume_run: Path | None = None, split_file: Path | None = None,
    initializations: Sequence[InitializationConfig] | None = None,
    reference_file: Path | None = None, reference_fold: int = 0,
) -> SubjectSplit:
    """Load exact saved roles or inherit source roles before splitting new animals.

    Pretrained source fitting/validation/test animals retain their roles in new
    splits. New animals fill the requested overall validation/test proportions.
    Explicit/resumed memberships are never changed and must remain independent
    of source fitting; selection-validation may be shared across both stages.
    A reference_file selects one fold's three roles before applying these checks.
    """
    from .initialization import pretraining_roles

    config = data.config
    if initializations is None:
        initializations = (config.initialization,)
    inherited = pretraining_roles(data, initializations)
    split: SubjectSplit | None = None
    if resume_run is not None:
        saved = Path(resume_run) / "splits.json"
        split = SubjectSplit.load(saved, data.subjects)
        if split_file is not None:
            requested = SubjectSplit.load(split_file, data.subjects)
            if requested != split:
                raise ValueError("SPLIT_FILE differs from the resumed run's split")
    elif split_file is not None:
        split = SubjectSplit.load(split_file, data.subjects)
    elif reference_file is not None:
        from .splits import reference_folds

        folds = reference_folds(data, reference_file)
        if type(reference_fold) is not int or not 0 <= reference_fold < len(folds):
            raise ValueError("REFERENCE_FOLD must identify a fold in REFERENCE_SPLITS")
        selected = folds[reference_fold]
        train_ids = tuple(selected.train_ids)
        validation_ids = tuple(selected.val_ids)
        test_ids = tuple(selected.test_ids)
        split = SubjectSplit(train=train_ids, validation=validation_ids, test=test_ids)
        split.validate(data.subjects)

    if split is not None:
        for animal in split.validation:
            if inherited.get(animal) == "train":
                raise ValueError(f"Pretraining leakage: validation animal was fitted by the source: {animal}")
        for animal in split.test:
            if inherited.get(animal) in ("train", "validation"):
                raise ValueError(f"Pretraining leakage: test animal was used by the source: {animal}")
        return split

    settings = config.fixed_split
    seed = config.split_seed
    if not inherited:
        from .splits import modality_annotation_strata

        strata = modality_annotation_strata(data)
        return make_split(
            data.subjects, seed=seed,
            validation_fraction=settings.validation_fraction, test_fraction=settings.test_fraction,
            strata=strata,
        )

    groups: dict[str, list[str]] = {"train": [], "validation": [], "test": []}
    for animal, role in inherited.items():
        groups[role].append(animal)
    free = sorted(set(data.subjects.subject_ids) - set(inherited))
    count = len(data.subjects.subject_ids)
    desired_validation = int(np.ceil(count * settings.validation_fraction))
    desired_test = int(np.ceil(count * settings.test_fraction))
    minimum_validation = int(not groups["validation"])
    minimum_test = int(not groups["test"])
    budget = len(free) - int(not groups["train"])
    if budget < minimum_validation + minimum_test:
        raise ValueError("Pretraining leaves too few independent animals for validation/test; use another source or scratch")

    validation_count = max(minimum_validation, desired_validation - len(groups["validation"]))
    test_count = max(minimum_test, desired_test - len(groups["test"]))
    if validation_count + test_count > budget:
        extra_budget = budget - minimum_validation - minimum_test
        extras = np.asarray([validation_count - minimum_validation, test_count - minimum_test])
        proportions = extras / extras.sum()
        quotas = proportions * extra_budget
        allocated = np.floor(quotas).astype(int)
        remainder = extra_budget - int(allocated.sum())
        order = np.argsort(-(quotas - allocated), kind="stable")
        allocated[order[:remainder]] += 1
        validation_count = minimum_validation + int(allocated[0])
        test_count = minimum_test + int(allocated[1])

    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(free).tolist()
    groups["test"].extend(shuffled[:test_count])
    validation_end = test_count + validation_count
    groups["validation"].extend(shuffled[test_count:validation_end])
    groups["train"].extend(shuffled[validation_end:])
    train_ids = tuple(sorted(groups["train"]))
    validation_ids = tuple(sorted(groups["validation"]))
    test_ids = tuple(sorted(groups["test"]))
    split = SubjectSplit(train=train_ids, validation=validation_ids, test=test_ids)
    split.validate(data.subjects)
    return split


def annotation_values(scan: Rodent) -> set[int]:
    """Resolve schema-v3 annotation names; never infer availability from voxels.

    label_mapping describes a vocabulary and is not an annotation declaration.
    SHAM means verified lesion-free and supplies an available empty lesion mask,
    even when has_lesion_annotation is false. For other scans, a false annotation
    flag means unknown lesion supervision rather than verified absence.
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
    if lesion_annotation is True or scan.get_sham():
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
    50/50 group ratio or oversample animals. Inner validation uses the same strata.
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
        validation = sorted(shuffled[:validation_count].tolist())
        if strata is not None:
            from .splits import stratified_validation_ids

            validation = stratified_validation_ids(pool.tolist(), strata, validation_count, seed + index)
        train = sorted(set(pool.tolist()) - set(validation))
        fold = FoldSplit(
            index=index, train_ids=train,
            val_ids=validation, test_ids=sorted(ids[test_indexes].tolist()),
        )
        fold.validate(set(subjects))
        folds.append(fold)
    return folds


def make_pretrained_cv_splits(
    data: PreparedData, *, subject_ids: Sequence[str] | None = None,
    strata: Mapping[str, str] | None = None,
    initializations_by_fold: Mapping[int, Sequence[InitializationConfig]] | None = None,
    reference_file: Path | None = None,
) -> list[FoldSplit]:
    """Reuse source folds and distribute additional animals across outer holdouts.

    Each pretrained fold must cover the same matching source animals and keep
    its fitting/selection/holdout roles. Together their holdouts must cover each
    animal exactly once. A shared checkpoint cannot satisfy that requirement for
    overlapping cohorts. Sources on independent cohorts retain ordinary CV.
    """
    from .initialization import pretraining_roles

    config = data.config
    if subject_ids is None:
        subject_ids = data.subjects.subject_ids
    animals = sorted(subject_ids)
    data.subjects.scan_keys(animals)
    if not 2 <= config.training.k <= len(animals):
        raise ValueError("k must be between 2 and the number of CV animals")
    if strata is not None and set(strata) != set(animals):
        raise ValueError("CV strata must cover exactly the selected animals")

    inherited: list[dict[str, str]] = []
    for index in range(config.training.k):
        initialization = config.initialization_by_fold.get(index, config.initialization)
        selected: Sequence[InitializationConfig] = (initialization,)
        if initializations_by_fold is not None:
            selected = initializations_by_fold.get(index, selected)
        roles = pretraining_roles(data, selected, subject_ids=animals)
        inherited.append(roles)
    if reference_file is not None:
        from .splits import reference_folds

        folds = reference_folds(data, reference_file, subject_ids=animals, strata=strata)
        if len(folds) != config.training.k:
            raise ValueError("Configured K_FOLD_COUNT must match the reference plan's fold count")
        for fold, roles in zip(folds, inherited):
            for animal in fold.val_ids:
                if roles.get(animal) == "train":
                    raise ValueError(f"Pretraining leakage: reference validation animal was fitted by the source: {animal}")
            for animal in fold.test_ids:
                if roles.get(animal) in ("train", "validation"):
                    raise ValueError(f"Pretraining leakage: reference test animal was used by the source: {animal}")
        return folds
    matched = set(inherited[0])
    if any(set(roles) != matched for roles in inherited):
        raise ValueError("Pretraining source folds must cover the same animals; provide matching sources for every fold")
    if not matched:
        return make_cv_splits(
            animals, k=config.training.k, seed=config.split_seed,
            validation_fraction=config.cv_validation_fraction, strata=strata,
        )

    held_out = [animal for roles in inherited for animal, role in roles.items() if role == "test"]
    if sorted(held_out) != sorted(matched):
        raise ValueError("Pretrained CV requires one matching source checkpoint per fold, with disjoint source holdouts covering each animal once")

    additional = sorted(set(animals) - matched)
    extra_folds: list[FoldSplit] | None = None
    if additional:
        additional_strata: dict[str, str] | None = None
        if strata is not None:
            additional_strata = {animal: strata[animal] for animal in additional}
        largest_holdout = int(np.ceil(len(additional) / config.training.k))
        if len(additional) >= config.training.k and len(additional) - largest_holdout >= 2:
            extra_folds = make_cv_splits(
                additional, k=config.training.k, seed=config.split_seed,
                validation_fraction=config.cv_validation_fraction, strata=additional_strata,
            )
        else:
            # Source roles already supply training and validation. Tiny additions
            # can have empty roles in a fold while still being held out once.
            rng = np.random.default_rng(config.split_seed)
            shuffled = rng.permutation(additional)
            holdouts = np.array_split(shuffled, config.training.k)
            extra_folds = []
            for index, holdout in enumerate(holdouts):
                pool = [animal for animal in shuffled.tolist() if animal not in holdout]
                desired = int(np.ceil(len(pool) * config.cv_validation_fraction))
                validation_count = min(desired, max(0, len(pool) - 1))
                validation = pool[:validation_count]
                train = pool[validation_count:]
                test = holdout.tolist()
                extra = FoldSplit(index=index, train_ids=train, val_ids=validation, test_ids=test)
                extra_folds.append(extra)

    folds: list[FoldSplit] = []
    for index, roles in enumerate(inherited):
        train = [animal for animal, role in roles.items() if role == "train"]
        validation = [animal for animal, role in roles.items() if role == "validation"]
        test = [animal for animal, role in roles.items() if role == "test"]
        if extra_folds is not None:
            extra = extra_folds[index]
            train.extend(extra.train_ids)
            validation.extend(extra.val_ids)
            test.extend(extra.test_ids)
        train.sort()
        validation.sort()
        test.sort()
        fold = FoldSplit(index=index, train_ids=train, val_ids=validation, test_ids=test)
        fold.validate(set(animals))
        folds.append(fold)
    return folds


def make_leave_one_animal_out_splits(subjects: list[str]) -> list[FoldSplit]:
    """Hold out one complete animal and fit all others under a fixed protocol.

    No inner validation is reserved. Use TrainingConfig.fixed_epochs=True so
    held-out scans are evaluated only after fitting. SubjectIndex expands each
    animal to every registered longitudinal scan.
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
        fold = FoldSplit(index=index, train_ids=training_ids, val_ids=[], test_ids=[animal])
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
    evaluation_patches: EvaluationPatches | None = None

    def patch_collection(self) -> EvaluationPatches:
        """Reuse the same evaluation samples across roles, folds and study members."""
        from .patches import EvaluationPatches

        if self.evaluation_patches is None:
            self.evaluation_patches = EvaluationPatches.create(self)
        elif self.evaluation_patches.contract != EvaluationPatches.expected_contract(self):
            raise ValueError("Cached evaluation patches differ from the requested grid or sampling")

        return self.evaluation_patches

    def save_evaluation_patches(self, path: Path, *, resume: bool = False) -> None:
        """Load saved samples on resume; never replace an established collection."""
        from .patches import EvaluationPatches

        if path.exists():
            saved = EvaluationPatches.load(path, self)
            if self.evaluation_patches is not None and saved.samples != self.evaluation_patches.samples:
                raise ValueError("Evaluation patches differ from the shared collection")
            self.evaluation_patches = saved
            return

        settings = self.config.evaluation_patches
        legacy_center = settings.sampling == "center" and not settings.left_right_pairs
        if resume and not legacy_center:
            raise FileNotFoundError(f"Cannot resume without saved evaluation patches: {path}")

        collection = self.patch_collection()
        collection.save(path)

    def make_split(self) -> SubjectSplit:
        """Generate fixed roles, inheriting pretraining memberships when present."""
        return resolve_training_split(self)

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
        """Expand training repetitions or saved validation/test patches per scan."""
        from ..preprocessing.generator import MultiTaskGenerator
        from .patches import FixedPatchPreprocessor

        config = self.config
        keys = self.subjects.scan_keys(ids)
        stage: Literal["training", "validation", "inference"] = "validation"
        if training:
            stage = "training"
            keys = keys * config.training.data_multiplier

        pipeline = make_preprocessor(config, stage=stage)
        sample_preprocessors = None
        if not training:
            collection = self.patch_collection()
            selected = set(keys)
            samples = [sample for sample in collection.samples if sample.scan in selected]
            keys = [sample.scan for sample in samples]
            pipeline = make_preprocessor(config, stage="inference")
            size = config.preprocessing.patch_size
            sample_preprocessors = [FixedPatchPreprocessor(pipeline, sample, size) for sample in samples]
        settings = {
            "dataset_path": str(config.data.root),
            "labels": list(config.labels.mapping),
            "label_mapping": config.labels.mapping,
            "input_shape": config.preprocessing.patch_size,
            "in_channels": 1,
            "num_classes": len(config.labels.names),
            "batch_size": config.training.batch_size,
        }
        missing_as_background: bool = config.loss.annotation_policy == "missing_as_background"
        partial_labels: bool = config.loss.regions == "dice_ce_gradient_fix" and not missing_as_background
        return MultiTaskGenerator(
            ids=keys, loader=self.loader, config=settings, shuffle=training,
            preprocessor=pipeline, drop_last=False, partial_labels=partial_labels,
            with_classifier=config.model.classifier, batch_size=batch_size,
            sample_preprocessors=sample_preprocessors,
            missing_labels_as_background=missing_as_background,
        )


def prepare_data(config: ExperimentConfig, *, require_training_loss: bool = True) -> PreparedData:
    """Register datasets and resolve missing source labels per scan.

    Normal training requires a partial-label-aware loss for incomplete annotations.
    An explicit missing-as-background control changes targets, not GT metadata.
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

            if scan.get_sham():
                lesion_data: np.ndarray = scan.labels.get_fdata(caching="unchanged")
                lesion_voxels = np.any(lesion_data == 1)
                del lesion_data
                if lesion_voxels:
                    raise ValueError(f"{scan.id}: verified lesion absence conflicts with lesion-positive voxels")
                available.add(1)

            scan.unavailable_labels = sorted(set(labels) - available)
            masked_policy: bool = config.loss.annotation_policy == "masked"
            if require_training_loss and masked_policy and scan.unavailable_labels and config.loss.regions != "dice_ce_gradient_fix":
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
    prepared = PreparedData(config=config, loader=loader, subjects=subjects)
    validate_inventory(prepared)
    return prepared


def validate_inventory(data: PreparedData) -> pd.DataFrame:
    """Cross-check registered normalized scans against mapping.xlsx when present.

    Compare exact cohort, normalized ID, timepoint and original ID; similar
    original names never establish biological identity across cohorts.
    Legacy layouts or datasets without a workbook retain their explicit contracts.
    """
    manifest = data.subjects.manifest(data.subjects.subject_ids)
    inventory_path = data.config.data.root / "mapping.xlsx"
    if not inventory_path.is_file():
        return manifest

    inventory = pd.read_excel(inventory_path, sheet_name="Mapping", dtype=str).fillna("")
    columns = ["dataset", "anonymized_subject_id", "timepoint_or_context", "original_subject_id"]
    if not set(columns).issubset(inventory.columns):
        raise ValueError("mapping.xlsx is missing scan identity columns")

    normalized = {dataset.name for dataset in data.config.data.datasets if dataset.layout == "normalized"}
    selected = inventory.loc[inventory["dataset"].isin(normalized), columns]
    keys = columns[:3]
    if selected.duplicated(keys).any():
        raise ValueError("mapping.xlsx contains duplicated normalized scan identities")
    declared = selected.set_index(keys)["original_subject_id"].to_dict()
    for row in manifest.itertuples():
        if row.Dataset not in normalized:
            continue
        timepoint = row.Timepoint or ""
        key = (row.Dataset, row.SubjectID, timepoint)
        if declared.get(key) != row.OriginalID:
            raise ValueError(f"{row.Subject}/{row.Scan}: metadata identity differs from mapping.xlsx")

    return manifest


def make_preprocessor(
    config: ExperimentConfig, *, stage: Literal["training", "validation", "inference"],
    label_mapping: dict[int, int] | None = None,
) -> Preprocessor:
    """Build shared geometry and acquisition-style MRI-only intensity augmentation.

    Training uses the configured channel mapping. Full-volume evaluation can
    supply a separate mapping so merged training classes do not erase GT sides.
    New training runs blur/add noise on the native grid before resampling,
    followed by physical affine augmentation. Saved resampled-grid protocols
    retain affine/blur/noise after resampling. MRI grid resampling uses the saved
    spline order (linear by default), affine MRI augmentation stays linear and
    masks use nearest neighbour. Validation and inference have no augmentation.
    """
    from ..preprocessing.preprocessor import (
        CorrectX10, Flip, GaussianBlur, MapLabels, Noise, Normalize, PadderCutter,
        Preprocessor, PreprocessingStep, RandomAffine, RandomCropping, Reorient, Resample,
    )

    if stage not in ("training", "validation", "inference"):
        raise ValueError("Unknown preprocessing stage")

    settings = config.preprocessing
    settings.validate()
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
    augment_training: bool = stage == "training" and settings.augment
    intensity_steps: list[PreprocessingStep] = []
    if augment_training:
        probability = settings.augmentation_probability
        blur_range = settings.blur_sigma_range
        blur_units = settings.blur_sigma_units
        noise_range = settings.noise_std_range
        noise_units = settings.noise_std_units

        blur = GaussianBlur(
            gaussian_blur_range=blur_range, probability=probability, sigma_units=blur_units,
        )
        noise = Noise(noise_range=noise_range, probability=probability, std_units=noise_units)
        intensity_steps = [blur, noise]
        if settings.intensity_augmentation_grid == "native":
            steps.extend(intensity_steps)

    resampling_order: int = settings.resampling_order
    resampling = Resample(
        target_resolution=settings.spacing_mm, interpolation=resampling_order, orthogonal=False,
    )
    steps.append(resampling)

    if augment_training:
        affine = RandomAffine(
            scale_range=settings.scale_range, rotation_range=settings.rotation_range,
            probability=settings.augmentation_probability,
        )
        steps.append(affine)
        if settings.intensity_augmentation_grid == "resampled":
            steps.extend(intensity_steps)

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
