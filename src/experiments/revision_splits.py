"""Shared animal folds for TBI pretraining, MouseFull and reviewer controls."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from .data import PreparedData, SubjectSplit
from ..evaluation.k_fold import FoldSplit, SubjectIndex
from ..utils.loader import RodentDatasets


def select_animals(data: PreparedData, animals: Sequence[str]) -> PreparedData:
    """Copy the loader index with all scans of exactly the requested animals."""
    keys = data.subjects.scan_keys(animals)
    loader = RodentDatasets(labels=list(data.config.labels.mapping))
    overrides: dict[tuple[str, str], str] = {}
    for key in keys:
        scan = data.loader.get_subject(key)
        loader.datasets_dic.setdefault(key[0], {})[key[1]] = scan
        overrides[key] = data.subjects.subject_for_scan(key)
    subjects = SubjectIndex(loader, subject_overrides=overrides)
    return replace(data, loader=loader, subjects=subjects, evaluation_patches=None)


def validate_folds(folds: Sequence[FoldSplit], animals: Sequence[str]) -> None:
    """Require disjoint roles and one outer test appearance per animal."""
    if len(folds) < 2 or [fold.index for fold in folds] != list(range(len(folds))):
        raise ValueError("Provide at least two folds with consecutive indexes")
    holdouts: list[str] = []
    for fold in folds:
        fold.validate(set(animals))
        if not fold.test_ids:
            raise ValueError("Every fold requires an independent outer test set")
        holdouts.extend(fold.test_ids)
    if sorted(holdouts) != sorted(animals):
        raise ValueError("Every animal must occur in exactly one outer holdout")


def project_folds(folds: Sequence[FoldSplit], animals: Sequence[str]) -> list[FoldSplit]:
    """Preserve each selected animal's role in every source/target fold."""
    selected = set(animals)
    result: list[FoldSplit] = []
    for fold in folds:
        train = sorted(selected.intersection(fold.train_ids))
        validation = sorted(selected.intersection(fold.val_ids))
        test = sorted(selected.intersection(fold.test_ids))
        result.append(FoldSplit(fold.index, train, validation, test))
    validate_folds(result, animals)
    return result


def merge_folds(first: Sequence[FoldSplit], second: Sequence[FoldSplit]) -> list[FoldSplit]:
    """Combine independent cohorts without changing inherited assignments."""
    if len(first) != len(second):
        raise ValueError("Cohort fold counts must match")
    result: list[FoldSplit] = []
    for left, right in zip(first, second):
        train = sorted(left.train_ids + right.train_ids)
        validation = sorted(left.val_ids + right.val_ids)
        test = sorted(left.test_ids + right.test_ids)
        result.append(FoldSplit(left.index, train, validation, test))
    animals = first[0].train_ids + first[0].val_ids + first[0].test_ids
    animals += second[0].train_ids + second[0].val_ids + second[0].test_ids
    validate_folds(result, animals)
    return result


def _balanced_tests(animals: Sequence[str], groups: dict[str, str], k: int, seed: int) -> list[list[str]]:
    """Spread each annotation stratum round-robin, including strata smaller than k."""
    rng = np.random.default_rng(seed)
    tests: list[list[str]] = [[] for _ in range(k)]
    for group in sorted({groups[animal] for animal in animals}):
        members = sorted(animal for animal in animals if groups[animal] == group)
        shuffled = rng.permutation(members).tolist()
        start = min(range(k), key=lambda index: len(tests[index]))
        for offset, animal in enumerate(shuffled):
            tests[(start + offset) % k].append(animal)
    return tests


def balanced_folds(
    data: PreparedData, *, k: int, seed: int, validation_fraction: float,
    anchor: SubjectSplit | None = None,
) -> list[FoldSplit]:
    """Balance anatomy coverage; an optional fixed split remains exact in fold zero."""
    animals = data.subjects.subject_ids
    if not 2 <= k <= len(animals):
        raise ValueError("Too few selected animals for the requested folds")
    groups = data.annotation_groups(include_lesion=False)
    tests = _balanced_tests(animals, groups, k, seed)
    if anchor is not None:
        anchor.validate(data.subjects)
        pool = sorted(set(animals) - set(anchor.test))
        remaining = _balanced_tests(pool, groups, k - 1, seed)
        tests = [list(anchor.test), *remaining]

    result: list[FoldSplit] = []
    for index, test in enumerate(tests):
        if anchor is not None and index == 0:
            train = list(anchor.train)
            validation = list(anchor.validation)
        else:
            pool = sorted(set(animals) - set(test))
            count = max(1, int(np.ceil(len(pool) * validation_fraction)))
            if count >= len(pool):
                raise ValueError("Too few animals for separate fitting and inner validation")
            rng = np.random.default_rng(seed + index)
            shuffled = rng.permutation(pool).tolist()
            validation = sorted(shuffled[:count])
            train = sorted(shuffled[count:])
        result.append(FoldSplit(index, train, validation, sorted(test)))
    validate_folds(result, animals)
    return result


def read_selection(path: Path) -> list[FoldSplit] | SubjectSplit:
    """Read existing repository splits; legacy two-role CV is deliberately rejected."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))

    def role(item: dict[str, object], name: str) -> list[str]:
        """Reject scan-index arrays or malformed roles before constructing folds."""
        values = item.get(name)
        if not isinstance(values, list) or any(not isinstance(value, str) or not value for value in values):
            raise ValueError(f"Split role {name} must be a list of canonical animal IDs")
        return values

    if isinstance(payload, dict):
        train_ids = tuple(role(payload, "train"))
        validation_ids = tuple(role(payload, "validation"))
        test_ids = tuple(role(payload, "test"))
        return SubjectSplit(train_ids, validation_ids, test_ids)
    if not isinstance(payload, list) or not payload:
        raise ValueError("Expected a fixed split object or a list of CV folds")
    folds: list[FoldSplit] = []
    for index, item in enumerate(payload):
        if not isinstance(item, dict) or not item.get("test"):
            raise ValueError("Provided CV folds require separate train, validation and test roles")
        train = role(item, "train")
        validation = role(item, "validation")
        test = role(item, "test")
        folds.append(FoldSplit(index, train, validation, test))
    return folds


@dataclass(frozen=True)
class SuiteSplits:
    """Exact stage memberships plus the interpolation pilot's fixed split."""

    d1: list[FoldSplit]
    mouse_full: list[FoldSplit]
    rats: list[FoldSplit]
    interpolation: SubjectSplit

    def to_dict(self) -> dict[str, object]:
        """Serialize one auditable plan, independent of model training seeds."""
        return {
            "d1": [fold.to_dict() for fold in self.d1],
            "mouse_full": [fold.to_dict() for fold in self.mouse_full],
            "rats": [fold.to_dict() for fold in self.rats],
            "interpolation": {
                "train": self.interpolation.train, "validation": self.interpolation.validation,
                "test": self.interpolation.test,
            },
        }


def make_suite_splits(
    d1: PreparedData, mouse_full: PreparedData, rats: PreparedData | None, *,
    k: int, seed: int, validation_fraction: float, folds_file: Path | None = None,
    anchor_split_file: Path | None = None, rat_folds_file: Path | None = None,
) -> SuiteSplits:
    """Generate or inherit a master plan and prove source/target compatibility."""
    d1_ids = d1.subjects.subject_ids
    mouse_ids = mouse_full.subjects.subject_ids
    sham_ids = sorted(set(mouse_ids) - set(d1_ids))
    sham = select_animals(mouse_full, sham_ids)
    d1_for_balance = select_animals(mouse_full, d1_ids)
    supplied: list[FoldSplit] | SubjectSplit | None = None
    selected_path = folds_file or anchor_split_file
    if selected_path is not None:
        supplied = read_selection(selected_path)
        if folds_file is not None and not isinstance(supplied, list):
            raise ValueError("folds_file needs CV folds; use anchor_split_file for a fixed split")
        if anchor_split_file is not None and not isinstance(supplied, SubjectSplit):
            raise ValueError("anchor_split_file needs one fixed split")

    if isinstance(supplied, list):
        ids = supplied[0].train_ids + supplied[0].val_ids + supplied[0].test_ids
        k = len(supplied)
        if set(ids) == set(mouse_ids):
            validate_folds(supplied, mouse_ids)
            mouse_folds = supplied
            d1_folds = project_folds(mouse_folds, d1_ids)
        elif set(ids) == set(d1_ids):
            validate_folds(supplied, d1_ids)
            d1_folds = supplied
            extra = balanced_folds(sham, k=k, seed=seed, validation_fraction=validation_fraction)
            mouse_folds = merge_folds(d1_folds, extra)
        else:
            raise ValueError("Provided folds must cover exactly selected D1 or selected D1+D3 animals")
    else:
        anchor_d1: SubjectSplit | None = None
        anchor_sham: SubjectSplit | None = None
        if supplied is not None:
            ids = set(supplied.train + supplied.validation + supplied.test)
            if ids == set(d1_ids):
                anchor_d1 = supplied
            elif ids == set(mouse_ids):
                supplied.validate(mouse_full.subjects)
                selected_d1 = set(d1_ids)
                selected_sham = set(sham_ids)
                anchor_d1 = SubjectSplit(*(
                    tuple(animal for animal in role if animal in selected_d1)
                    for role in (supplied.train, supplied.validation, supplied.test)
                ))
                anchor_sham = SubjectSplit(*(
                    tuple(animal for animal in role if animal in selected_sham)
                    for role in (supplied.train, supplied.validation, supplied.test)
                ))
            else:
                raise ValueError("Anchor must cover selected D1 or selected D1+D3 animals")
        d1_folds = balanced_folds(
            d1_for_balance, k=k, seed=seed, validation_fraction=validation_fraction, anchor=anchor_d1,
        )
        sham_folds = balanced_folds(
            sham, k=k, seed=seed, validation_fraction=validation_fraction, anchor=anchor_sham,
        )
        mouse_folds = merge_folds(d1_folds, sham_folds)

    rat_folds: list[FoldSplit] = []
    if rats is not None:
        rat_ids = rats.subjects.subject_ids
        if rat_folds_file is not None:
            loaded = read_selection(rat_folds_file)
            if not isinstance(loaded, list):
                raise ValueError("Rat folds require an outer CV plan")
            rat_folds = loaded
            validate_folds(rat_folds, rat_ids)
        else:
            rat_folds = balanced_folds(
                rats, k=len(rat_ids), seed=seed, validation_fraction=validation_fraction,
            )
        if any(len(fold.test_ids) != 1 for fold in rat_folds):
            raise ValueError("Rat evaluation must leave exactly one complete animal out per fold")

    projected = project_folds(mouse_folds, d1_ids)
    for target, source in zip(projected, d1_folds):
        for role, animals in target.to_dict().items():
            if set(animals) != set(source.to_dict()[role]):
                raise ValueError("Pretraining and MouseFull animal roles differ")
    first = d1_folds[0]
    interpolation = SubjectSplit(tuple(first.train_ids), tuple(first.val_ids), tuple(first.test_ids))
    return SuiteSplits(d1_folds, mouse_folds, rat_folds, interpolation)


def split_membership(data: PreparedData, folds: Sequence[FoldSplit], task: str) -> pd.DataFrame:
    """Expand saved animal memberships to every scan, for collaborator handoff."""
    frames: list[pd.DataFrame] = []
    for fold in folds:
        for role, ids in fold.to_dict().items():
            table = data.subjects.manifest(ids)
            frames.append(table.assign(Task=task, Fold=fold.index, DatasetRole=role))
    return pd.concat(frames, ignore_index=True)
