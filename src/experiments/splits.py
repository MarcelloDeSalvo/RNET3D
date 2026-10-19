"""Shared animal split validation, reference folds and cohort extension."""

from __future__ import annotations

import json
import re
from dataclasses import replace
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

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
    anchor: SubjectSplit | None = None, strata: Mapping[str, str] | None = None,
) -> list[FoldSplit]:
    """Balance anatomy coverage; an optional fixed split remains exact in fold zero."""
    animals = data.subjects.subject_ids
    if not 2 <= k <= len(animals):
        raise ValueError("Too few selected animals for the requested folds")
    groups = data.annotation_groups(include_lesion=False)
    if strata is not None:
        groups = {animal: strata[animal] for animal in animals}
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


def read_selection(
    path: Path, *, seed: int = 42, validation_fraction: float = 0.2,
) -> list[FoldSplit] | SubjectSplit:
    """Read saved roles, converting historical two-role CV into independent tests.

    D1_SUBJ_0001 is normalized to D1/SUBJ_0001. In a train/val CV plan, val
    remains the outer holdout and inner validation is drawn only from train.
    Explicit train/validation/test plans retain their memberships and order.
    The input file is never modified.
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not np.isfinite(validation_fraction) or not 0 < validation_fraction < 1:
        raise ValueError("Inner validation fraction must be between zero and one")

    def role(item: dict[str, object], name: str) -> list[str]:
        """Reject scan-index arrays or malformed roles before constructing folds."""
        values = item.get(name)
        if not isinstance(values, list) or any(not isinstance(value, str) or not value for value in values):
            raise ValueError(f"Split role {name} must be a list of canonical animal IDs")
        normalized: list[str] = []
        for value in values:
            match = re.fullmatch(r"(D\d+)_(SUBJ_\d+)", value)
            if match is not None:
                value = "/".join(match.groups())
            normalized.append(value)
        return normalized

    if isinstance(payload, dict):
        train_ids = tuple(role(payload, "train"))
        validation_ids = tuple(role(payload, "validation"))
        test_ids = tuple(role(payload, "test"))
        return SubjectSplit(train_ids, validation_ids, test_ids)
    if not isinstance(payload, list) or not payload:
        raise ValueError("Expected a fixed split object or a list of CV folds")
    folds: list[FoldSplit] = []
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError("CV entries must be split objects")
        train = role(item, "train")
        validation_name: str = "validation"
        if "val" in item:
            validation_name = "val"
            if "validation" in item:
                raise ValueError("Use val or validation, not both")
        validation = role(item, validation_name)
        test: list[str] = []
        if "test" in item:
            test = role(item, "test")
            if not test:
                raise ValueError("Explicit CV test roles must not be empty")
        else:
            # Validate the original partition before splitting its training pool.
            original = FoldSplit(index, train, validation)
            original.validate(set(train + validation))
            test = validation
            pool = sorted(train)
            count = max(1, int(np.ceil(len(pool) * validation_fraction)))
            if count >= len(pool):
                raise ValueError("Historical CV needs at least two training animals for inner validation")
            rng = np.random.default_rng(seed + index)
            shuffled = rng.permutation(pool).tolist()
            train = sorted(shuffled[count:])
            validation = sorted(shuffled[:count])
        folds.append(FoldSplit(index, train, validation, test))
    first = folds[0]
    animals = first.train_ids + first.val_ids + first.test_ids
    validate_folds(folds, animals)
    return folds


def reference_folds(
    data: PreparedData, path: Path, *, subject_ids: Sequence[str] | None = None,
    strata: Mapping[str, str] | None = None,
) -> list[FoldSplit]:
    """Preserve reference animal roles and extend additional cohorts once.

    The same seed and inner fraction produce identical D1 roles for fixed
    training, CV and MouseFull ablations. Additional SHAM animals
    are balanced by anatomy coverage unless explicit strata are supplied.
    """
    config = data.config
    loaded = read_selection(
        path, seed=config.split_seed, validation_fraction=config.cv_validation_fraction,
    )
    if not isinstance(loaded, list):
        raise ValueError("Reference splits must contain a complete K-Fold plan")
    animals = data.subjects.subject_ids
    if subject_ids is not None:
        animals = list(subject_ids)
    first = loaded[0]
    source_ids = set(first.train_ids + first.val_ids + first.test_ids)
    missing = source_ids - set(animals)
    if missing:
        names = ", ".join(sorted(missing)[:5])
        raise ValueError(f"Reference animals are not registered: {names}; use the full cohort or disable REFERENCE_SPLITS")
    additional = sorted(set(animals) - source_ids)
    if not additional:
        return loaded
    extra = select_animals(data, additional)
    extra_folds = balanced_folds(
        extra, k=len(loaded), seed=config.split_seed,
        validation_fraction=config.cv_validation_fraction, strata=strata,
    )
    return merge_folds(loaded, extra_folds)
