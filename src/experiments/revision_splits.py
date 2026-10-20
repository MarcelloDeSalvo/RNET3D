"""Shared animal folds for TBI pretraining, MouseFull and reviewer controls."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pandas as pd

from .data import PreparedData, SubjectSplit
from .splits import (
    balanced_folds, merge_folds, project_folds, read_selection, reference_fold_order,
    reorder_folds, select_animals, validate_folds, modality_annotation_strata,
)
from ..evaluation.k_fold import FoldSplit


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
    use_validation: bool = True,
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
        groups = modality_annotation_strata(mouse_full)
        supplied = read_selection(
            selected_path, seed=seed, validation_fraction=validation_fraction,
            use_validation=use_validation, strata=groups,
        )
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
            extra = balanced_folds(
                sham, k=k, seed=seed, validation_fraction=validation_fraction,
                use_validation=use_validation,
            )
            order = reference_fold_order(selected_path, k)
            extra = reorder_folds(extra, order)
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
            use_validation=use_validation,
        )
        sham_folds = balanced_folds(
            sham, k=k, seed=seed, validation_fraction=validation_fraction, anchor=anchor_sham,
            use_validation=use_validation,
        )
        mouse_folds = merge_folds(d1_folds, sham_folds)

    rat_folds: list[FoldSplit] = []
    if rats is not None:
        rat_ids = rats.subjects.subject_ids
        if rat_folds_file is not None:
            loaded = read_selection(
                rat_folds_file, seed=seed, validation_fraction=validation_fraction,
                use_validation=use_validation,
            )
            if not isinstance(loaded, list):
                raise ValueError("Rat folds require an outer CV plan")
            rat_folds = loaded
            validate_folds(rat_folds, rat_ids)
        else:
            rat_folds = balanced_folds(
                rats, k=len(rat_ids), seed=seed, validation_fraction=validation_fraction,
                use_validation=use_validation,
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
