"""Anatomy coverage strata and deterministic, independent outer animal folds."""

from collections import Counter
from dataclasses import replace

import pytest

from test_experiments import experiment
from src.experiments.config import LabelConfig
from src.experiments.data import fold_annotation_summary, make_cv_splits, prepare_data


def test_anatomy_groups_exclude_lesion_and_require_every_scan(experiment):
    data = prepare_data(experiment)
    complete = data.loader.get_subject(("D1_T1w", "SUBJ_0000"))
    complete.unavailable_labels = [1]  # Anatomy complete, lesion annotation unavailable.
    groups = data.annotation_groups(include_lesion=False)
    assert groups == {"D1/SUBJ_0000": "complete", "D1/SUBJ_0001": "partial",
                      "D1/SUBJ_0002": "complete", "D1/SUBJ_0003": "partial"}
    assert data.annotation_groups(include_lesion=True)["D1/SUBJ_0000"] == "partial"

    longitudinal = data.subjects.scan_keys(["D1/SUBJ_0003"])
    data.loader.get_subject(longitudinal[0]).unavailable_labels = []
    assert data.annotation_groups()["D1/SUBJ_0003"] == "partial"
    data.loader.get_subject(longitudinal[1]).unavailable_labels = []
    assert data.annotation_groups()["D1/SUBJ_0003"] == "complete"
    assert data.annotation_groups(["D1/SUBJ_0003"]) == {"D1/SUBJ_0003": "complete"}


def test_annotation_groups_require_all_merged_class_sources(experiment):
    labels = LabelConfig(mapping={0: 0, 1: 1, 2: 2, 12: 2}, names=("Background", "Lesion", "CC"))
    data = prepare_data(replace(experiment, labels=labels))
    assert set(data.annotation_groups().values()) == {"partial"}
    scan = data.loader.get_subject(("D1_T1w", "SUBJ_0000"))
    assert scan.unavailable_labels == [12]
    scan.unavailable_labels = []
    assert data.annotation_groups()["D1/SUBJ_0000"] == "complete"


def test_balanced_outer_folds_spread_groups_without_repeating_animals():
    ids = [f"animal_{index:02d}" for index in range(17)]
    groups = {animal: "partial" for animal in ids}
    for animal in ids[:6]:
        groups[animal] = "complete"
    folds = make_cv_splits(ids, k=5, seed=42, strata=groups)
    repeated = make_cv_splits(list(reversed(ids)), k=5, seed=42, strata=groups)
    assert folds == repeated
    counts = [Counter(groups[animal] for animal in fold.test_ids) for fold in folds]
    for group in ("complete", "partial"):
        values = [count[group] for count in counts]
        assert max(values) - min(values) <= 1
    assert sorted(animal for fold in folds for animal in fold.test_ids) == sorted(ids)
    sizes = [len(fold.test_ids) for fold in folds]
    assert max(sizes) - min(sizes) <= 1
    for fold in folds:
        fold.validate(set(ids))
        assert fold.train_ids and fold.val_ids


def test_balance_summary_counts_biological_animals_not_longitudinal_scans(experiment):
    data = prepare_data(experiment)
    groups = data.annotation_groups()
    folds = make_cv_splits(data.subjects.subject_ids, k=2, seed=21, strata=groups)
    summary = fold_annotation_summary(folds, data)
    holdout = summary.loc[summary["Role"] == "holdout"]
    assert holdout["Complete"].tolist() == [1, 1]
    assert holdout["Partial"].tolist() == [1, 1]
    assert holdout[["Complete", "Partial"]].sum().sum() == 4
    assert sum(len(data.subjects.scan_keys(fold.test_ids)) for fold in folds) == 5


def test_rare_annotation_group_is_preserved_even_when_not_in_every_fold():
    ids = [f"animal_{index}" for index in range(11)]
    groups = {animal: "partial" for animal in ids}
    groups[ids[0]] = "complete"
    with pytest.warns(UserWarning, match="least populated class"):
        folds = make_cv_splits(ids, k=3, seed=19, strata=groups)
    assert sum(ids[0] in fold.test_ids for fold in folds) == 1
    assert sorted(animal for fold in folds for animal in fold.test_ids) == sorted(ids)


@pytest.mark.parametrize("groups", [
    {"a": "complete"},
    {"a": "complete", "b": "partial", "extra": "partial"},
    {"a": "", "b": "partial"},
    {"a": None, "b": "partial"},
])
def test_invalid_strata_fail_before_splitting(groups):
    with pytest.raises(ValueError, match="strata"):
        make_cv_splits(["a", "b"], k=2, seed=42, strata=groups)
