"""Preserve historical outer holdouts across training, transfer and ablations."""

import json
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from test_revision_suite import revision_settings
from test_pretraining_splits import write_roles
from src.experiments.data import prepare_data, resolve_training_selection
from src.experiments.splits import project_folds, read_selection, reference_folds
from src.experiments import revision_suite as suite
from src.experiments.revision_summary import export_suite_summary


@pytest.fixture
def reference_file(revision_settings, tmp_path):
    """Two old train/val folds using the dataset's underscore ID spelling."""
    animals = [f"D1_SUBJ_{number:04d}" for number in range(6)]
    folds = []
    for index in range(2):
        test = animals[index::2]
        train = [animal for animal in animals if animal not in test]
        folds.append({"train": train, "val": test})
    path = tmp_path / "D1_splits_final.json"
    path.write_text(json.dumps(folds), encoding="utf-8")
    return path


def test_old_holdouts_are_preserved_and_inner_validation_uses_only_old_train(reference_file):
    original = reference_file.read_bytes()
    saved = json.loads(original)
    folds = read_selection(reference_file)
    for previous, current in zip(saved, folds):
        expected_test = [animal.replace("D1_", "D1/", 1) for animal in previous["val"]]
        expected_pool = {animal.replace("D1_", "D1/", 1) for animal in previous["train"]}
        assert current.test_ids == expected_test
        assert set(current.train_ids + current.val_ids) == expected_pool
        assert not set(current.train_ids).intersection(current.val_ids)
    assert folds == read_selection(reference_file)
    assert reference_file.read_bytes() == original


def test_explicit_three_role_export_preserves_resolved_roles_without_resplitting(reference_file, tmp_path):
    """Materializing the existing interpretation must not reduce outer tests."""
    original = json.loads(reference_file.read_bytes())
    resolved = read_selection(reference_file, seed=42, validation_fraction=0.2)
    path = tmp_path / "explicit_three_way.json"
    payload = [fold.to_dict() for fold in resolved]
    path.write_text(json.dumps(payload), encoding="utf-8")

    # Once saved explicitly, generation settings cannot move animals again.
    reloaded = read_selection(path, seed=999, validation_fraction=0.5)
    assert reloaded == resolved
    for old, fold in zip(original, reloaded):
        expected = [animal.replace("D1_", "D1/", 1) for animal in old["val"]]
        assert fold.test_ids == expected
        assert set(fold.val_ids).isdisjoint(fold.test_ids)


def test_fixed_and_cv_selections_share_the_same_reference_roles(revision_settings, reference_file):
    plan = suite.prepare_suite(revision_settings)
    data = plan.data["d1"]
    cv = resolve_training_selection(data, k_fold=True, reference_file=reference_file)
    for index, fold in enumerate(cv.folds):
        fixed = resolve_training_selection(data, reference_file=reference_file, reference_fold=index)
        assert fixed.split.train == tuple(fold.train_ids)
        assert fixed.split.validation == tuple(fold.val_ids)
        assert fixed.split.test == tuple(fold.test_ids)


def test_mouse_full_extends_reference_without_changing_d1_roles(revision_settings, reference_file):
    settings = replace(revision_settings, reference_splits=reference_file)
    plan = suite.prepare_suite(settings)
    full = reference_folds(plan.data["mouse_full"], reference_file)
    d1 = reference_folds(plan.data["d1"], reference_file)
    assert project_folds(full, plan.data["d1"].subjects.subject_ids) == d1
    assert full == plan.splits.mouse_full
    assert d1 == plan.splits.d1
    assert resolve_training_selection(plan.data["mouse_full"], reference_file=reference_file).split.test == tuple(full[0].test_ids)


def test_resumption_and_explicit_split_override_the_new_default(revision_settings, reference_file, tmp_path):
    data = suite.prepare_suite(revision_settings).data["d1"]
    saved = resolve_training_selection(data).split
    folder = tmp_path / "old_run"
    folder.mkdir()
    saved.save(folder / "splits.json")
    assert resolve_training_selection(data, resume_run=folder, reference_file=reference_file).split == saved
    assert resolve_training_selection(data, split_file=folder / "splits.json", reference_file=reference_file).split == saved


def test_pretraining_still_rejects_an_incompatible_reference(revision_settings, reference_file, tmp_path):
    data = suite.prepare_suite(revision_settings).data["d1"]
    fold = reference_folds(data, reference_file)[0]
    roles = fold.to_dict()
    source = write_roles(data, tmp_path / "source", roles)
    target = replace(data, config=replace(data.config, initialization=source))
    assert resolve_training_selection(target, reference_file=reference_file).split.test == tuple(fold.test_ids)
    with pytest.raises(ValueError, match="Pretraining leakage"):
        resolve_training_selection(target, reference_file=reference_file, reference_fold=1)
    with pytest.raises(ValueError, match="Pretraining leakage"):
        resolve_training_selection(target, k_fold=True, reference_file=reference_file)


def test_ablation_defaults_and_cv_override(revision_settings, reference_file):
    settings = replace(revision_settings, reference_splits=reference_file)
    plan = suite.prepare_suite(settings)
    controls = [job for job in plan.jobs if job.group.endswith("_ablation")]
    assert len(controls) == 7
    assert all(job.kind == "fixed_split" for job in controls)
    assert all(job.kind == "cross_validation" for job in plan.jobs if job.name in ("d1_reference", "mouse_full_reference"))
    repeated = suite.prepare_suite(replace(settings, ablation_mode="cross_validation"))
    assert all(job.kind == "cross_validation" for job in repeated.jobs if job.name.startswith("d1_no_"))


def test_fixed_controls_start_from_scratch_and_preserve_reference_roles(revision_settings, reference_file, monkeypatch):
    from src.experiments import training
    from types import SimpleNamespace

    plan = suite.prepare_suite(replace(revision_settings, reference_splits=reference_file, reference_fold=1))
    selected = plan.splits.d1[1]
    called = []

    def run(data, split, **kwargs):
        called.append((data.config, split, kwargs))
        return SimpleNamespace(run_dir=plan.folder)

    monkeypatch.setattr(training, "run_training", run)
    job = next(job for job in plan.jobs if job.name == "d1_no_attention")
    suite._execute_job(plan, job, callbacks=[])
    config, split, options = called[0]
    assert split.test == tuple(selected.test_ids)
    assert split.validation == tuple(selected.val_ids)
    assert config.training.seed == job.seed + 1
    assert options["callbacks"] == []

    from src.experiments.config import InitializationConfig

    def source(*args, **kwargs):
        pytest.fail("Mouse ablations must not load a pretrained checkpoint")

    monkeypatch.setattr(InitializationConfig, "from_run", staticmethod(source))
    job = next(job for job in plan.jobs if job.name == "mouse_full_no_classifier")
    suite._execute_job(plan, job, callbacks=[])
    assert called[1][0].initialization.mode == "scratch"
    assert called[1][0].initialization_by_fold == {}
    assert called[1][1].test == tuple(plan.splits.mouse_full[1].test_ids)


def test_suite_reserves_pretraining_for_independent_rats(revision_settings, monkeypatch):
    from src.experiments.config import InitializationConfig

    plan = suite.prepare_suite(revision_settings)

    def forbidden(*args, **kwargs):
        pytest.fail("No mouse job may resolve source weights")

    monkeypatch.setattr(InitializationConfig, "from_run", staticmethod(forbidden))
    for job in plan.jobs:
        if job.task == "rats":
            assert job.source == f"d1_reference/seed_{job.seed}"
            continue
        assert job.source is None
        # Refit duration is read from CV reports; it does not affect initialization.
        monkeypatch.setattr(suite, "_refit_epochs", lambda *args: 1)
        config = suite.job_configuration(plan, job)
        assert config.initialization.mode == "scratch"
        assert config.initialization_by_fold == {}

    mouse_job = next(job for job in plan.jobs if job.name == "mouse_full_reference")
    invalid = replace(mouse_job, source="d1_reference/seed_42")
    with pytest.raises(ValueError, match="start from scratch"):
        suite.job_configuration(plan, invalid)


@pytest.mark.parametrize("reference_fold", [0, 1])
def test_fixed_report_reuses_only_matching_reference_fold(revision_settings, reference_file, reference_fold):
    settings = replace(revision_settings, reference_splits=reference_file, reference_fold=reference_fold)
    folder = suite.run_revision_suite(settings, plan_only=True)
    contract = json.loads((folder / "suite.json").read_text())
    tests = contract["splits"]["d1"][reference_fold]["test"]
    reference_tests = [fold["test"] for fold in contract["splits"]["d1"]]
    for name, role, groups in (
        ("d1_reference", "holdout", reference_tests),
        ("d1_no_attention", "test", [tests]),
    ):
        target = folder / "experiments" / name / "seed_42" / "summary"
        target.mkdir(parents=True)
        rows = []
        for fold, animals in enumerate(groups):
            for animal in animals:
                value = 0.8
                if name == "d1_reference" and fold != reference_fold:
                    value = 0.1
                for metric, unit, score in (("Dice", "fraction", value), ("ReferenceVolume", "mm3", 1.0), ("PredictionVolume", "mm3", 1.0)):
                    rows.append({
                        "Subject": animal, "Scan": animal.split("/")[-1], "Dataset": "D1", "Condition": "TBI",
                        "Head": "regions", "Region": "Lesion", "Label": 1, "Fold": fold,
                        "Metric": metric, "Unit": unit, "Value": score, "Status": "ok",
                        "DatasetRole": role, "Seed": 42,
                    })
        pd.DataFrame(rows).to_csv(target / "full_volume_metrics.csv", index=False)
    paper = export_suite_summary(folder, bootstrap_samples=100)
    main = pd.read_csv(paper / "main_results.csv")
    ablations = pd.read_csv(paper / "ablations.csv")
    assert main.loc[main.Task.eq("d1"), "Folds"].tolist() == [2]
    matched = ablations.loc[ablations.Task.eq("d1")]
    assert set(matched.Variant) == {"reference", "no_attention"}
    assert matched.Folds.tolist() == [1, 1]
    assert set(matched.ReferenceFold) == {reference_fold}
    assert set(pd.to_numeric(matched.LesionDice)) == {0.8}
    candidate = matched.loc[matched.Variant.eq("no_attention")].iloc[0]
    assert candidate["PairedAnimals"] == len(tests)
    assert candidate["MeanDifference"] == pytest.approx(0)
