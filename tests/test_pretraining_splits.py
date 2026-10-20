"""Keep pretrained animals in their roles without reserving a second holdout."""

import json
import shutil
from dataclasses import replace

import pytest
import pandas as pd

from test_experiments import experiment
from test_pretraining_leakage import source_run
from src.experiments.config import DatasetSpec, InitializationConfig
from src.experiments.data import (
    SubjectSplit, make_cv_splits, make_pretrained_cv_splits, make_split,
    prepare_data, resolve_training_split,
)
from src.experiments.initialization import pretraining_audit


def write_roles(data, folder, roles):
    """Save source identities including unused test animals; no weights are fitted."""
    settings = source_run(data, folder, roles["train"], roles["validation"])
    (folder / "splits.json").write_text(json.dumps(roles), encoding="utf-8")
    data.subjects.manifest(data.subjects.subject_ids).to_csv(folder / "subjects.csv", index=False)
    return settings


@pytest.fixture
def pretrained_data(experiment, tmp_path):
    data = prepare_data(experiment)
    roles = {"train": ["D1/SUBJ_0000"], "validation": ["D1/SUBJ_0001"], "test": ["D1/SUBJ_0002"]}
    settings = write_roles(data, tmp_path / "source", roles)
    return replace(data, config=replace(experiment, initialization=settings))


def add_cohort(data, count=8):
    """Add independent miniature animals while retaining a longitudinal D1 animal."""
    root = data.config.data.root
    for index in range(count):
        destination = root / "D3" / f"SUBJ_{index + 1000:04d}"
        shutil.copytree(root / "D1" / "SUBJ_0000", destination)
        metadata_file = destination / "metadata.json"
        metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
        metadata.update(dataset="D3", normalized_id=destination.name, original_id=f"new-{index}")
        if index % 2:
            metadata["annotated_regions"] = ["Les"]
        metadata_file.write_text(json.dumps(metadata), encoding="utf-8")
    datasets = (DatasetSpec(name="D1"), DatasetSpec(name="D3"))
    config = replace(data.config, data=replace(data.config.data, datasets=datasets))
    return prepare_data(config)


def test_fixed_split_inherits_roles_and_retains_every_longitudinal_scan(pretrained_data):
    split = resolve_training_split(pretrained_data)
    assert split == pretrained_data.make_split()
    assert split.train == ("D1/SUBJ_0000", "D1/SUBJ_0003")
    assert split.validation == ("D1/SUBJ_0001",)
    assert split.test == ("D1/SUBJ_0002",)
    assert len(pretrained_data.subjects.scan_keys(split.train)) == 3
    protected = [*split.validation, *split.test]
    report = pretraining_audit(
        pretrained_data, pretrained_data.config.initialization, protected,
        validation_ids=split.validation,
    )
    assert report["shared_validation_animals"] == ["D1/SUBJ_0001"]


def test_additional_cohort_fills_overall_fractions_without_extra_holdouts(pretrained_data):
    data = add_cohort(pretrained_data)
    split = resolve_training_split(data)
    assert (len(split.train), len(split.validation), len(split.test)) == (7, 3, 2)
    assert "D1/SUBJ_0000" in split.train
    assert "D1/SUBJ_0001" in split.validation
    assert "D1/SUBJ_0002" in split.test
    assert split == resolve_training_split(data)
    split.validate(data.subjects)


def test_explicit_and_resumed_memberships_are_not_regenerated(pretrained_data, tmp_path):
    split = resolve_training_split(pretrained_data)
    folder = tmp_path / "target"
    folder.mkdir()
    path = folder / "splits.json"
    split.save(path)
    assert resolve_training_split(pretrained_data, split_file=path) == split
    assert resolve_training_split(pretrained_data, resume_run=folder) == split
    swapped = SubjectSplit(train=split.train, validation=split.test, test=split.validation)
    other = tmp_path / "requested.json"
    swapped.save(other)
    with pytest.raises(ValueError, match="differs from the resumed"):
        resolve_training_split(pretrained_data, resume_run=folder, split_file=other)
    with pytest.raises(ValueError, match="test animal was used"):
        resolve_training_split(pretrained_data, split_file=other)


def test_training_cannot_be_used_for_validation_even_with_shared_validation_enabled(pretrained_data):
    data = pretrained_data
    with pytest.raises(ValueError, match="Pretraining leakage"):
        pretraining_audit(data, data.config.initialization, ["D1/SUBJ_0000"], validation_ids=["D1/SUBJ_0000"])
    with pytest.raises(ValueError, match="Pretraining leakage"):
        pretraining_audit(data, data.config.initialization, ["D1/SUBJ_0001"])


def test_conflicting_variant_sources_are_rejected_before_selecting_a_split(pretrained_data, tmp_path):
    data = replace(pretrained_data, config=replace(pretrained_data.config, initialization=InitializationConfig()))
    roles = {"train": ["D1/SUBJ_0001"], "validation": ["D1/SUBJ_0000"], "test": ["D1/SUBJ_0002"]}
    other = write_roles(data, tmp_path / "other_source", roles)
    sources = [pretrained_data.config.initialization, other]
    with pytest.raises(ValueError, match="conflicting roles"):
        resolve_training_split(pretrained_data, initializations=sources)


def test_animal_renumbering_matches_original_identity_and_scan_path(pretrained_data):
    folder = pretrained_data.config.initialization.checkpoint.parent.parent
    for path in (folder / "splits.json", folder / "fold_0" / "split.json"):
        roles = json.loads(path.read_text(encoding="utf-8"))
        roles["train"] = ["old/animal"]
        path.write_text(json.dumps(roles), encoding="utf-8")
    manifest_file = folder / "subjects.csv"
    manifest = pd.read_csv(manifest_file)
    manifest.loc[manifest["Subject"] == "D1/SUBJ_0000", "Subject"] = "old/animal"
    manifest.to_csv(manifest_file, index=False)
    split = resolve_training_split(pretrained_data)
    assert "D1/SUBJ_0000" in split.train


def test_missing_source_test_manifest_cannot_silently_change_roles(pretrained_data):
    path = pretrained_data.config.initialization.checkpoint.parent.parent / "subjects.csv"
    manifest = pd.read_csv(path)
    manifest.loc[manifest["Subject"] != "D1/SUBJ_0002"].to_csv(path, index=False)
    with pytest.raises(ValueError, match="does not cover declared"):
        resolve_training_split(pretrained_data)


def test_scratch_memberships_use_modality_and_anatomy_strata(experiment):
    from src.experiments.splits import modality_annotation_strata

    data = prepare_data(experiment)
    strata = modality_annotation_strata(data)
    expected = make_split(data.subjects, seed=experiment.split_seed, strata=strata)
    assert resolve_training_split(data) == expected


def test_completed_two_way_source_is_accepted_without_validation_and_still_checks_leakage(experiment, tmp_path):
    training = replace(experiment.training, epochs=3, monitor="loss", fixed_epochs=True)
    data = prepare_data(replace(experiment, training=training))
    settings = source_run(data, tmp_path / "fixed_source", ["D1/SUBJ_0000"], [])
    state_path = settings.checkpoint.parent / "training_state.json"
    state_path.write_text(json.dumps({
        "completed_epochs": 2, "selected_checkpoint": "latest/selected_2.weights.h5",
    }))
    with pytest.raises(ValueError, match="completed budget"):
        pretraining_audit(data, settings, ["D1/SUBJ_0002"])
    state_path.write_text(json.dumps({
        "completed_epochs": 3, "selected_checkpoint": "latest/selected_3.weights.h5",
    }))
    audit = pretraining_audit(data, settings, ["D1/SUBJ_0002"])
    assert str(state_path) in audit["sha256"]
    with pytest.raises(ValueError, match="Pretraining leakage"):
        pretraining_audit(data, settings, ["D1/SUBJ_0000"])


def cv_sources(data, folder):
    """Two source folds with one selection animal and two outer holdouts each."""
    roles = [
        {"train": ["D1/SUBJ_0000"], "validation": ["D1/SUBJ_0001"], "test": ["D1/SUBJ_0002", "D1/SUBJ_0003"]},
        {"train": ["D1/SUBJ_0002"], "validation": ["D1/SUBJ_0003"], "test": ["D1/SUBJ_0000", "D1/SUBJ_0001"]},
    ]
    source_run(data, folder, roles[0]["train"], roles[0]["validation"])
    (folder / "splits.json").write_text(json.dumps(roles), encoding="utf-8")
    data.subjects.manifest(data.subjects.subject_ids).to_csv(folder / "subjects.csv", index=False)
    settings = {}
    for index, selection in enumerate(roles):
        fold_dir = folder / f"fold_{index}"
        fold_dir.mkdir(exist_ok=True)
        # Source fixed-style fold files can omit test; the selected CV root supplies it.
        payload = {"train": selection["train"], "validation": selection["validation"]}
        (fold_dir / "split.json").write_text(json.dumps(payload), encoding="utf-8")
        checkpoint = fold_dir / "best.weights.h5"
        checkpoint.write_bytes(b"provenance-only checkpoint")
        settings[index] = InitializationConfig(mode="weights", checkpoint=checkpoint)
    return settings, roles


def test_pretrained_cv_preserves_source_roles_and_balances_new_animals(experiment, tmp_path):
    source_config = replace(experiment, training=replace(experiment.training, k=2))
    source_data = prepare_data(source_config)
    settings, source_roles = cv_sources(source_data, tmp_path / "source_cv")
    target = replace(source_data, config=replace(source_config, initialization_by_fold=settings))
    data = add_cohort(target)
    strata = data.annotation_groups(include_lesion=False)
    folds = make_pretrained_cv_splits(data, strata=strata)
    assert folds == make_pretrained_cv_splits(data, strata=strata)
    held_out = []
    for fold, source in zip(folds, source_roles):
        assert set(source["train"]).issubset(fold.train_ids)
        assert set(source["validation"]).issubset(fold.val_ids)
        assert set(source["test"]).issubset(fold.test_ids)
        held_out.extend(fold.test_ids)
        new_groups = [strata[animal] for animal in fold.test_ids if animal.startswith("D3/")]
        assert new_groups.count("complete") == 2
        assert new_groups.count("partial") == 2
        protected = [*fold.val_ids, *fold.test_ids]
        pretraining_audit(data, settings[fold.index], protected, validation_ids=fold.val_ids)
    assert sorted(held_out) == data.subjects.subject_ids


def test_one_overlapping_checkpoint_cannot_be_reused_for_full_cv(pretrained_data):
    config = replace(pretrained_data.config, training=replace(pretrained_data.config.training, k=2))
    data = replace(pretrained_data, config=config)
    with pytest.raises(ValueError, match="one matching source checkpoint per fold"):
        make_pretrained_cv_splits(data)


def test_sources_from_independent_cohorts_keep_ordinary_cv(experiment, tmp_path):
    config = replace(experiment, training=replace(experiment.training, k=2))
    data = prepare_data(config)
    source = source_run(data, tmp_path / "independent", ["independent"], ["other-independent"])
    target = replace(data, config=replace(config, initialization=source))
    expected = make_cv_splits(data.subjects.subject_ids, k=2, seed=config.split_seed, validation_fraction=config.cv_validation_fraction)
    assert make_pretrained_cv_splits(target) == expected


def test_tiny_additional_cv_cohort_still_has_one_outer_holdout(experiment, tmp_path):
    config = replace(experiment, training=replace(experiment.training, k=2))
    source_data = prepare_data(config)
    settings, _ = cv_sources(source_data, tmp_path / "source_cv")
    target = replace(source_data, config=replace(config, initialization_by_fold=settings))
    data = add_cohort(target, count=1)

    folds = make_pretrained_cv_splits(data)

    held_out = [animal for fold in folds for animal in fold.test_ids]
    assert sorted(held_out) == data.subjects.subject_ids


def test_ancestor_selection_animals_can_be_fitted_in_a_later_stage(pretrained_data, tmp_path):
    data = replace(pretrained_data, config=replace(pretrained_data.config, initialization=InitializationConfig()))
    parent_roles = {"train": ["D1/SUBJ_0000"], "validation": ["D1/SUBJ_0001"], "test": ["D1/SUBJ_0002"]}
    parent = write_roles(data, tmp_path / "parent", parent_roles)
    child_data = replace(data, config=replace(data.config, initialization=parent))
    child_roles = {"train": ["D1/SUBJ_0000", "D1/SUBJ_0001"], "validation": ["D1/SUBJ_0003"], "test": ["D1/SUBJ_0002"]}
    child = write_roles(child_data, tmp_path / "child", child_roles)
    target = replace(data, config=replace(data.config, initialization=child))

    split = resolve_training_split(target)

    assert "D1/SUBJ_0001" in split.train
    assert split.validation == ("D1/SUBJ_0003",)
    assert split.test == ("D1/SUBJ_0002",)


def test_too_few_independent_animals_is_rejected_before_splitting(pretrained_data):
    folder = pretrained_data.config.initialization.checkpoint.parent.parent
    roles = {"train": ["D1/SUBJ_0000", "D1/SUBJ_0002", "D1/SUBJ_0003"], "validation": ["D1/SUBJ_0001"]}
    for path in (folder / "splits.json", folder / "fold_0" / "split.json"):
        path.write_text(json.dumps(roles), encoding="utf-8")

    with pytest.raises(ValueError, match="too few independent"):
        resolve_training_split(pretrained_data)
