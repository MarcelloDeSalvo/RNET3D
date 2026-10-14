"""Verify explicit animal split fractions and full-volume annotation semantics."""

import ast
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import pytest

from test_experiments import experiment
from src.experiments.config import DatasetSpec, ExperimentConfig, FixedSplitConfig, LossConfig
from src.experiments.data import SubjectSplit, annotation_values, make_split, prepare_data


def test_default_fractions_apply_to_the_complete_animal_cohort():
    """A 100-animal cohort gets 70/20/10, rather than sequential 72/20/8."""
    animals = [f"animal-{number:03d}" for number in range(100)]
    index = SimpleNamespace(subject_ids=animals, scan_keys=lambda ids: list(ids))
    split = make_split(index, seed=42)

    assert (len(split.train), len(split.validation), len(split.test)) == (70, 20, 10)
    assert split == make_split(index, seed=42)
    assert FixedSplitConfig().train_fraction == pytest.approx(0.7)


@pytest.mark.parametrize("fraction", [True, 0, -0.1, 1, float("nan"), float("inf"), "0.2"])
def test_invalid_split_fractions_are_rejected(fraction):
    settings = FixedSplitConfig(validation_fraction=fraction)
    with pytest.raises(ValueError, match="Split fractions"):
        settings.validate()


def test_holdout_fractions_must_leave_training_animals():
    settings = FixedSplitConfig(validation_fraction=0.7, test_fraction=0.3)
    with pytest.raises(ValueError, match="leave training"):
        settings.validate()


def test_prepared_split_uses_saved_configuration_and_keeps_longitudinal_scans(experiment):
    """Custom ratios reach the splitter, while all scans inherit one animal role."""
    settings = FixedSplitConfig(validation_fraction=0.5, test_fraction=0.25)
    config = replace(experiment, fixed_split=settings)
    data = prepare_data(config)
    split = data.make_split()

    assert (len(split.train), len(split.validation), len(split.test)) == (1, 2, 1)
    roles = (split.train, split.validation, split.test)
    scan_sets = [set(data.subjects.scan_keys(role)) for role in roles]
    longitudinal = {("D1_T1w", "SUBJ_0003_7days"), ("D1_T1w", "SUBJ_0003_14days")}
    assert sum(longitudinal <= scans for scans in scan_sets) == 1
    summary = data.split_summary(split)
    assert summary["AnimalFraction"].tolist() == [0.25, 0.5, 0.25]
    assert summary["Scans"].sum() == 5


def test_small_cohort_rounding_is_reported_as_actual_fractions(experiment):
    data = prepare_data(experiment)
    split = data.make_split()
    assert (len(split.train), len(split.validation), len(split.test)) == (2, 1, 1)
    assert data.split_summary(split)["AnimalFraction"].tolist() == [0.5, 0.25, 0.25]


def test_saved_manual_memberships_override_generation_fractions(experiment, tmp_path):
    data = prepare_data(experiment)
    animals = data.subjects.subject_ids
    manual = SubjectSplit(train=tuple(animals[:1]), validation=tuple(animals[1:2]), test=tuple(animals[2:]))
    path = tmp_path / "manuscript_split.json"
    manual.save(path)

    changed = replace(experiment, fixed_split=FixedSplitConfig(validation_fraction=0.5, test_fraction=0.25))
    changed_data = prepare_data(changed)
    assert SubjectSplit.load(path, changed_data.subjects) == manual
    assert SubjectSplit.load(path, changed_data.subjects) != changed_data.make_split()


def test_study_selection_uses_configured_fractions_and_saved_memberships(experiment, tmp_path):
    from src.experiments.ablation import study_selection

    config = replace(experiment, fixed_split=FixedSplitConfig(validation_fraction=0.5, test_fraction=0.25))
    data = prepare_data(config)
    selection = study_selection(data, mode="fixed_split", study_dir=tmp_path)
    assert selection.split == data.make_split()
    selection.split.save(tmp_path / "splits.json")

    changed_data = replace(data, config=experiment)
    resumed = study_selection(changed_data, mode="fixed_split", study_dir=tmp_path, resume=True)
    assert resumed.split == selection.split


def test_new_contracts_roundtrip_and_old_saved_configuration_gets_defaults(experiment):
    payload = json.loads(json.dumps(experiment.to_dict()))
    assert payload["fixed_split"] == {"validation_fraction": 0.2, "test_fraction": 0.1}
    assert payload["data"]["datasets"][0]["annotation_scope"] == "full_volume"
    assert ExperimentConfig.from_dict(payload) == experiment

    payload.pop("fixed_split")
    for dataset in payload["data"]["datasets"]:
        dataset.pop("annotation_scope")
    restored = ExperimentConfig.from_dict(payload)
    assert restored == experiment
    restored.validate()


@pytest.mark.parametrize("previous_flag", [False, True])
def test_saved_sham_flag_is_removed_when_loading_configuration(experiment, previous_flag):
    """Previously saved settings load without exposing the retired toggle."""
    payload = json.loads(json.dumps(experiment.to_dict()))
    payload["data"]["datasets"][0]["sham_lesion_absent"] = previous_flag

    restored = ExperimentConfig.from_dict(payload)

    assert restored == experiment
    assert "sham_lesion_absent" not in restored.to_dict()["data"]["datasets"][0]


def test_previous_default_configuration_can_resume_without_rewriting_its_contract(experiment, tmp_path):
    from src.experiments.training import save_provenance, validate_resume

    data = prepare_data(experiment)
    save_provenance(experiment, data, tmp_path)
    path = tmp_path / "experiment.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.pop("fixed_split")
    payload["model"].pop("multiscale")
    payload["model"].pop("deep_supervision")
    for dataset in payload["data"]["datasets"]:
        dataset.pop("annotation_scope")
    path.write_text(json.dumps(payload), encoding="utf-8")

    validate_resume(data, tmp_path)
    changed = replace(experiment, fixed_split=FixedSplitConfig(validation_fraction=0.3, test_fraction=0.1))
    validate_resume(replace(data, config=changed), tmp_path)

    # Newly saved generation policies are explicit and remain checked strictly.
    path.write_text(json.dumps(experiment.to_dict()), encoding="utf-8")
    with pytest.raises(ValueError, match="configuration differs"):
        validate_resume(replace(data, config=changed), tmp_path)


def test_older_study_contract_can_resume_with_new_declarations(experiment, tmp_path, monkeypatch):
    """Adding defaults must not invalidate an otherwise unchanged saved study."""
    from src.evaluation.k_fold import TrainingResult
    from src.experiments import ablation

    def runner(data, split, *, resume_run):
        folder = resume_run
        if folder is None:
            folder = data.config.training.start_run()
        return TrainingResult(run_dir=folder)

    monkeypatch.setattr(ablation, "run_training", runner)
    data = prepare_data(experiment)
    split = data.make_split()
    variants = [ablation.AblationVariant("reference")]
    folder = tmp_path / "study"
    ablation.run_ablation(data, variants, study_dir=folder, split=split)
    path = folder / "study.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["base"].pop("fixed_split")
    payload["base"]["model"].pop("multiscale")
    payload["base"]["model"].pop("deep_supervision")
    for dataset in payload["base"]["data"]["datasets"]:
        dataset.pop("annotation_scope")
    path.write_text(json.dumps(payload), encoding="utf-8")

    ablation.run_ablation(data, variants, study_dir=folder, split=split, resume=True)
    changed_split = SubjectSplit(train=split.train, validation=split.test, test=split.validation)
    with pytest.raises(ValueError, match="splits differ"):
        ablation.run_ablation(data, variants, study_dir=folder, split=changed_split, resume=True)

    payload["base"]["loss"].pop("gradient_fix_version")
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="objective changed"):
        ablation.run_ablation(data, variants, study_dir=folder, split=split, resume=True)


def test_old_partial_label_objective_loads_and_trains_with_version_one(experiment):
    import tensorflow as tf
    from src.experiments.training import build_model

    payload = experiment.to_dict()
    payload["loss"].pop("gradient_fix_version")
    old_config = ExperimentConfig.from_dict(payload)
    assert old_config.loss.gradient_fix_version == 1
    old_config.validate()
    data = prepare_data(old_config, require_training_loss=False)
    assert len(data.subjects.subject_ids) == 4

    training_data = prepare_data(old_config)
    assert len(training_data.subjects.subject_ids) == 4
    model = build_model(old_config)
    assert model.output_names[0] == "regions_logits"

    targets = tf.constant([[[[[1., 0., -1.]]]]])
    logits = tf.constant([[[[[0., 0., 100.]]]]])
    from src.evaluation.losses import diceCELoss_gradient_fix

    expected = diceCELoss_gradient_fix(gamma=0.8, batch_wise=True, version=1)
    actual = model.loss["regions_logits"](targets, logits)
    np.testing.assert_allclose(actual.numpy(), expected(targets, logits).numpy())


def test_resume_rejects_silent_partial_label_loss_version_changes(experiment, tmp_path):
    from src.experiments.training import save_provenance, validate_resume

    data = prepare_data(experiment)
    save_provenance(experiment, data, tmp_path)
    path = tmp_path / "experiment.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["loss"].pop("gradient_fix_version")
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="objective changed"):
        validate_resume(data, tmp_path)


def test_old_version_one_resume_is_accepted_and_reverse_switch_is_rejected(experiment, tmp_path):
    from src.experiments.training import save_provenance, validate_resume

    version_one = replace(experiment, loss=replace(experiment.loss, gradient_fix_version=1))
    data = prepare_data(version_one)
    save_provenance(version_one, data, tmp_path)
    path = tmp_path / "experiment.json"
    payload = json.loads(path.read_text())
    payload["loss"].pop("gradient_fix_version")
    path.write_text(json.dumps(payload))
    validate_resume(data, tmp_path)

    payload["loss"]["gradient_fix_version"] = 2
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="objective changed"):
        validate_resume(data, tmp_path)


def test_old_legacy_loss_configuration_still_resumes(experiment, tmp_path):
    from src.experiments.training import save_provenance, validate_resume

    complete = DatasetSpec(available_labels=(0, 1, 2))
    config = replace(
        experiment, loss=LossConfig(regions="dice_ce"),
        data=replace(experiment.data, datasets=(complete,)),
    )
    data = prepare_data(config)
    save_provenance(config, data, tmp_path)
    path = tmp_path / "experiment.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["loss"].pop("gradient_fix_version")
    path.write_text(json.dumps(payload), encoding="utf-8")

    validate_resume(data, tmp_path)


@pytest.mark.parametrize("scope", ["slices", "scribbles", "partial_volume", None])
def test_sparse_annotation_scope_is_rejected_before_loading(experiment, scope):
    dataset = DatasetSpec(annotation_scope=scope)
    config = replace(experiment, data=replace(experiment.data, datasets=(dataset,)))
    with pytest.raises(ValueError, match="spatial validity mask"):
        prepare_data(config)


def test_sparse_metadata_cannot_be_overridden_by_full_volume_available_labels(experiment):
    path = experiment.data.root / "D1" / "SUBJ_0000" / "metadata.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    metadata["annotation_scope"] = "slices"
    path.write_text(json.dumps(metadata), encoding="utf-8")
    dataset = DatasetSpec(available_labels=(0, 1, 2))
    config = replace(experiment, data=replace(experiment.data, datasets=(dataset,)))

    with pytest.raises(ValueError, match="annotation scope"):
        prepare_data(config)


@pytest.mark.parametrize("labels", [(1, 2), (0, 1, 1), (0, True), (0, -1), (0, 1.0)])
def test_annotation_overrides_require_valid_source_values(labels):
    dataset = DatasetSpec(available_labels=labels)
    with pytest.raises(ValueError, match="available_labels"):
        dataset.validate()


def test_available_empty_roi_remains_supervised_and_missing_roi_is_masked(experiment):
    """An empty annotated mask means known negatives, not missing annotation."""
    path = experiment.data.root / "D1" / "SUBJ_0000" / "Labels.nii.gz"
    source = nib.load(path)
    empty_mask = np.zeros(source.shape, dtype=np.int16)
    nib.save(nib.Nifti1Image(empty_mask, source.affine), path)
    data = prepare_data(experiment)

    available_scan = data.loader.get_subject(("D1_T1w", "SUBJ_0000"))
    assert annotation_values(available_scan) == {0, 1, 2}
    generator = data.make_generator(["D1/SUBJ_0000", "D1/SUBJ_0001"], training=False)
    regions = generator[0][1][0]
    assert np.all(regions[0, ..., 1:] == 0)
    assert np.all(regions[1, ..., 2] == -1)
    meaning = data.annotation_summary().set_index("Class")["AnnotationMeaning"]
    assert "task remainder" in meaning["Background"]
    assert "known negatives" in meaning["CC"]


@pytest.mark.parametrize("value", ["false", 0, 1])
def test_lesion_annotation_flag_is_not_coerced_from_other_types(experiment, value):
    data = prepare_data(experiment)
    scan = data.loader.get_subject(("D1_T1w", "SUBJ_0000"))
    scan.metadata["has_lesion_annotation"] = value
    with pytest.raises(ValueError, match="must be a boolean"):
        annotation_values(scan)


@pytest.mark.parametrize("filename", [
    "01_tbi_mouse_training.ipynb", "03_rat_finetuning.ipynb", "05_mouse_full_training.ipynb",
    "07_ablation.ipynb", "07f_ablation.ipynb", "08_unet_comparison.ipynb",
])
def test_fixed_split_notebooks_expose_policy_and_use_shared_selection(filename):
    path = Path(__file__).resolve().parents[1] / "notebook" / filename
    notebook = json.loads(path.read_text(encoding="utf-8"))
    source = "\n".join("".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code")
    ast.parse(source)
    assert "fixed_split=split_settings" in source
    assert "split_settings = FixedSplitConfig(" in source
    assert 'annotation_scope="full_volume"' in source
    selectors = ("resolve_training_selection(", "study_selection(")
    assert any(selector in source for selector in selectors)
