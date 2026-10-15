"""Check matched settings, routing, recovery contracts and animal aggregation."""

import json
import ast
from pathlib import Path
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from test_experiments import experiment
from src.experiments import ablation
from src.experiments.config import InferenceConfig, ModelConfig
from src.experiments.data import make_cv_splits, make_split, prepare_data
from src.evaluation.k_fold import FoldTiming, TrainingResult


@pytest.fixture
def prepared(experiment):
    model = ModelConfig(filters=(2, 4, 8, 16, 32))
    training = replace(experiment.training, data_multiplier=3, k=2)
    preprocessing = replace(experiment.preprocessing, augment=True, random_crop=True)
    config = replace(experiment, model=model, training=training, preprocessing=preprocessing)
    return prepare_data(config)


def test_variants_change_only_the_requested_setting(prepared):
    base = prepared.config
    variants = [ablation.AblationVariant("reference"), ablation.AblationVariant("no_attention", "attention", False)]
    plan = ablation.ablation_plan(base, variants)
    assert plan["Category"].tolist() == ["reference", "architecture"]
    changed = variants[1].configuration(base)
    assert changed.model == replace(base.model, attention=False)
    assert changed.preprocessing == base.preprocessing
    assert changed.training == base.training
    assert changed.loss == base.loss
    assert changed.labels == base.labels
    assert base.model.attention


@pytest.mark.parametrize("variant", [
    ablation.AblationVariant("same", "residual", True),
    ablation.AblationVariant("bad_bool", "attention", 0),
    ablation.AblationVariant("bad_count", "data_multiplier", True),
    ablation.AblationVariant("unknown", "unimplemented", False),
    ablation.AblationVariant("bad_fusion", "deep_supervision", None),
    ablation.AblationVariant("bad_loss", "regions", "unknown"),
    ablation.AblationVariant("bad_version", "gradient_fix_version", True),
    ablation.AblationVariant("bad_version", "gradient_fix_version", 4),
    ablation.AblationVariant("../escape"),
])
def test_invalid_or_ineffective_variants_fail_before_running(prepared, variant):
    with pytest.raises(ValueError):
        variant.configuration(prepared.config)


def test_matched_runs_share_data_splits_seeds_and_resume_paths(prepared, tmp_path, monkeypatch):
    split = make_split(prepared.subjects, seed=42)
    variants = [ablation.AblationVariant("reference"), ablation.AblationVariant("no_attention", "attention", False)]
    calls = []

    def runner(data, selection, *, resume_run=None):
        config = data.config
        path = config.training.start_run() if resume_run is None else resume_run
        assert path.is_dir()
        calls.append((data, selection, resume_run))
        table = pd.DataFrame([{
            "Fold": 0, "DatasetRole": "test", "Subject": split.test[0], "Scan": "scan",
            "Dataset": "D1", "Region": "Lesion", "Metric": "Dice", "Value": 0.5,
            "Unit": "unitless", "Status": "ok",
        }])
        return TrainingResult(run_dir=path, evaluations={"regions": [table]}, timings=[FoldTiming(fold=0)])

    monkeypatch.setattr(ablation, "run_training", runner)
    folder = tmp_path / "study"
    result = ablation.run_ablation(prepared, variants, study_dir=folder, split=split, seeds=(11, 22))
    assert len(result.runs) == 4
    assert result.metrics["Seed"].unique().tolist() == [11, 22]
    for data, selection, resume_run in calls:
        assert data.loader is prepared.loader
        assert data.subjects is prepared.subjects
        assert selection is split
        assert data.config.split_seed == prepared.config.split_seed
        assert data.config.preprocessing == prepared.config.preprocessing
        assert resume_run is None
    assert (folder / "full_volume_metrics.csv").is_file()
    assert len(json.loads((folder / "runs.json").read_text())) == 4

    # Earlier studies omit the two fields whose defaults preserve masking.
    saved = json.loads((folder / "study.json").read_text())
    saved["base"]["loss"].pop("annotation_policy")
    for variant in saved["variants"]:
        variant.pop("disable_classifier")
    (folder / "study.json").write_text(json.dumps(saved))
    ablation.run_ablation(prepared, variants, study_dir=folder, split=split, seeds=(11, 22), resume=True)
    assert all(path is not None for _, _, path in calls[4:])
    with pytest.raises(ValueError, match="Resume study"):
        ablation.run_ablation(prepared, variants, study_dir=folder, split=split, seeds=(22, 11), resume=True)


def test_cv_routes_identical_outer_folds_to_every_variant(prepared, tmp_path, monkeypatch):
    folds = make_cv_splits(prepared.subjects.subject_ids, k=2, seed=13)
    variants = [ablation.AblationVariant("reference"), ablation.AblationVariant("no_residual", "residual", False)]
    received = []

    def runner(data, *, folds, resume_run):
        received.append(folds)
        return TrainingResult(run_dir=data.config.training.start_run())

    monkeypatch.setattr(ablation, "run_cross_validation", runner)
    ablation.run_ablation(prepared, variants, study_dir=tmp_path / "cv", mode="cross_validation", folds=folds)
    assert received == [folds, folds]


def test_animal_summary_retains_model_identity_and_ignores_missing_values(tmp_path):
    common = {"Variant": "reference", "Seed": 42, "Fold": 0, "DatasetRole": "test",
              "Subject": "rat", "Dataset": "D2", "Region": "Lesion", "Metric": "Dice", "Unit": "unitless"}
    table = pd.DataFrame([
        {**common, "Scan": "7days", "Value": 0.4},
        {**common, "Scan": "14days", "Value": 0.8},
        {**common, "Scan": "30days", "Value": np.nan},
        {**common, "Seed": 43, "Scan": "7days", "Value": 0.2},
    ])
    result = ablation.AblationResult(study_dir=tmp_path, metrics=table)
    summary = ablation.animal_metric_values(result)
    assert len(summary) == 2
    first = summary.loc[summary["Seed"] == 42]
    assert first["Value"].item() == pytest.approx(0.6)
    assert first["Scans"].item() == 2


def test_tiny_real_study_uses_core_training_and_reuses_completed_runs(prepared, tmp_path):
    """Fit new component/loss variants and recover their native evaluations."""
    preprocessing = replace(prepared.config.preprocessing, augment=False)
    inference = InferenceConfig(enabled=False, stride=16)
    model = replace(prepared.config.model, classifier=True)
    loss = replace(prepared.config.loss, gradient_fix_version=1)
    config = replace(prepared.config, model=model, loss=loss, preprocessing=preprocessing, inference=inference)
    data = replace(prepared, config=config)
    split = make_split(data.subjects, seed=42)
    variants = [
        ablation.AblationVariant("reference"),
        ablation.AblationVariant("no_multiscale", "multiscale", False),
        ablation.AblationVariant("no_deep_supervision", "deep_supervision", False),
        ablation.AblationVariant("no_classifier", "classifier", False),
        ablation.AblationVariant("v3", "gradient_fix_version", 3),
        ablation.AblationVariant("unmasked_no_classifier", "annotation_policy", "missing_as_background", disable_classifier=True),
    ]
    folder = tmp_path / "real_study"
    result = ablation.run_ablation(data, variants, study_dir=folder, split=split)
    assert len(result.runs) == 6
    assert result.timings.shape[0] == 6
    assert result.metrics.empty
    assert set(result.patch_metrics["Variant"]) == {variant.name for variant in variants}
    for run in result.runs.values():
        assert (run.run_dir / "fold_0" / "completed.json").is_file()
        assert np.isfinite(run.histories[0]["loss"]).all()

    resumed = ablation.run_ablation(data, variants, study_dir=folder, split=split, resume=True)
    for name, run in result.runs.items():
        assert resumed.runs[name].histories == run.histories
    pd.testing.assert_frame_equal(resumed.patch_metrics, result.patch_metrics)


@pytest.mark.parametrize("parameter,value", [
    ("multiscale", False), ("deep_supervision", False), ("gradient_fix_version", 3),
    ("regions", "dice_ce"),
])
def test_new_variants_replace_only_one_configuration_field(prepared, parameter, value):
    base = prepared.config
    variant = ablation.AblationVariant("changed", parameter, value)

    changed = variant.configuration(base)
    plan = ablation.ablation_plan(base, [ablation.AblationVariant("reference"), variant])

    expected = base.to_dict()
    section = "model"
    if variant.category == "loss":
        section = "loss"
    expected[section][parameter] = value
    assert changed.to_dict() == expected
    assert plan.iloc[1]["ReferenceValue"] == base.to_dict()[section][parameter]
    assert plan.iloc[1]["Value"] == value


def test_version_change_requires_an_active_gradient_fix_objective(prepared):
    base = replace(prepared.config, loss=replace(prepared.config.loss, regions="dice_ce"))
    variant = ablation.AblationVariant("v3", "gradient_fix_version", 3)

    with pytest.raises(ValueError, match="only changes a gradient-fix"):
        variant.configuration(base)


def test_legacy_loss_row_is_rejected_on_partial_data_before_any_run(prepared, tmp_path, monkeypatch):
    variants = [
        ablation.AblationVariant("reference"),
        ablation.AblationVariant("legacy", "regions", "dice_ce"),
    ]
    split = prepared.make_split()
    folder = tmp_path / "invalid_study"

    def forbidden(*args, **kwargs):
        pytest.fail("Every loss row must be checked before the first fit")

    monkeypatch.setattr(ablation, "run_training", forbidden)
    with pytest.raises(ValueError, match="legacy: partial annotations"):
        ablation.run_ablation(prepared, variants, study_dir=folder, split=split)
    assert not folder.exists()


@pytest.mark.parametrize("training", [True, False])
def test_unmasked_control_changes_targets_but_preserves_annotation_metadata(prepared, training):
    base = replace(
        prepared.config, model=replace(prepared.config.model, classifier=True),
        loss=replace(prepared.config.loss, gradient_fix_version=1),
    )
    variant = ablation.AblationVariant(
        "missing_as_background_no_classifier", "annotation_policy", "missing_as_background",
        disable_classifier=True,
    )
    changed = variant.configuration(base)
    expected = base.to_dict()
    expected["loss"]["annotation_policy"] = "missing_as_background"
    expected["model"]["classifier"] = False
    assert changed.to_dict() == expected

    data = replace(prepared, config=changed)
    ids = ["D1/SUBJ_0001"]
    keys = data.subjects.scan_keys(ids)
    before = [list(data.loader.get_subject(key).unavailable_labels) for key in keys]
    _, targets = data.make_generator(ids, training=training)[0]
    assert len(targets) == 2
    assert np.all(targets[0] >= 0) and np.all(targets[0].sum(axis=-1) == 1)
    assert np.all(targets[0][..., 2] == 0)
    assert [data.loader.get_subject(key).unavailable_labels for key in keys] == before
    assert before == [[2]]
    masked = replace(prepared, config=base)
    _, original_targets = masked.make_generator(ids, training=training)[0]
    assert len(original_targets) == 3
    assert np.all(original_targets[0][..., 2] == -1)
    assert base.loss.annotation_policy == "masked"


def test_unmasked_v1_supervises_previously_unavailable_logits():
    import tensorflow as tf
    from src.evaluation.losses import diceCELoss_gradient_fix

    objective = diceCELoss_gradient_fix(version=1)
    logits = tf.Variable([[[[[0.2, 0.4, 0.6]]]]], dtype=tf.float32)
    with tf.GradientTape() as tape:
        loss = objective(tf.constant([[[[[1., 0., 0.]]]]]), logits)
    unmasked_gradient = tape.gradient(loss, logits).numpy()
    with tf.GradientTape() as tape:
        loss = objective(tf.constant([[[[[1., 0., -1.]]]]]), logits)
    masked_gradient = tape.gradient(loss, logits).numpy()
    assert np.isfinite(unmasked_gradient).all()
    assert unmasked_gradient[..., 2].item() > 0
    assert masked_gradient[..., 2].item() == 0


@pytest.mark.parametrize("version", [2, 3])
def test_unmasked_background_control_rejects_foreground_only_objectives(prepared, version):
    loss = replace(prepared.config.loss, gradient_fix_version=version, annotation_policy="missing_as_background")
    with pytest.raises(ValueError, match="require dice_ce or gradient-fix v1"):
        loss.validate_for_training()


def test_unmasked_policy_is_explicit_and_old_configs_still_default_to_masking(prepared):
    from src.experiments.config import ExperimentConfig

    saved = prepared.config.to_dict()
    saved["loss"].pop("annotation_policy")
    assert ExperimentConfig.from_dict(saved).loss.annotation_policy == "masked"
    invalid = ablation.AblationVariant("ambiguous", "classifier", False, disable_classifier=True)
    with pytest.raises(ValueError, match="only for the explicit"):
        invalid.configuration(prepared.config)


def test_ordinary_loss_on_partial_annotations_requires_explicit_background_control(experiment):
    loss = replace(experiment.loss, regions="dice_ce", annotation_policy="missing_as_background")
    data = prepare_data(replace(experiment, loss=loss))
    _, targets = data.make_generator(["D1/SUBJ_0001"], training=False)[0]
    assert np.all(targets[0] >= 0)
    assert data.loader.get_subject(("D1_T1w", "SUBJ_0001")).unavailable_labels == [2]
    with pytest.raises(ValueError, match="partial annotations require"):
        prepare_data(replace(experiment, loss=replace(loss, annotation_policy="masked")))


def test_mouse_full_notebook_enables_the_explicit_control_by_default(prepared):
    notebook = json.loads((Path(__file__).resolve().parents[1] / "notebook/07f_ablation.ipynb").read_text(encoding="utf-8"))
    matrix = next(cell for cell in notebook["cells"] if cell.get("id") == "00629d66")
    base = replace(
        prepared.config, model=replace(prepared.config.model, classifier=True),
        loss=replace(prepared.config.loss, gradient_fix_version=1),
    )
    namespace = {"config": base, "USE_TBI_PRETRAINING": False, "MODE": "fixed_split", "TRAINING_SEEDS": (42,)}
    exec(compile(ast.parse("".join(matrix["source"])), "07f-matrix", "exec"), namespace)
    assert namespace["INCLUDE_MISSING_AS_BACKGROUND_CONTROL"] is True
    variants = namespace["variants"]
    assert len(variants) == 8
    control = next(variant for variant in variants if variant.name == "missing_as_background_no_classifier")
    config = control.configuration(base)
    assert not config.model.classifier and config.loss.annotation_policy == "missing_as_background"
    row = namespace["plan"].loc[namespace["plan"].Variant.eq(control.name)].iloc[0]
    assert row.Category == "combined_control" and not row.Classifier
    assert row.AnnotationPolicy == "missing_as_background"
