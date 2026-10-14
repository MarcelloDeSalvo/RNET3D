"""Exercise animal splits, annotation policies, head contracts and a tiny full run."""

import json
from dataclasses import replace
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

from src.experiments.config import (
    DataConfig, DatasetSpec, EvaluationPatchConfig, ExperimentConfig, InferenceConfig, LabelConfig,
    LossConfig, ModelConfig, PreprocessingConfig,
)
from src.experiments.data import SubjectSplit, annotation_values, make_split, prepare_data
from src.evaluation.k_fold import TrainingConfig


@pytest.fixture
def experiment(tmp_path):
    """Four animals, one longitudinal, with explicit complete/partial annotation."""
    root = tmp_path / "dataset"
    affine = np.diag([0.1, 0.1, 0.1, 1.0])
    rng = np.random.default_rng(9)
    for number in range(4):
        subject = f"SUBJ_{number:04d}"
        folder = root / "D1" / subject
        folder.mkdir(parents=True)
        timepoints = []
        if number == 3:
            timepoints = ["7days", "14days"]
        names = ["Les"]
        if number % 2 == 0:
            names.append("cl-CC")
        metadata = {
            "dataset": "D1", "normalized_id": subject, "original_id": f"animal-{number}",
            "condition": "TBI", "modality": "T1w", "timepoints": timepoints,
            "annotated_regions": names, "has_lesion_annotation": True,
        }
        (folder / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
        for timepoint in timepoints or [None]:
            scan_folder = folder
            if timepoint:
                scan_folder = folder / timepoint
                scan_folder.mkdir()
            labels = np.zeros((16, 16, 16), dtype=np.int16)
            labels[4:8, 4:8, 4:8] = 1
            if "cl-CC" in names:
                labels[8:10, 8:10, 8:10] = 2
            arrays = {
                "N4": rng.normal(size=labels.shape).astype(np.float32),
                "Labels": labels,
                "brain_mask": np.ones(labels.shape, dtype=np.int16),
            }
            for name, data in arrays.items():
                image = nib.Nifti1Image(data, affine)
                image.header.set_xyzt_units("mm")
                nib.save(image, scan_folder / f"{name}.nii.gz")

    training = TrainingConfig(
        model_name="tiny", epochs=1, batch_size=2, output_dir=str(tmp_path / "runs"),
        default_callbacks=False, fit_kwargs={"verbose": 0, "workers": 0},
    )
    reference = root / "D1" / "SUBJ_0000" / "N4.nii.gz"
    config = ExperimentConfig(
        data=DataConfig(root=root),
        preprocessing=PreprocessingConfig(reference=reference, patch_size=(16, 16, 16), augment=False),
        training=training,
        labels=LabelConfig(mapping={0: 0, 1: 1, 2: 2}, names=("Background", "Lesion", "CC")),
        model=ModelConfig(
            architecture="unet", filters=(2, 4, 8, 16, 32), residual=False, attention=False,
        ),
        loss=LossConfig(regions="dice_ce_gradient_fix"),
        inference=InferenceConfig(enabled=True, stride=16),
        evaluation_patches=EvaluationPatchConfig(patches_per_scan=1, sampling="center"),
    )
    return config


def test_split_groups_timepoints_and_roundtrips(experiment, tmp_path):
    data = prepare_data(experiment)
    split = make_split(data.subjects, seed=21)
    same = make_split(data.subjects, seed=21)
    assert split == same
    assert len(data.subjects.subject_ids) == 4
    assert len(data.subjects.scan_keys(data.subjects.subject_ids)) == 5
    path = tmp_path / "split.json"
    split.save(path)
    assert SubjectSplit.load(path, data.subjects) == split
    summary = data.split_summary(split)
    assert summary["Animals"].sum() == 4
    assert summary["Scans"].sum() == 5

    leaked = SubjectSplit(train=split.train, validation=split.validation, test=split.train)
    with pytest.raises(ValueError, match="overlapping"):
        leaked.validate(data.subjects)


def test_partial_annotations_fail_with_probability_loss(experiment):
    config = replace(experiment, loss=LossConfig(regions="dice_ce"))
    with pytest.raises(ValueError, match="partial annotations"):
        prepare_data(config)


@pytest.mark.parametrize("stage", ["training", "validation", "inference"])
def test_extra_anatomy_maps_to_task_background_without_changing_selected_labels(experiment, stage):
    """Ignore the reported extra classes in every stage, preserving mask semantics."""
    from src.experiments.data import make_preprocessor

    assert experiment.labels.unknown == "background"
    data = prepare_data(experiment)
    scan = data.loader.get_subject(("D1_T1w", "SUBJ_0000"))
    source = scan.labels.get_fdata().astype(np.int16)
    extra_labels = (4, 7, 9, 14, 17)
    for position, label in enumerate(extra_labels):
        source[position, 0, 0] = label
    labels = nib.Nifti1Image(source, scan.labels.affine, scan.labels.header)
    pipeline = make_preprocessor(experiment, stage=stage)
    _, masks = pipeline.preprocess(scan.image, [labels, scan.brain_mask])
    actual = masks[0].get_fdata()
    expected = source.copy()
    expected[np.isin(expected, extra_labels)] = 0
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(masks[1].get_fdata(), scan.brain_mask.get_fdata())
    assert set(np.unique(actual)) == {0, 1, 2}


def test_strict_label_policy_remains_available(experiment):
    """An explicit strict policy still catches undeclared source-mask values."""
    from src.experiments.data import make_preprocessor

    config = replace(experiment, labels=replace(experiment.labels, unknown="error"))
    data = prepare_data(config)
    scan = data.loader.get_subject(("D1_T1w", "SUBJ_0000"))
    source = np.full(scan.labels.shape, 4, dtype=np.int16)
    labels = nib.Nifti1Image(source, scan.labels.affine, scan.labels.header)
    pipeline = make_preprocessor(config, stage="validation")
    with pytest.raises(ValueError, match="Unmapped mask labels"):
        pipeline.preprocess(scan.image, [labels, scan.brain_mask])


def test_vocabulary_is_not_annotation_availability(experiment):
    data = prepare_data(experiment)
    scan = data.loader.get_subject(("D1_T1w", "SUBJ_0001"))
    scan.metadata["label_mapping"] = [{"mask_value": 1}, {"mask_value": 2}]
    scan.metadata["has_lesion_annotation"] = False
    assert annotation_values(scan) == {0}
    scan.metadata["annotated_regions"] = ["Unknown"]
    with pytest.raises(ValueError, match="unknown annotation"):
        annotation_values(scan)


def test_validation_is_repeatable_keeps_tail_and_masks_partial_targets(experiment):
    data = prepare_data(experiment)
    ids = ["D1/SUBJ_0000", "D1/SUBJ_0001", "D1/SUBJ_0002"]
    generator = data.make_generator(ids, training=False)
    assert len(generator) == 2
    first_inputs, first_targets = generator[0]
    second_inputs, second_targets = generator[0]
    np.testing.assert_array_equal(first_inputs, second_inputs)
    for first, second in zip(first_targets, second_targets):
        np.testing.assert_array_equal(first, second)
    assert len(first_targets) == 2  # Partial annotation does not require a classifier.
    assert np.all(first_targets[0][1, ..., 2] == -1)
    assert generator[1][0].shape[0] == 1


@pytest.mark.parametrize(
    ("random_crop", "crop_mode", "expected_mode"),
    [(True, "center", "center"), (True, "random", "random"), (False, "center", None)],
)
def test_training_crop_settings(experiment, random_crop, crop_mode, expected_mode):
    """Use centre-biased sampling by default while retaining explicit alternatives."""
    from src.experiments.data import make_preprocessor
    from src.preprocessing.preprocessor import PadderCutter, RandomCropping

    assert experiment.preprocessing.random_crop is True
    assert experiment.preprocessing.crop_mode == "center"

    settings = replace(experiment.preprocessing, random_crop=random_crop, crop_mode=crop_mode)
    config = replace(experiment, preprocessing=settings)
    pipeline = make_preprocessor(config, stage="training")
    crops = [step for step in pipeline.steps if isinstance(step, (RandomCropping, PadderCutter))]
    assert len(crops) == 1
    crop = crops[0]

    if expected_mode is None:
        assert isinstance(crop, PadderCutter)
    else:
        assert isinstance(crop, RandomCropping)
        assert crop.mode == expected_mode
        assert crop.std is None
        assert crop.dim == settings.patch_size
        assert crop.pad_if_needed is True


@pytest.mark.parametrize("stage", ["validation", "inference"])
def test_random_training_crop_does_not_affect_evaluation(experiment, stage):
    """Validation keeps fixed crops and inference keeps the complete volume."""
    from src.experiments.data import make_preprocessor
    from src.preprocessing.preprocessor import PadderCutter, RandomCropping

    pipeline = make_preprocessor(experiment, stage=stage)
    assert not any(isinstance(step, RandomCropping) for step in pipeline.steps)
    fixed_crops = [step for step in pipeline.steps if isinstance(step, PadderCutter)]
    if stage == "validation":
        assert len(fixed_crops) == 1
    else:
        assert not fixed_crops


def test_invalid_crop_mode_is_rejected(experiment):
    settings = replace(experiment.preprocessing, crop_mode="invalid")
    config = replace(experiment, preprocessing=settings)
    with pytest.raises(ValueError, match="crop mode"):
        config.validate()


@pytest.mark.parametrize("stage", ["training", "validation", "inference"])
def test_experiment_handles_shear_before_affine_augmentation(experiment, stage):
    from src.experiments.data import make_preprocessor
    from src.preprocessing import preprocessor as preprocessing

    settings = replace(
        experiment.preprocessing, augment=True, augmentation_probability=1.0,
        flip_probability=0.0, rotation_range=(0.0, 0.0), scale_range=(1.0, 1.0),
    )
    config = replace(experiment, preprocessing=settings)
    data = prepare_data(config)
    scan = data.loader.get_subject(("D1_T1w", "SUBJ_0000"))
    affine = scan.image.affine.copy()
    affine[0, 1] = 0.000015
    image = nib.Nifti1Image(scan.image.get_fdata(), affine)
    labels = nib.Nifti1Image(scan.labels.get_fdata(), affine)
    brain = nib.Nifti1Image(scan.brain_mask.get_fdata(), affine)
    pipeline = make_preprocessor(config, stage=stage)
    result, masks = pipeline.preprocess(image, [labels, brain])
    preprocessing._lps_geometry(result)
    for mask in masks:
        assert mask.shape == result.shape
        np.testing.assert_array_equal(mask.affine, result.affine)
        assert set(np.unique(mask.get_fdata())) <= {0, 1, 2}
    if stage == "inference":
        restored = pipeline.deprocess(masks[0], image, {0: 0, 1: 1, 2: 2})
        assert restored.shape == image.shape
        np.testing.assert_array_equal(restored.affine, image.affine)


def test_training_repetition_does_not_duplicate_validation(experiment):
    settings = replace(experiment.training, data_multiplier=3)
    experiment = replace(experiment, training=settings)
    data = prepare_data(experiment)
    ids = ["D1/SUBJ_0000", "D1/SUBJ_0001"]
    assert len(data.make_generator(ids, training=True).ids) == 6
    assert len(data.make_generator(ids, training=False).ids) == 2


def test_merged_class_is_unavailable_if_only_one_side_is_annotated(experiment):
    from src.preprocessing.generator import MultiTaskGenerator

    data = prepare_data(experiment)
    key = ("D1_T1w", "SUBJ_0001")
    scan = data.loader.get_subject(key)
    scan.unavailable_labels = [13]
    settings = {
        "dataset_path": str(experiment.data.root), "labels": [0, 1, 3, 13],
        "label_mapping": {0: 0, 1: 1, 3: 2, 13: 2}, "input_shape": (16, 16, 16),
        "in_channels": 1, "num_classes": 3, "batch_size": 1,
    }
    generator = MultiTaskGenerator([key], data.loader, settings, shuffle=False, partial_labels=True)
    targets = generator[0][1]
    assert np.all(targets[0][..., 2] == -1)


@pytest.mark.parametrize("roi_loss,version", [("dice_ce", 2), ("dice_ce_gradient_fix", 1), ("dice_ce_gradient_fix", 2)])
def test_tiny_training_exports_probabilities_and_held_out_metrics(experiment, roi_loss, version):
    import tensorflow as tf
    from src.experiments.training import run_training

    experiment = replace(experiment, loss=replace(experiment.loss, gradient_fix_version=version))
    if roi_loss == "dice_ce":
        # Synthetic absent CC structures are declared annotated for this case.
        dataset = DatasetSpec(available_labels=(0, 1, 2))
        data_settings = replace(experiment.data, datasets=(dataset,))
        experiment = replace(experiment, data=data_settings, loss=LossConfig(regions=roi_loss))
    data = prepare_data(experiment)
    split = SubjectSplit(
        train=("D1/SUBJ_0000", "D1/SUBJ_0001"),
        validation=("D1/SUBJ_0002",), test=("D1/SUBJ_0003",),
    )
    result = run_training(data, split)
    cached = run_training(data, split, resume_run=result.run_dir)
    assert cached.histories == result.histories
    assert set(cached.evaluations["regions"][0]["Scan"]) == {"SUBJ_0003_7days", "SUBJ_0003_14days"}
    assert len(result.histories[0]["loss"]) == 1
    if roi_loss == "dice_ce_gradient_fix":
        for key in ("regions_logits_mean_dice", "val_regions_logits_mean_dice"):
            scores = result.histories[0][key]
            assert len(scores) == 1
            assert 0 <= scores[0] <= 1
    assert (result.run_dir / "experiment.json").is_file()
    assert (result.run_dir / "subjects.csv").is_file()
    assert (result.run_dir / "fold_0" / "best.weights.h5").is_file()
    model = tf.keras.models.load_model(result.run_dir / "fold_0" / "model.keras", compile=False)
    assert model.output_names == ["regions", "brain_mask"]
    inputs = np.zeros((1, 16, 16, 16, 1), dtype=np.float32)
    probabilities = model(inputs, training=False)[0].numpy()
    np.testing.assert_allclose(probabilities.sum(axis=-1), 1, atol=1e-6)
    table = result.evaluations["regions"][0]
    assert set(table["Subject"]) == set(split.test)
    if roi_loss == "dice_ce_gradient_fix":
        assert set(table.loc[table["Region"] == "CC", "Status"]) == {"unavailable"}
    else:
        assert "unavailable" not in set(table["Status"])
    lesion_dice = table.loc[(table["Region"] == "Lesion") & (table["Metric"] == "Dice"), "Value"]
    assert np.isfinite(lesion_dice).all()
    assert set(table["Scan"]) == {"SUBJ_0003_7days", "SUBJ_0003_14days"}
    outputs = result.run_dir / "fold_0" / "evaluations" / "test" / "regions"
    timings = pd.read_csv(outputs / "inference_timings.csv")
    assert len(timings) == 2
    for timepoint in ("7days", "14days"):
        prediction_path = outputs / "predictions" / "D1_T1w" / f"SUBJ_0003_{timepoint}" / "regions.nii.gz"
        prediction = nib.load(prediction_path)
        source = data.loader.get_subject(("D1_T1w", f"SUBJ_0003_{timepoint}"))
        assert prediction.shape == source.image.shape
        np.testing.assert_allclose(prediction.affine, source.image.affine)


def test_legacy_registration_requires_annotation_override(experiment):
    legacy = DatasetSpec(name="legacy", layout="legacy", modality="FLASH")
    config = replace(experiment, data=DataConfig(root=experiment.data.root, datasets=(legacy,)))
    with pytest.raises(ValueError, match="explicit available_labels"):
        config.validate()


def test_legacy_source_runs_through_the_same_core(experiment):
    import shutil

    folder = experiment.data.root / "FLASH" / "mouse" / "Anat"
    folder.mkdir(parents=True)
    original = experiment.data.root / "D1" / "SUBJ_0000"
    for name in ("N4", "Labels", "brain_mask"):
        shutil.copyfile(original / f"{name}.nii.gz", folder / f"mouse_{name}.nii.gz")
    legacy = DatasetSpec(
        name="FLASH", layout="legacy", modality="FLASH", sub_folder="Anat",
        available_labels=(0, 1, 2),
    )
    settings = replace(experiment.data, datasets=(legacy,))
    config = replace(experiment, data=settings, loss=LossConfig(regions="dice_ce"))
    data = prepare_data(config)
    assert data.subjects.subject_ids == ["FLASH/mouse"]
    inputs, targets = data.make_generator(["FLASH/mouse"], training=False)[0]
    assert inputs.shape == (1, 16, 16, 16, 1)
    assert len(targets) == 2


def test_cv_splits_stay_fixed_when_training_seed_changes(experiment, monkeypatch):
    from src.evaluation import k_fold
    from src.experiments.training import run_cross_validation

    observed_seeds = []

    def inspect_fold(**kwargs):
        context = kwargs["context"]
        run_dir = context.output_dir.parent
        assert (run_dir / "experiment.json").is_file()  # Provenance precedes fitting.
        observed_seeds.append(context.config.seed)

    monkeypatch.setattr(k_fold, "train_fold", inspect_fold)
    first_settings = replace(experiment.training, k=2, seed=11)
    first_config = replace(experiment, training=first_settings, split_seed=19)
    second_settings = replace(first_settings, seed=47)
    second_config = replace(first_config, training=second_settings)
    first = run_cross_validation(prepare_data(first_config))
    second = run_cross_validation(prepare_data(second_config))
    first_splits = json.loads((first.run_dir / "splits.json").read_text())
    second_splits = json.loads((second.run_dir / "splits.json").read_text())
    assert first_splits == second_splits
    assert observed_seeds == [11, 11, 47, 47]
    manifest = pd.read_csv(first.run_dir / "subjects.csv")
    assert set(manifest["DatasetRole"]) == {"cv"}
    assert len(manifest) == 5


def test_rnet_export_reloads_with_probability_and_classifier_heads(experiment, tmp_path):
    import tensorflow as tf
    from src.experiments.training import build_model, export_model

    settings = ModelConfig(architecture="3drnet", filters=(2, 4, 8, 16, 32), classifier=True)
    config = replace(experiment, model=settings)
    model = build_model(config)
    assert model.output_names == ["regions_logits", "brain_mask", "classifier"]
    path = tmp_path / "rnet.keras"
    export_model(model, path)
    restored = tf.keras.models.load_model(path, compile=False)
    assert restored.output_names == ["regions", "brain_mask", "classifier"]
    inputs = np.zeros((1, 16, 16, 16, 1), dtype=np.float32)
    probabilities = restored(inputs, training=False)[0].numpy()
    np.testing.assert_allclose(probabilities.sum(axis=-1), 1, atol=1e-6)


@pytest.mark.parametrize("filename", [
    "01_tbi_mouse_training.ipynb", "02_tbi_mouse_kfold.ipynb",
    "03_rat_finetuning.ipynb", "04_rat_kfold.ipynb", "05_mouse_full_training.ipynb",
    "06_mouse_full_kfold.ipynb",
    "07_ablation.ipynb",
    "07f_ablation.ipynb",
    "08_unet_comparison.ipynb",
    "09_inference_evaluation.ipynb",
    "10_revision_results.ipynb",
])
def test_notebook_has_valid_python_cells(filename):
    """Validate syntax without forbidding a researcher's saved execution outputs."""
    import ast
    import nbformat

    path = Path(__file__).resolve().parents[1] / "notebook" / filename
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            ast.parse(cell.source)


def test_fixed_split_resume_restores_nadam_and_validates_configuration(experiment, monkeypatch):
    import tensorflow as tf
    from src.experiments import training

    settings = replace(experiment.training, epochs=2, run_id="interrupted")
    config = replace(experiment, training=settings, inference=InferenceConfig(enabled=False, stride=16))
    data = prepare_data(config)
    split = SubjectSplit(
        train=("D1/SUBJ_0000", "D1/SUBJ_0001"),
        validation=("D1/SUBJ_0002",), test=("D1/SUBJ_0003",),
    )
    control = {"interrupt": True}
    iterations = []

    class Interrupt(tf.keras.callbacks.Callback):
        def on_train_begin(self, logs=None):
            iterations.append(int(self.model.optimizer.iterations.numpy()))

        def on_epoch_begin(self, epoch, logs=None):
            if epoch == 1 and control["interrupt"]:
                raise RuntimeError("Paused for recovery test")

    original_callbacks = training.callbacks

    def callbacks(configuration, context):
        return original_callbacks(configuration, context) + [Interrupt()]

    monkeypatch.setattr(training, "callbacks", callbacks)
    with pytest.raises(RuntimeError, match="Paused"):
        training.run_training(data, split)
    folder = Path(settings.output_dir) / settings.model_name / settings.run_id
    control["interrupt"] = False
    resumed = training.run_training(data, split, resume_run=folder)
    assert len(resumed.histories[0]["loss"]) == 2
    assert iterations == [0, 1]
    changed = replace(config, learning_rate=0.002)
    with pytest.raises(ValueError, match="configuration differs"):
        training.run_training(prepare_data(changed), split, resume_run=folder)


@pytest.mark.parametrize("architecture", ["unet", "3drnet"])
def test_backbone_transfer_freezes_features_and_preserves_new_head_contract(experiment, tmp_path, architecture):
    import tensorflow as tf
    from src.experiments.config import InitializationConfig
    from src.experiments.initialization import backbone
    from src.experiments.training import build_model, export_model

    settings = replace(experiment.model, architecture=architecture)
    experiment = replace(experiment, model=settings)
    source = build_model(experiment)
    source_features = backbone(source)
    path = tmp_path / "source.keras"
    export_model(source, path)
    labels = LabelConfig(mapping={0: 0, 1: 1, 2: 2, 3: 3}, names=("Background", "Lesion", "CC", "Ventricle"))
    initialization = InitializationConfig(mode="backbone", checkpoint=path, freeze_backbone=True)
    config = replace(experiment, labels=labels, initialization=initialization)
    target = build_model(config)
    target_features = backbone(target)
    assert target.output_shape[0][-1] == 4
    assert target.trainable_weights  # The new heads remain trainable.
    assert all(not layer.trainable for layer in target_features.layers)
    for source_weights, target_weights in zip(source_features.get_weights(), target_features.get_weights()):
        np.testing.assert_array_equal(source_weights, target_weights)
    assert target.initialization_report["reinitialized_heads"]

    different_model = replace(config.model, residual=True)
    with pytest.raises(ValueError, match="backbone graphs differ"):
        build_model(replace(config, model=different_model))


@pytest.mark.parametrize("checkpoint_kind", ["weights", "inference"])
def test_complete_pretrained_initialization_retains_all_heads(experiment, tmp_path, checkpoint_kind):
    """Rat fine-tuning copies complete mouse weights, including both output heads."""
    from src.experiments.config import InitializationConfig
    from src.experiments.training import build_model, export_model

    source = build_model(experiment)
    if checkpoint_kind == "weights":
        checkpoint = tmp_path / "mouse.weights.h5"
        source.save_weights(str(checkpoint))
    else:
        checkpoint = tmp_path / "mouse.keras"
        export_model(source, checkpoint)

    initialization = InitializationConfig(mode="weights", checkpoint=checkpoint)
    target_config = replace(experiment, initialization=initialization, learning_rate=0.0001)
    target = build_model(target_config)
    for expected, actual in zip(source.get_weights(), target.get_weights()):
        np.testing.assert_array_equal(actual, expected)
    assert len(source.get_weights()) == len(target.get_weights())
    assert all(layer.trainable for layer in target.layers)
    assert int(target.optimizer.iterations.numpy()) == 0
    assert "reinitialized_heads" not in target.initialization_report


def test_outer_cv_folds_keep_longitudinal_animals_out_of_selection(experiment, tmp_path):
    from src.experiments.data import fold_summary, make_cv_splits
    from src.evaluation.k_fold import load_fold_splits

    data = prepare_data(experiment)
    folds = make_cv_splits(data.subjects.subject_ids, k=2, seed=21)
    held_out = []
    for fold in folds:
        assert not set(fold.test_ids) & set(fold.train_ids + fold.val_ids)
        held_out.extend(fold.test_ids)
    assert sorted(held_out) == data.subjects.subject_ids
    summary = fold_summary(folds, data.subjects)
    assert summary.loc[summary["Role"] == "holdout", "Scans"].sum() == 5
    path = tmp_path / "outer.json"
    path.write_text(json.dumps([fold.to_dict() for fold in folds]))
    assert load_fold_splits(path, data.subjects.subject_ids, 2) == folds


def test_leave_one_animal_out_evaluates_all_timepoints_once(experiment, tmp_path):
    """Match the old rat split without leaking a longitudinal scan into training."""
    from src.experiments.data import make_leave_one_animal_out_splits
    from src.evaluation.k_fold import load_fold_splits

    data = prepare_data(experiment)
    animals = data.subjects.subject_ids
    folds = make_leave_one_animal_out_splits(animals)
    assert len(folds) == len(animals)
    assert folds == make_leave_one_animal_out_splits(list(reversed(animals)))
    evaluated_scans = []
    for fold in folds:
        assert len(fold.val_ids) == 1
        assert len(fold.train_ids) == len(animals) - 1
        assert not fold.test_ids
        train_scans = data.subjects.scan_keys(fold.train_ids)
        val_scans = data.subjects.scan_keys(fold.val_ids)
        assert not set(train_scans) & set(val_scans)
        evaluated_scans.extend(val_scans)
        generator = data.make_generator(fold.val_ids, training=False, batch_size=1)
        assert len(generator) == len(val_scans)

    assert sorted(evaluated_scans) == sorted(data.subjects.scan_keys(animals))
    path = tmp_path / "rat_folds.json"
    path.write_text(json.dumps([fold.to_dict() for fold in folds]))
    assert load_fold_splits(path, animals, len(animals)) == folds

    with pytest.raises(ValueError, match="unique"):
        make_leave_one_animal_out_splits(["rat", "rat"])
    with pytest.raises(ValueError, match="at least two"):
        make_leave_one_animal_out_splits(["rat"])


def test_identity_override_joins_confirmed_animals(experiment):
    identities = {"D1/SUBJ_0000": "confirmed-animal", "D1/SUBJ_0003": "confirmed-animal"}
    settings = replace(experiment.data, animal_ids=identities)
    data = prepare_data(replace(experiment, data=settings))
    assert len(data.subjects.subject_ids) == 3
    assert len(data.subjects.scan_keys(["confirmed-animal"])) == 3


def test_annotation_summary_counts_declarations_instead_of_voxels(experiment):
    data = prepare_data(experiment)
    coverage = data.annotation_summary().set_index("Class")
    assert coverage.loc["CC", "AnnotatedScans"] == 2
    assert coverage.loc["CC", "UnannotatedScans"] == 3
    assert coverage.loc["Lesion", "AnnotatedScans"] == 5
    assert coverage["TotalScans"].eq(5).all()


@pytest.mark.parametrize("version", [1, 2])
def test_tiny_mouse_full_run_combines_partial_cohorts_and_classifier(experiment, version):
    """Fit/export the three-head partial-label workflow on tiny mixed mouse data."""
    import shutil
    import tensorflow as tf
    from src.experiments.training import run_training

    source_folder = experiment.data.root / "D1" / "SUBJ_0000"
    sham_folder = experiment.data.root / "D3" / "SUBJ_0004"
    sham_folder.mkdir(parents=True)
    for name in ("N4", "Labels", "brain_mask"):
        shutil.copyfile(source_folder / f"{name}.nii.gz", sham_folder / f"{name}.nii.gz")
    mask = nib.load(sham_folder / "Labels.nii.gz")
    labels_data = mask.get_fdata().astype(np.int16)
    labels_data[labels_data == 1] = 0
    nib.save(nib.Nifti1Image(labels_data, mask.affine), sham_folder / "Labels.nii.gz")
    metadata = {
        "dataset": "D3", "normalized_id": "SUBJ_0004", "original_id": "sham-4",
        "condition": "SHAM", "modality": "T1w", "annotated_regions": ["cl-CC"],
        "has_lesion_annotation": False, "timepoints": [],
    }
    (sham_folder / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    mapping = {0: 0, 1: 1, 2: 2, 3: 3, 5: 4, 6: 5, 12: 6, 13: 7, 15: 8, 16: 9, 21: 10}
    names = ("Background", "Lesion", "cl-CC", "cl-V", "cl-Hipp", "cl-Ctx",
             "il-CC", "il-V", "il-Hipp", "il-Ctx", "tV")
    settings = replace(experiment.data, datasets=(DatasetSpec(name="D1"), DatasetSpec(name="D3")))
    model = ModelConfig(architecture="3drnet", filters=(2, 4, 8, 16, 32), classifier=True)
    loss = replace(experiment.loss, gradient_fix_version=version)
    config = replace(experiment, data=settings, labels=LabelConfig(mapping=mapping, names=names), model=model, loss=loss)
    data = prepare_data(config)
    coverage = data.annotation_summary()
    lesion = coverage.loc[(coverage["Dataset"] == "D3") & (coverage["Class"] == "Lesion")]
    assert lesion["UnannotatedScans"].item() == 1
    _, targets = data.make_generator(["D3/SUBJ_0004"], training=False)[0]
    assert np.all(targets[0][..., 1] == -1)
    assert targets[2].item() == 0

    split = SubjectSplit(
        train=("D1/SUBJ_0000", "D3/SUBJ_0004"), validation=("D1/SUBJ_0002",),
        test=("D1/SUBJ_0001", "D1/SUBJ_0003"),
    )
    result = run_training(data, split)
    assert np.isfinite(result.histories[0]["loss"]).all()
    for key in ("regions_logits_mean_dice", "val_regions_logits_mean_dice"):
        scores = result.histories[0][key]
        assert len(scores) == 1
        assert 0 <= scores[0] <= 1
    restored = tf.keras.models.load_model(result.run_dir / "fold_0" / "model.keras", compile=False)
    assert restored.output_names == ["regions", "brain_mask", "classifier"]
    assert restored.output_shape[0][-1] == 11
    table = result.evaluations["regions"][0]
    assert set(table["Scan"]) == {"SUBJ_0001", "SUBJ_0003_7days", "SUBJ_0003_14days"}
    missing = table.loc[table["Region"] == "il-Ctx"]
    assert missing["Status"].eq("unavailable").all()
    assert missing["Value"].isna().all()
