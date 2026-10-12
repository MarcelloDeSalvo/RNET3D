"""Exercise animal splits, annotation policies, head contracts and a tiny full run."""

import json
from dataclasses import replace
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

from src.experiments.config import (
    DataConfig, DatasetSpec, ExperimentConfig, InferenceConfig, LabelConfig,
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
        model=ModelConfig(architecture="unet", filters=(2, 4, 8, 16, 32)),
        loss=LossConfig(regions="dice_ce_gradient_fix"),
        inference=InferenceConfig(enabled=True, stride=16),
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


@pytest.mark.parametrize("roi_loss", ["dice_ce", "dice_ce_gradient_fix"])
def test_tiny_training_exports_probabilities_and_held_out_metrics(experiment, roi_loss):
    import tensorflow as tf
    from src.experiments.training import run_training

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
    assert len(result.histories[0]["loss"]) == 1
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


def test_notebook_has_valid_cells_and_no_saved_outputs():
    import ast
    import nbformat

    path = Path(__file__).resolve().parents[1] / "notebook" / "01_training.ipynb"
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            ast.parse(cell.source)
            assert not cell.outputs
