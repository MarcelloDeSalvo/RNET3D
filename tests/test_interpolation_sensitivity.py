"""Configurable MRI splines with shared discrete targets and matched study inputs."""

import ast
import json
from dataclasses import replace
from pathlib import Path

import nibabel as nib
import numpy as np
import plotly.graph_objects as go
import pytest
import SimpleITK as sitk
import tensorflow as tf

from test_experiments import experiment
from src.experiments.config import ExperimentConfig
from src.experiments.data import make_preprocessor, prepare_data
from src.preprocessing.preprocessor import RandomAffine, Resample


@pytest.mark.parametrize("stage", ["training", "validation", "inference"])
@pytest.mark.parametrize("order", [0, 1, 2, 3])
def test_resampling_order_routes_to_every_stage_without_changing_affine_augmentation(experiment, stage, order):
    preprocessing = replace(experiment.preprocessing, augment=True, resampling_order=order)
    config = replace(experiment, preprocessing=preprocessing)
    pipeline = make_preprocessor(config, stage=stage)
    resampling = next(step for step in pipeline.steps if isinstance(step, Resample))
    assert resampling.interpolation == order
    if stage == "training":
        affine = next(step for step in pipeline.steps if isinstance(step, RandomAffine))
        assert affine.interpolation == sitk.sitkLinear


@pytest.mark.parametrize("invalid", [True, -1, 6, 1.5, "1"])
def test_bad_order_is_rejected_before_fitting(experiment, invalid):
    preprocessing = replace(experiment.preprocessing, resampling_order=invalid)
    with pytest.raises(ValueError, match="resampling_order"):
        preprocessing.validate()


def test_older_configurations_keep_the_linear_default(experiment):
    saved = experiment.to_dict()
    saved["preprocessing"].pop("resampling_order")
    loaded = ExperimentConfig.from_dict(saved)
    assert loaded.preprocessing.resampling_order == 1
    assert loaded.to_dict() == experiment.to_dict()


def test_actual_splines_change_mri_but_keep_gt_values_and_physical_grid(experiment):
    values = np.random.default_rng(7).normal(size=(16, 16, 16)).astype(np.float32)
    labels = np.zeros(values.shape, dtype=np.int16)
    labels[3:8, 4:9, 5:10] = 1
    labels[10:12, 6:8, 4:6] = 2
    affine = np.diag([0.15, 0.12, 0.25, 1])
    source = nib.Nifti1Image(values, affine)
    mask = nib.Nifti1Image(labels, affine)
    outputs = []
    masks = []
    for order in (0, 1, 2, 3):
        preprocessing = replace(experiment.preprocessing, resampling_order=order, correct_units=False)
        config = replace(experiment, preprocessing=preprocessing)
        pipeline = make_preprocessor(config, stage="inference")
        image, targets = pipeline.preprocess(imgs=source, masks=[mask])
        outputs.append(image)
        masks.append(targets[0])

    for output, target in zip(outputs, masks):
        np.testing.assert_array_equal(output.affine, outputs[0].affine)
        np.testing.assert_array_equal(target.affine, output.affine)
        np.testing.assert_array_equal(target.get_fdata(), masks[0].get_fdata())
        assert set(np.unique(target.get_fdata())).issubset({0, 1, 2})
        assert output.shape == outputs[0].shape
    for first, second in zip(outputs, outputs[1:]):
        assert not np.allclose(first.get_fdata(), second.get_fdata())


@pytest.fixture
def pilot_experiment(experiment):
    """D1-style labels with both isotropic and genuinely resampled acquisitions."""
    for folder in (experiment.data.root / "D1").iterdir():
        metadata_path = folder / "metadata.json"
        metadata = json.loads(metadata_path.read_text())
        metadata["annotated_regions"] = ["Les", "cl-V", "il-V", "tV"]
        anisotropic = folder.name in ("SUBJ_0000", "SUBJ_0001")
        metadata["modality"] = "T1w"
        if anisotropic:
            metadata["modality"] = "T2w"
        metadata_path.write_text(json.dumps(metadata))
        for path in folder.rglob("*.nii.gz"):
            image = nib.load(path)
            values = image.get_fdata()
            affine = image.affine.copy()
            if anisotropic:
                affine[2, 2] = 0.2
            if path.name == "Labels.nii.gz":
                values[values == 2] = 3
                values[9:11, 3:5, 5:7] = 13
                values[7:9, 7:9, 7:9] = 21
            stored = values.astype(image.get_data_dtype())
            nib.save(nib.Nifti1Image(stored, affine, image.header), path)
    return experiment


def test_evaluation_patch_coordinates_are_shared_but_mri_is_recomputed(pilot_experiment):
    from src.experiments.config import LabelConfig

    configured = replace(pilot_experiment, labels=LabelConfig())
    data = prepare_data(configured)
    collection = data.patch_collection()
    affected = ["D1/SUBJ_0000", "D1/SUBJ_0001"]
    batches = []
    for order in (0, 1, 2, 3):
        preprocessing = replace(data.config.preprocessing, resampling_order=order)
        configured = replace(data.config, preprocessing=preprocessing)
        current = replace(data, config=configured)
        assert current.patch_collection() is collection
        generator = current.make_generator(affected, training=False)
        batches.append(generator[0])
    for inputs, targets in batches[1:]:
        np.testing.assert_array_equal(targets[0], batches[0][1][0])
        np.testing.assert_array_equal(targets[1], batches[0][1][1])
        assert not np.allclose(inputs, batches[0][0])


def notebook_context(experiment, folder, *, run=False, resume=False, monkeypatch):
    """Execute real cells with a tiny synthetic budget and no displayed UI."""
    notebook_path = Path(__file__).resolve().parents[1] / "notebook/check_interpolation_sensitiity.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    replacements = {
        "DATA_ROOT": f"Path({str(experiment.data.root)!r})", "STUDY_DIR": f"Path({str(folder)!r})",
        "RUN_STUDY": repr(run), "RESUME_STUDY": repr(resume),
        "EPOCHS": "1", "STEPS_PER_EPOCH": "None", "BATCH_SIZE": "2",
        "PATCH_SIZE": "(16, 16, 16)", "FILTERS": "(2, 4, 8, 16, 32)",
        "INFERENCE_STRIDE": "16", "BOOTSTRAP_SAMPLES": "100",
    }
    shown = []
    monkeypatch.setattr(go.Figure, "show", lambda figure, **kwargs: shown.append(kwargs))
    namespace = {"__name__": "__main__"}
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        tree = ast.parse("".join(cell["source"]))
        for node in tree.body:
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id in replacements:
                node.value = ast.parse(replacements[node.target.id], mode="eval").body
        ast.fix_missing_locations(tree)
        exec(compile(tree, str(notebook_path), "exec"), namespace)
    return namespace, shown


def test_notebook_default_never_trains_or_creates_results(pilot_experiment, tmp_path, monkeypatch):
    monkeypatch.setattr(tf.keras.Model, "fit", lambda *args, **kwargs: pytest.fail("Default notebook must not train"))
    folder = tmp_path / "disabled"
    namespace, shown = notebook_context(pilot_experiment, folder, monkeypatch=monkeypatch)
    assert namespace["study"] is None and not folder.exists()
    assert namespace["grid_audit"].ResamplingNeeded.any()
    assert not namespace["grid_audit"].ResamplingNeeded.all()
    assert namespace["plan"].MRIResamplingOrder.tolist() == [0, 1, 2, 3]
    assert shown == []


def test_notebook_trains_four_tiny_models_and_exports_comparable_results(pilot_experiment, tmp_path, monkeypatch):
    folder = tmp_path / "pilot"
    namespace, shown = notebook_context(pilot_experiment, folder, run=True, monkeypatch=monkeypatch)
    study = namespace["study"]
    assert len(study.runs) == 4
    assert set(study.metrics.Variant) == {"order_0", "order_1", "order_2", "order_3"}
    assert set(study.patch_metrics.DatasetRole) == {"validation", "test"}
    assert len(namespace["validation_ranking"]) == 4
    assert not namespace["test_tables"].comparisons.empty
    assert not namespace["sequence_summary"].empty
    assert len(shown) == 2 and all(item == {"renderer": "vscode"} for item in shown)
    contract = json.loads((folder / "study.json").read_text())
    assert [variant["order"] for variant in contract["variants"]] == [0, 1, 2, 3]
    shared = json.loads((folder / "evaluation_patches.json").read_text())
    weights = []
    for order in (0, 1, 2, 3):
        run = folder / "runs" / f"order_{order}" / "seed_42"
        saved = ExperimentConfig.load(run / "experiment.json")
        assert saved.preprocessing.resampling_order == order
        patches = json.loads((run / "evaluation_patches.json").read_text())
        assert patches == shared
        state = json.loads((run / "fold_0/training_state.json").read_text())
        assert state["completed_epochs"] == 1
        weights.append(run / "fold_0" / state["selected_checkpoint"])
    assert all(path.is_file() for path in weights)

    monkeypatch.setattr(tf.keras.Model, "fit", lambda *args, **kwargs: pytest.fail("Completed fits must be reused"))
    resumed, _ = notebook_context(pilot_experiment, folder, run=True, resume=True, monkeypatch=monkeypatch)
    assert len(resumed["study"].runs) == 4
