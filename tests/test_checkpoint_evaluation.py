"""Evaluate stopped checkpoints without fitting or modifying recovery artifacts."""

from dataclasses import replace
import ast
import hashlib
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import pandas as pd
import pytest
import tensorflow as tf

from test_experiments import experiment
from src.experiments.config import InitializationConfig
from src.experiments.training import build_model, inference_model
from src.evaluation import inference


def test_version_one_partial_label_weights_remain_evaluable_without_compilation(experiment, tmp_path, monkeypatch):
    """The loss revision changes optimization, not the stored network weights."""
    model = build_model(experiment)
    run = tmp_path / "old_run"
    fold = write_run(experiment, run)
    model.save_weights(str(fold / "best.weights.h5"))
    path = run / "experiment.json"
    settings = json.loads(path.read_text(encoding="utf-8"))
    settings["loss"].pop("gradient_fix_version")
    path.write_text(json.dumps(settings), encoding="utf-8")

    def forbidden(*args, **kwargs):
        pytest.fail("Old-checkpoint evaluation must not compile or fit")

    monkeypatch.setattr(tf.keras.Model, "compile", forbidden)
    monkeypatch.setattr(tf.keras.Model, "fit", forbidden)
    selection = inference.best_checkpoint_settings(run)
    result = inference.evaluate_saved_model(selection)
    assert result.data.config.loss.gradient_fix_version == 1
    assert set(result.metrics["Scan"]) == {"SUBJ_0003_7days", "SUBJ_0003_14days"}


def write_run(config, run):
    """Persist a fixed-split run contract without marking fitting complete."""
    fold = run / "fold_0"
    fold.mkdir(parents=True)
    (run / "experiment.json").write_text(json.dumps(config.to_dict()), encoding="utf-8")
    split = {
        "train": ["D1/SUBJ_0000", "D1/SUBJ_0001"],
        "validation": ["D1/SUBJ_0002"], "test": ["D1/SUBJ_0003"],
    }
    (run / "splits.json").write_text(json.dumps(split), encoding="utf-8")
    (fold / "split.json").write_text(json.dumps(split), encoding="utf-8")
    return fold


def artifact_hashes(run):
    """Include every source-run file, not just its weights and JSON state."""
    return {
        str(path.relative_to(run)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in run.rglob("*") if path.is_file()
    }


@pytest.mark.parametrize("architecture,classifier", [("unet", False), ("3drnet", True)])
@pytest.mark.parametrize("roi_loss", ["dice_ce", "dice_ce_gradient_fix"])
def test_stopped_checkpoint_matches_all_heads_and_preserves_training_files(
    experiment, tmp_path, monkeypatch, architecture, classifier, roi_loss,
):
    settings = replace(experiment.model, architecture=architecture, classifier=classifier)
    config = replace(experiment, model=settings, loss=replace(experiment.loss, regions=roi_loss))
    source = build_model(config)
    inputs = np.ones((1, 16, 16, 16, 1), dtype=np.float32)
    expected = inference_model(source)(inputs, training=False)
    run = tmp_path / "run"
    fold = write_run(config, run)
    source.save_weights(str(fold / "best.weights.h5"))
    selected = fold / "latest" / "selected_1.weights.h5"
    selected.parent.mkdir()
    shutil.copyfile(fold / "best.weights.h5", selected)
    state = {"fit_complete": False, "completed_epochs": 1, "selected_checkpoint": "latest/selected_1.weights.h5"}
    (fold / "training_state.json").write_text(json.dumps(state), encoding="utf-8")
    # A stopped save may leave the public best file uncommitted/corrupted.
    # Use the snapshot paired with the committed recovery record instead.
    (fold / "best.weights.h5").write_bytes(b"uncommitted overwrite")
    (fold / "training.csv").write_text("epoch,loss\n0,0.5\n", encoding="utf-8")

    # Rat transfer sources may have moved; offline loading needs only fitting weights.
    initialization = InitializationConfig(mode="weights", checkpoint=tmp_path / "unavailable_mouse.weights.h5")
    offline_config = replace(config, initialization=initialization)
    (run / "experiment.json").write_text(json.dumps(offline_config.to_dict()), encoding="utf-8")
    before = artifact_hashes(run)

    def forbidden(*args, **kwargs):
        pytest.fail("Checkpoint evaluation must not compile or fit")

    monkeypatch.setattr(tf.keras.Model, "compile", forbidden)
    monkeypatch.setattr(tf.keras.Model, "fit", forbidden)
    evaluate = inference.ExperimentVolumeEvaluator.evaluate

    def check_model(self):
        assert getattr(self.model, "optimizer", None) is None
        actual = self.model(inputs, training=False)
        for expected_head, actual_head in zip(expected, actual):
            np.testing.assert_allclose(actual_head, expected_head, rtol=1e-6, atol=1e-6)
        return evaluate(self)

    monkeypatch.setattr(inference.ExperimentVolumeEvaluator, "evaluate", check_model)
    reference_override = None
    if architecture == "3drnet":
        reference_override = Path(__file__).resolve().parents[1] / "example/RARE/TBI_fm_19_49/Anat/TBI_fm_19_49_N4.nii.gz"
    selection = inference.best_checkpoint_settings(run, reference=reference_override)
    assert selection.model_file == selected.resolve()
    assert selection.subject_ids == ("D1/SUBJ_0003",)
    assert selection.checkpoint_kind == "weights"
    assert selection.reference == reference_override
    assert run.resolve() not in selection.output_dir.parents
    result = inference.evaluate_saved_model(selection)
    assert set(result.metrics["Scan"]) == {"SUBJ_0003_7days", "SUBJ_0003_14days"}
    assert set(result.metrics["DatasetRole"]) == {"test"}
    assert artifact_hashes(run) == before
    provenance = json.loads((result.output_dir / "evaluation.json").read_text())
    assert provenance["checkpoint_kind"] == "weights"
    assert provenance["model_sha256"] == hashlib.sha256(selected.read_bytes()).hexdigest()
    if reference_override is not None:
        assert Path(provenance["evaluation_experiment"]["preprocessing"]["reference"]) == reference_override
    assert Path(provenance["source_experiment"]["preprocessing"]["reference"]) == config.preprocessing.reference

    reopened = inference.load_saved_evaluation(result.output_dir)
    assert set(reopened.metrics["Scan"]) == set(result.metrics["Scan"])
    inventory = inference.saved_prediction_manifest(reopened.data, reopened.output_dir)
    assert len(inventory) == 2
    assert set(inventory["Fold"]) == {0}
    for path in inventory["PredictionPath"]:
        for name in ("regions", "brain_mask"):
            mask = nib.load(Path(path).with_name(f"{name}.nii.gz"))
            labels = mask.get_fdata()
            np.testing.assert_array_equal(labels, np.rint(labels))
            assert mask.dataobj.slope == 1
            assert mask.dataobj.inter == 0
    assert inference.saved_prediction_manifest(reopened.data, reopened.output_dir, role="validation").empty
    figure = inference.prediction_comparison_figure(reopened.data, inventory, max_slices=2)
    assert len(figure.layout.updatemenus[0].buttons) == 6
    saved_metrics = (result.output_dir / "full_volume_metrics.csv").read_bytes()
    viewed = inference.load_saved_evaluation(result.output_dir, reference=config.preprocessing.reference)
    assert viewed.data.config.preprocessing.reference == config.preprocessing.reference
    assert (result.output_dir / "full_volume_metrics.csv").read_bytes() == saved_metrics
    assert artifact_hashes(run) == before


def test_cv_selection_distinguishes_holdout_validation_and_shared_test(experiment, tmp_path):
    run = tmp_path / "cv"
    fold = write_run(experiment, run)
    root_split = [{"train": ["D1/SUBJ_0000"], "validation": ["D1/SUBJ_0002"], "test": ["D1/SUBJ_0001"]}]
    (run / "splits.json").write_text(json.dumps(root_split))
    (fold / "split.json").write_text(json.dumps(root_split[0]))
    metadata = {"evaluation": {"test_ids": ["D1/SUBJ_0003"]}}
    (run / "metadata.json").write_text(json.dumps(metadata))
    (fold / "best.weights.h5").write_bytes(b"selection only")
    assert inference.best_checkpoint_settings(run, role="holdout").subject_ids == ("D1/SUBJ_0001",)
    assert inference.best_checkpoint_settings(run, role="validation").subject_ids == ("D1/SUBJ_0002",)
    assert inference.best_checkpoint_settings(run, role="test").subject_ids == ("D1/SUBJ_0003",)

    del root_split[0]["test"]
    (fold / "split.json").write_text(json.dumps(root_split[0]))
    with pytest.raises(ValueError, match="No saved animals"):
        inference.best_checkpoint_settings(run, role="holdout")
    assert inference.best_checkpoint_settings(run, role="validation").role == "validation"


def test_checkpoint_requires_a_saved_epoch_and_separate_output(experiment, tmp_path):
    run = tmp_path / "run"
    fold = write_run(experiment, run)
    with pytest.raises(FileNotFoundError, match="complete an epoch"):
        inference.best_checkpoint_settings(run)
    (fold / "best.weights.h5").write_bytes(b"selection only")
    with pytest.raises(ValueError, match="outside"):
        inference.best_checkpoint_settings(run, output_dir=run / "preview")
    with pytest.raises(ValueError, match="Fixed-split"):
        inference.best_checkpoint_settings(run, role="holdout")
    selection = inference.best_checkpoint_settings(run)
    unsafe = replace(selection, output_dir=run / "preview")
    with pytest.raises(ValueError, match="outside"):
        inference.evaluate_saved_model(unsafe)


@pytest.mark.parametrize("filename,role", [
    ("01_tbi_mouse_training.ipynb", "test"),
    ("02_tbi_mouse_kfold.ipynb", "holdout"),
    ("04_rat_kfold.ipynb", "validation"),
    ("07f_ablation.ipynb", "holdout"),
])
def test_final_checkpoint_cell_needs_no_training_result_or_live_config(
    experiment, tmp_path, monkeypatch, filename, role,
):
    """Exercise the real final cell with only repository setup and a run path."""
    run = tmp_path / "run"
    fold = write_run(experiment, run)
    if role == "holdout":
        split = json.loads((fold / "split.json").read_text())
        (run / "splits.json").write_text(json.dumps([split]))
    (fold / "best.weights.h5").write_bytes(b"selection only")
    observed = []

    def evaluate(settings):
        observed.append(settings)
        metrics = pd.DataFrame({"Metric": ["Dice"], "Value": [0.5]})
        return SimpleNamespace(metrics=metrics, data=object(), output_dir=settings.output_dir)

    class Figure:
        def show(self, *, renderer):
            assert renderer == "vscode"

    monkeypatch.setattr(inference, "evaluate_saved_model", evaluate)
    monkeypatch.setattr(inference, "saved_prediction_manifest", lambda **kwargs: object())
    monkeypatch.setattr(inference, "prediction_comparison_figure", lambda **kwargs: Figure())
    path = Path(__file__).resolve().parents[1] / "notebook" / filename
    notebook = json.loads(path.read_text(encoding="utf-8"))
    tree = ast.parse("".join(notebook["cells"][-1]["source"]))
    overrides = {"RUN_CHECKPOINT_EVALUATION": True, "CHECKPOINT_ROLE": role}
    for node in tree.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id in overrides:
                node.value = ast.Constant(overrides[node.target.id])
            elif node.target.id == "CHECKPOINT_RUN":
                node.value = ast.parse(f"Path({str(run)!r})", mode="eval").body
    context = {"Path": Path, "REFERENCE": experiment.preprocessing.reference, "display": lambda value: None}
    ast.fix_missing_locations(tree)
    exec(compile(tree, str(path), "exec"), context)
    assert len(observed) == 1
    assert observed[0].role == role
    assert observed[0].checkpoint_kind == "weights"
    assert "result" not in context and "config" not in context and "prepared" not in context
