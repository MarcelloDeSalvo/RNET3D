"""Gate discovery, unchanged inference state, checkpoint loading and notebook flow."""

import ast
import hashlib
import json
from pathlib import Path
from dataclasses import replace

import nibabel as nib
import numpy as np
import plotly.graph_objects as go
import pytest
import tensorflow as tf

from test_experiments import experiment
from src.evaluation.attention import attention_gate_table, attention_figure, inspect_attention, load_attention_model
from src.experiments.training import build_network, training_view
from src.models.networks import mt_r_net_3d


@pytest.fixture
def model():
    tf.keras.backend.clear_session()
    return mt_r_net_3d((16, 16, 16), 1, 3, (2, 4, 8, 16, 32), with_classifier=True)


def test_discovery_uses_gate_connections_and_excludes_classifier_and_brain(model):
    table = attention_gate_table(model)
    assert len(table) == 4
    assert table.NativeShape.tolist() == [(2, 2, 2), (4, 4, 4), (8, 8, 8), (16, 16, 16)]
    assert not set(table.Gate) & {"brain_mask", "classifier"}
    assert len(set(table.CoefficientLayer)) == 4


def test_inspection_preserves_weights_and_actual_sigmoid_scale(model):
    gates = attention_gate_table(model)
    for name in gates.Gate:
        convolution = model.get_layer(name).input._keras_history.layer
        kernel, bias = convolution.get_weights()
        kernel.fill(0)
        bias.fill(np.log(3))
        convolution.set_weights([kernel, bias])
    before = [value.copy() for value in model.get_weights()]
    values = np.random.default_rng(42).normal(size=(16, 16, 16)).astype(np.float32)
    image = nib.Nifti1Image(values, np.diag([0.1, 0.1, 0.1, 1]))
    image.header.set_xyzt_units("mm")
    result = inspect_attention(model, image)
    assert len(result.maps) == 4
    for gate in result.maps.values():
        assert gate.shape == image.shape
        np.testing.assert_array_equal(gate.affine, image.affine)
        assert gate.header.get_xyzt_units()[0] == "mm"
        np.testing.assert_allclose(gate.get_fdata(), 0.75, atol=1e-6)
    for original, current in zip(before, model.get_weights()):
        np.testing.assert_array_equal(original, current)
    np.testing.assert_allclose(result.summary.Mean, 0.75, atol=1e-6)
    with pytest.raises(ValueError, match="shape"):
        inspect_attention(model, nib.Nifti1Image(values[:8], image.affine))


def test_attention_disabled_reports_no_gates():
    tf.keras.backend.clear_session()
    model = mt_r_net_3d((16, 16, 16), 1, 3, (2, 4, 8, 16, 32), attention=False)
    with pytest.raises(ValueError, match="attention disabled"):
        attention_gate_table(model)


def test_flexible_legacy_input_dimensions_are_resolved_from_the_inspected_patch():
    tf.keras.backend.clear_session()
    model = mt_r_net_3d((None, None, None), 1, 3, (2, 4, 8, 16, 32))
    table = attention_gate_table(model)
    assert len(table) == 4 and all(shape == (None, None, None) for shape in table.NativeShape)
    image = nib.Nifti1Image(np.zeros((16, 16, 16), dtype=np.float32), np.eye(4))
    result = inspect_attention(model, image)
    assert result.summary.NativeShape.tolist() == [(2, 2, 2), (4, 4, 4), (8, 8, 8), (16, 16, 16)]
    assert all(gate.shape == image.shape for gate in result.maps.values())


def saved_run(experiment, folder):
    """Write a tiny exact fitting graph, with a deliberately different best path."""
    config = replace(
        experiment, model=replace(experiment.model, architecture="3drnet", attention=True),
        loss=replace(experiment.loss, gradient_fix_version=1),
    )
    fold = folder / "fold_0"
    fold.mkdir(parents=True)
    (folder / "experiment.json").write_text(json.dumps(config.to_dict(), default=str))
    model = training_view(build_network(config), config)
    source = fold / "selected_best.weights.h5"
    model.save_weights(str(source))
    (fold / "training_state.json").write_text(json.dumps({"selected_checkpoint": source.name}))
    (fold / "best.weights.h5").write_bytes(b"not the selected checkpoint")
    return config, model, source


def test_run_loader_follows_saved_selection_without_compiling_or_initializing(experiment, tmp_path, monkeypatch):
    folder = tmp_path / "run"
    _, original, source = saved_run(experiment, folder)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    def forbidden(*args, **kwargs):
        pytest.fail("Attention inspection must not compile, initialize transfer weights or fit")
    monkeypatch.setattr(tf.keras.Model, "compile", forbidden)
    monkeypatch.setattr(tf.keras.Model, "fit", forbidden)
    monkeypatch.setattr("src.experiments.training.initialize_model", forbidden)
    loaded, config, selected = load_attention_model(run_dir=folder)
    assert selected == source and config is not None
    assert loaded.output_names[0] == "regions_logits"
    for expected, actual in zip(original.get_weights(), loaded.get_weights()):
        np.testing.assert_array_equal(expected, actual)
    assert hashlib.sha256(source.read_bytes()).hexdigest() == digest
    assert len(attention_gate_table(loaded)) == 4


def test_complete_hdf5_model_loads_without_experiment_and_keeps_gate_topology(model, tmp_path):
    source = tmp_path / "legacy.h5"
    model.save(str(source), include_optimizer=False)
    loaded, config, selected = load_attention_model(model_file=source, checkpoint_kind="model")
    assert config is None and selected == source
    assert len(attention_gate_table(loaded)) == 4
    np.testing.assert_array_equal(model.get_weights()[0], loaded.get_weights()[0])
    with pytest.raises(ValueError, match="experiment.json"):
        load_attention_model(model_file=source, checkpoint_kind="weights")


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_slice_slider_keeps_maps_mri_and_gt_aligned(model, axis):
    image = nib.Nifti1Image(np.ones((16, 16, 16), dtype=np.float32), np.eye(4))
    result = inspect_attention(model, image)
    labels = nib.Nifti1Image(np.ones(image.shape, dtype=np.int16), image.affine)
    figure = attention_figure(result, labels=labels, axis=axis, max_slices=3)
    assert len(figure.frames) == 3 and len(figure.data) == 14
    assert figure.layout.coloraxis.cmin == 0 and figure.layout.coloraxis.cmax == 1
    assert all(len(frame.data) == len(figure.data) for frame in figure.frames)
    bad = nib.Nifti1Image(np.ones((8, 8, 8), dtype=np.int16), image.affine)
    with pytest.raises(ValueError, match="share the MRI patch grid"):
        attention_figure(result, labels=bad)


@pytest.mark.parametrize("center", [None, (8, 8, 8)])
def test_notebook_runs_on_saved_weights_and_paired_files_without_training(experiment, tmp_path, monkeypatch, center):
    folder = tmp_path / "run"
    saved_run(experiment, folder)
    shown = []
    monkeypatch.setattr(go.Figure, "show", lambda figure, **kwargs: shown.append(kwargs))
    monkeypatch.setattr(tf.keras.Model, "fit", lambda *args, **kwargs: pytest.fail("No fitting"))
    monkeypatch.setattr(tf.keras.Model, "compile", lambda *args, **kwargs: pytest.fail("No compilation"))
    path = Path(__file__).resolve().parents[1] / "notebook/check_attention.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    context = {}
    replacements = {
        "RUN_DIR": f"Path({str(folder)!r})", "INPUT_MODE": "'files'", "MAX_SLICES": "3",
        "IMAGE_PATH": f"Path({str(experiment.preprocessing.reference)!r})",
        "ROI_PATH": f"Path({str(experiment.preprocessing.reference.parent / 'Labels.nii.gz')!r})",
        "PATCH_CENTER": repr(center),
    }
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        tree = ast.parse("".join(cell["source"]))
        for node in tree.body:
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                if node.target.id in replacements:
                    node.value = ast.parse(replacements[node.target.id], mode="eval").body
        ast.fix_missing_locations(tree)
        exec(compile(tree, str(path), "exec"), context)
    assert len(context["attention_result"].maps) == 4
    assert nib.aff2axcodes(context["patch_image"].affine) == ("R", "I", "A")
    assert context["patch_roi"].shape == context["patch_image"].shape
    np.testing.assert_array_equal(context["patch_roi"].affine, context["patch_image"].affine)
    assert shown == [{"renderer": "vscode"}]
