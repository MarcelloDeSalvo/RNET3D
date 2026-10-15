"""Known voxel logits, target masks and unchanged saved segmentation graphs."""

import ast
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Literal

import nibabel as nib
import numpy as np
import plotly.graph_objects as go
import pytest
import tensorflow as tf

from test_experiments import experiment
from src.evaluation.attention import gradcam_figure, segmentation_feature_table, segmentation_gradcam
from src.experiments.training import build_network, training_view
from src.experiments.data import prepare_data
from src.models.networks import mt_r_net_3d


def linear_segmenter(*, logits=False):
    """Known per-voxel scores (-x, x), independent of sigmoid/softmax saturation."""
    inputs = tf.keras.Input((4, 4, 4, 1))
    features = tf.keras.layers.Conv3D(1, 1, use_bias=False, kernel_initializer="ones", name="features")(inputs)
    scores = tf.keras.layers.Conv3D(2, 1, use_bias=False, name="scores")(features)
    activation, name = "softmax", "regions"
    if logits:
        activation, name = "linear", "regions_logits"
    regions = tf.keras.layers.Activation(activation, name=name)(scores)
    brain_score = tf.keras.layers.Conv3D(1, 1, use_bias=False, kernel_initializer="ones")(features)
    brain = tf.keras.layers.Activation("sigmoid", name="brain_mask")(brain_score)
    model = tf.keras.Model(inputs, [regions, brain])
    kernel = np.array([-1, 1], dtype=np.float32).reshape((1, 1, 1, 1, 2))
    model.get_layer("scores").set_weights([kernel])
    return model


@pytest.fixture
def image():
    values = np.ones((4, 4, 4), dtype=np.float32)
    values[:2] = -1
    return nib.Nifti1Image(values, np.diag([0.1, 0.1, 0.1, 1]))


@pytest.mark.parametrize("logits", [False, True])
def test_raw_class_scores_give_opposite_analytical_maps_and_preserve_state(image, logits):
    model = linear_segmenter(logits=logits)
    model.trainable = False
    before = [weight.copy() for weight in model.get_weights()]
    positive = segmentation_gradcam(model, image, class_index=1)
    negative = segmentation_gradcam(model, image, class_index=0)
    np.testing.assert_array_equal(positive.native_map, np.maximum(image.get_fdata(), 0))
    np.testing.assert_array_equal(negative.native_map, np.maximum(-image.get_fdata(), 0))
    assert positive.summary.FeatureLayer.item() == "features"
    assert positive.summary.TargetScore.item() == 0
    assert positive.summary.GradientL1.item() == pytest.approx(1)
    assert positive.summary.TargetVoxels.item() == 64
    np.testing.assert_array_equal(positive.heatmap.affine, image.affine)
    for expected, actual in zip(before, model.get_weights()):
        np.testing.assert_array_equal(expected, actual)


def test_prediction_and_provided_masks_define_fixed_mean_logit_targets(image):
    model = linear_segmenter()
    predicted = segmentation_gradcam(model, image, mask_mode="prediction")
    mask = np.zeros(image.shape, dtype=np.uint8)
    mask[2, 1, 1] = 1
    provided = nib.Nifti1Image(mask, image.affine)
    result = segmentation_gradcam(model, image, mask_mode="provided", spatial_mask=provided)
    assert predicted.summary.TargetVoxels.item() == 32
    assert result.summary.TargetVoxels.item() == 1
    assert result.summary.TargetScore.item() == predicted.summary.TargetScore.item() == 1
    assert result.summary.ScoreReduction.item() == "mean_logit"
    np.testing.assert_array_equal(result.native_map, predicted.native_map)
    brain = segmentation_gradcam(model, image, head="brain_mask", class_index=0, mask_mode="prediction")
    assert brain.summary.TargetVoxels.item() == 32
    assert brain.summary.TargetScore.item() == 1
    figure = gradcam_figure(result, labels=provided, max_slices=3)
    assert "regions channel 1" in figure.layout.title.text
    assert len(figure.frames) == 3


def test_empty_unknown_misaligned_or_nonbinary_targets_are_explicit(image):
    model = linear_segmenter()
    empty = nib.Nifti1Image(np.zeros(image.shape, dtype=np.uint8), image.affine)
    with pytest.raises(ValueError, match="No target voxels"):
        segmentation_gradcam(model, image, mask_mode="provided", spatial_mask=empty)
    ones = nib.Nifti1Image(np.ones(image.shape, dtype=np.float32), image.affine)
    with pytest.raises(ValueError, match="No target voxels"):
        segmentation_gradcam(model, ones, class_index=0, mask_mode="prediction")
    bad_affine = image.affine.copy()
    bad_affine[0, 3] = 0.1
    for invalid in (nib.Nifti1Image(np.zeros(image.shape), bad_affine),
                    nib.Nifti1Image(np.full(image.shape, 2, dtype=np.int16), image.affine)):
        with pytest.raises(ValueError, match="grid|binary"):
            segmentation_gradcam(model, image, mask_mode="provided", spatial_mask=invalid)
    with pytest.raises(ValueError, match="model channel"):
        segmentation_gradcam(model, image, class_index=13)
    with pytest.raises(ValueError, match="chosen segmentation path"):
        segmentation_gradcam(model, image, feature_layer="brain_mask")


@pytest.mark.parametrize("head", ["regions", "brain_mask"])
def test_actual_3drnet_head_uses_finest_decoder_and_keeps_batchnorm_state(head):
    tf.keras.backend.clear_session()
    model = mt_r_net_3d((16, 16, 16), 1, 3, (2, 4, 8, 16, 32), with_classifier=False)
    table = segmentation_feature_table(model, head=head)
    projection = "r_output_1"
    channel = 1
    if head == "brain_mask":
        projection, channel = "b_output_1", 0
    expected_layer = model.get_layer(projection).input._keras_history.layer.name
    assert table.loc[table.Default, "Layer"].item() == expected_layer
    values = np.random.default_rng(9).normal(size=(16, 16, 16)).astype(np.float32)
    before = [weight.copy() for weight in model.get_weights()]
    result = segmentation_gradcam(model, nib.Nifti1Image(values, np.eye(4)), head=head, class_index=channel)
    assert result.native_map.shape == (16, 16, 16)
    assert np.isfinite(result.heatmap.get_fdata()).all()
    for expected, actual in zip(before, model.get_weights()):
        np.testing.assert_array_equal(expected, actual)


def test_unet_training_view_unwraps_linear_output_before_selecting_decoder(experiment):
    model = training_view(build_network(experiment), experiment)
    table = segmentation_feature_table(model)
    layer = model.get_layer("regions_logits").input._keras_history.layer
    expected = layer.input._keras_history.layer.name
    assert table.loc[table.Default, "Layer"].item() == expected


def test_optional_notebook_uses_original_label_mapping_and_native_plotly(image, monkeypatch):
    path = Path(__file__).resolve().parents[1] / "notebook/check_attention.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    labels = np.zeros(image.shape, dtype=np.int16)
    labels[2:] = 13
    config = SimpleNamespace(labels=SimpleNamespace(names=("Background", "Ventricle"), mapping={0: 0, 3: 1, 13: 1}))
    namespace = {"model": linear_segmenter(), "patch_image": image, "patch_roi": nib.Nifti1Image(labels, image.affine),
                 "saved_config": config, "INPUT_MODE": "files", "np": np, "nib": nib,
                 "Literal": Literal, "display": lambda value: None, "gradcam_figure": gradcam_figure}
    shown = []
    monkeypatch.setattr(go.Figure, "show", lambda figure, **kwargs: shown.append(kwargs))
    for cell in notebook["cells"]:
        if cell.get("id") not in ("segmentation-gradcam-extract", "segmentation-gradcam-display"):
            continue
        tree = ast.parse("".join(cell["source"]))
        for node in tree.body:
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                if node.target.id == "RUN_SEGMENTATION_GRADCAM":
                    node.value = ast.Constant(True)
                elif node.target.id == "SEGMENTATION_MASK_MODE":
                    node.value = ast.Constant("provided")
                elif node.target.id == "SEGMENTATION_MAX_SLICES":
                    node.value = ast.Constant(3)
        ast.fix_missing_locations(tree)
        exec(compile(tree, str(path), "exec"), namespace)
    assert namespace["segmentation_result"].summary.TargetVoxels.item() == 32
    np.testing.assert_array_equal(namespace["patch_roi"].get_fdata(), labels)
    assert shown == [{"renderer": "vscode"}]


@pytest.mark.parametrize("mode", ["all", "provided"])
def test_normalized_notebook_does_not_invent_gt_for_unannotated_anatomy(experiment, image, mode):
    data = prepare_data(experiment)
    scan = data.loader.get_subject(("D1_T1w", "SUBJ_0001"))
    config = SimpleNamespace(labels=SimpleNamespace(names=("Background", "CC"), mapping={0: 0, 2: 1}))
    path = Path(__file__).resolve().parents[1] / "notebook/check_attention.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    cell = next(cell for cell in notebook["cells"] if cell.get("id") == "segmentation-gradcam-extract")
    tree = ast.parse("".join(cell["source"]))
    for node in tree.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id == "RUN_SEGMENTATION_GRADCAM":
                node.value = ast.Constant(True)
            elif node.target.id == "SEGMENTATION_MASK_MODE":
                node.value = ast.Constant(mode)
    ast.fix_missing_locations(tree)
    namespace = {"model": linear_segmenter(), "patch_image": image, "patch_roi": image, "scan": scan,
                 "saved_config": config, "INPUT_MODE": "normalized", "np": np, "nib": nib,
                 "Literal": Literal, "display": lambda value: None}
    code = compile(tree, str(path), "exec")
    if mode == "provided":
        with pytest.raises(ValueError, match="available ROI annotations"):
            exec(code, namespace)
    else:
        exec(code, namespace)
        assert namespace["segmentation_result"] is not None
    assert namespace["segmentation_gt"] is None
