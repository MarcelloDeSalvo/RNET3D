"""Analytical 3D classifier attribution and optional notebook integration."""

import ast
import json
from pathlib import Path
from typing import Literal

import nibabel as nib
import numpy as np
import plotly.graph_objects as go
import pytest
import tensorflow as tf

from src.evaluation.attention import classifier_feature_table, classifier_gradcam, gradcam_figure
from src.models.networks import mt_r_net_3d


def linear_classifier(*, bias=0.0, kernel=1.0):
    """A known score: kernel * spatial mean(input) + bias, followed by sigmoid."""
    inputs = tf.keras.Input((4, 4, 4, 1))
    features = tf.keras.layers.Conv3D(1, 1, use_bias=False, kernel_initializer="ones", name="features")(inputs)
    pooled = tf.keras.layers.GlobalAveragePooling3D()(features)
    output = tf.keras.layers.Dense(1, activation="sigmoid", name="classifier")(pooled)
    model = tf.keras.Model(inputs, output)
    model.get_layer("classifier").set_weights([
        np.array([[kernel]], dtype=np.float32), np.array([bias], dtype=np.float32),
    ])
    return model


def test_gradcam_matches_analytical_positive_and_negative_evidence_without_mutating_model():
    model = linear_classifier()
    data = np.ones((4, 4, 4), dtype=np.float32)
    data[:2] = -1
    image = nib.Nifti1Image(data, np.diag([0.1, 0.1, 0.1, 1]))
    image.header.set_xyzt_units("mm")
    before = [value.copy() for value in model.get_weights()]
    activation = model.get_layer("classifier").activation
    positive = classifier_gradcam(model, image, target="TBI")
    negative = classifier_gradcam(model, image, target="SHAM")
    np.testing.assert_array_equal(positive.native_map, np.maximum(data, 0))
    np.testing.assert_array_equal(negative.native_map, np.maximum(-data, 0))
    assert positive.summary.TBIPatchProbability.item() == 0.5
    assert negative.summary.Target.item() == "SHAM"
    assert positive.summary.GradientL1.item() == pytest.approx(1)
    assert positive.summary.FeatureLayer.item() == "features"
    assert positive.heatmap.header.get_xyzt_units()[0] == "mm"
    np.testing.assert_array_equal(positive.heatmap.affine, image.affine)
    assert model.get_layer("classifier").activation is activation
    for expected, actual in zip(before, model.get_weights()):
        np.testing.assert_array_equal(expected, actual)


def test_pre_sigmoid_score_retains_gradient_when_probability_saturates_and_model_is_frozen():
    model = linear_classifier(bias=100)
    model.trainable = False
    image = nib.Nifti1Image(np.ones((4, 4, 4), dtype=np.float32), np.eye(4))
    result = classifier_gradcam(model, image)
    assert result.summary.TBIPatchProbability.item() == 1
    assert result.summary.Target.item() == "TBI"
    assert result.summary.Status.item() == "ok"
    assert result.summary.GradientL1.item() == pytest.approx(1)
    np.testing.assert_array_equal(result.native_map, np.ones((4, 4, 4)))
    negative = classifier_gradcam(model, image, target="SHAM")
    assert negative.summary.Status.item() == "no_positive_attribution"
    assert not negative.native_map.any()


def test_zero_gradient_has_zero_finite_map_and_half_probability_resolves_to_sham():
    model = linear_classifier(kernel=0)
    image = nib.Nifti1Image(np.ones((4, 4, 4), dtype=np.float32), np.eye(4))
    result = classifier_gradcam(model, image)
    assert result.summary.Target.item() == "SHAM"
    assert result.summary.Status.item() == "zero_gradient"
    assert not result.heatmap.get_fdata().any()
    assert np.isfinite(result.native_map).all()


def test_separate_sigmoid_activation_uses_the_original_linear_score():
    inputs = tf.keras.Input((4, 4, 4, 1))
    features = tf.keras.layers.Conv3D(1, 1, use_bias=False, kernel_initializer="ones")(inputs)
    pooled = tf.keras.layers.GlobalAveragePooling3D()(features)
    logits = tf.keras.layers.Dense(1, use_bias=False, kernel_initializer="ones")(pooled)
    outputs = tf.keras.layers.Activation("sigmoid", name="classifier")(logits)
    model = tf.keras.Model(inputs, outputs)
    image = nib.Nifti1Image(np.ones((4, 4, 4), dtype=np.float32), np.eye(4))
    result = classifier_gradcam(model, image, target="TBI")
    assert result.summary.TBILogit.item() == pytest.approx(1)
    np.testing.assert_array_equal(result.native_map, np.ones((4, 4, 4)))


def test_real_classifier_uses_only_encoder_features_and_preserves_batchnorm_state():
    tf.keras.backend.clear_session()
    model = mt_r_net_3d((16, 16, 16), 1, 3, (2, 4, 8, 16, 32), with_classifier=True)
    table = classifier_feature_table(model)
    assert table.Default.sum() == 1
    assert not any(name.startswith("up_sampling") for name in table.Layer)
    before = [value.copy() for value in model.get_weights()]
    data = np.random.default_rng(8).normal(size=(16, 16, 16)).astype(np.float32)
    result = classifier_gradcam(model, nib.Nifti1Image(data, np.eye(4)))
    assert result.native_map.shape == (1, 1, 1)
    assert np.isfinite(result.heatmap.get_fdata()).all()
    for expected, actual in zip(before, model.get_weights()):
        np.testing.assert_array_equal(expected, actual)
    with pytest.raises(ValueError, match="classifier path"):
        classifier_gradcam(model, result.image, feature_layer="brain_mask")
    with pytest.raises(ValueError, match="target"):
        classifier_gradcam(model, result.image, target="lesion")
    with pytest.raises(ValueError, match="shape"):
        classifier_gradcam(model, nib.Nifti1Image(data[:8], np.eye(4)))


def test_no_classifier_is_explicitly_unsupported():
    inputs = tf.keras.Input((4, 4, 4, 1))
    outputs = tf.keras.layers.Activation("sigmoid", name="brain_mask")(inputs)
    model = tf.keras.Model(inputs, outputs)
    with pytest.raises(ValueError, match="no auxiliary classifier"):
        classifier_feature_table(model)


def test_gradcam_plot_uses_attribution_labels_and_preserves_slice_frames():
    model = linear_classifier()
    image = nib.Nifti1Image(np.ones((4, 4, 4), dtype=np.float32), np.eye(4))
    result = classifier_gradcam(model, image, target="TBI")
    figure = gradcam_figure(result, max_slices=3)
    assert "Grad-CAM for TBI" in figure.layout.title.text
    assert figure.layout.coloraxis.colorbar.title.text == "Normalized attribution"
    assert len(figure.frames) == 3
    assert "attribution" in figure.data[2].hovertemplate
    assert figure.layout.coloraxis.cmax == 1


@pytest.mark.parametrize("with_classifier", [True, False])
def test_optional_notebook_gradcam_section_can_run_without_attention_maps(with_classifier, monkeypatch):
    path = Path(__file__).resolve().parents[1] / "notebook/check_attention.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    model = linear_classifier()
    if not with_classifier:
        model = tf.keras.Model(model.inputs, model.get_layer("features").output)
    shown = []
    monkeypatch.setattr(go.Figure, "show", lambda figure, **kwargs: shown.append(kwargs))
    namespace = {"model": model, "patch_image": nib.Nifti1Image(np.ones((4, 4, 4), dtype=np.float32), np.eye(4)),
                 "patch_roi": None, "Literal": Literal, "display": lambda value: None}
    for cell in notebook["cells"]:
        if cell.get("id") not in ("classifier-gradcam-extract", "classifier-gradcam-display"):
            continue
        tree = ast.parse("".join(cell["source"]))
        for node in tree.body:
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                if node.target.id == "RUN_CLASSIFIER_GRADCAM":
                    node.value = ast.Constant(True)
                elif node.target.id == "GRADCAM_MAX_SLICES":
                    node.value = ast.Constant(3)
        ast.fix_missing_locations(tree)
        exec(compile(tree, str(path), "exec"), namespace)
    if with_classifier:
        assert namespace["gradcam_result"] is not None
        assert shown == [{"renderer": "vscode"}]
    else:
        assert namespace["gradcam_result"] is None and shown == []
