"""Preserve numerical objectives and expose the existing partial-label coupling."""

import json
from pathlib import Path

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from src.evaluation import losses, metrics
from src.models import networks


BASELINE_PATH = Path(__file__).parent / "fixtures" / "objective_baselines.json"
BASELINES = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))


@pytest.mark.parametrize("case", BASELINES["cases"])
@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "graph"])
def test_original_values_gradients_and_callback_names(case, compiled):
    """Check pre-refactor references in eager execution and tf.function."""
    module = losses if case["module"] == "losses" else metrics
    factory = getattr(module, case["factory"])
    callback = factory(**case["options"])
    assert callback.__name__ == case["callback_name"]

    if compiled:
        callback = tf.function(callback)

    target_key = "masked_target" if case["masked"] else "target"
    target = tf.constant(BASELINES["inputs"][target_key], dtype=tf.float32)
    predicted = tf.Variable(BASELINES["inputs"]["probabilities"], dtype=tf.float32)

    with tf.GradientTape() as tape:
        value = callback(target, predicted)

    np.testing.assert_allclose(value.numpy(), case["value"], rtol=1e-6, atol=1e-7)
    if "gradient" in case:
        gradient = tape.gradient(value, predicted)
        np.testing.assert_allclose(gradient.numpy(), case["gradient"], rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("batch_wise", [False, True])
def test_current_masking_blocks_probability_but_not_softmax_logit_gradients(batch_wise):
    """Document the existing behavior; this is not a claim of correct partial-label learning."""
    target = tf.constant([[[[[-1., 1., -1., 0., -1., -1., -1., 0., -1., -1., 0.]]]]])
    logits = tf.Variable(tf.zeros_like(target))
    objective = losses.diceCELoss(gamma=0.8, batch_wise=batch_wise, domain_adaptation=True)

    with tf.GradientTape(persistent=True) as tape:
        probabilities = tf.nn.softmax(logits, axis=-1)
        tape.watch(probabilities)
        value = objective(target, probabilities)

    probability_gradient = tape.gradient(value, probabilities).numpy()
    logit_gradient = tape.gradient(value, logits).numpy()
    unavailable = target.numpy() == -1

    np.testing.assert_array_equal(probability_gradient[unavailable], 0)
    assert np.all(np.abs(logit_gradient[unavailable]) > 1e-4)

    changed_logits = np.zeros(target.shape, dtype=np.float32)
    changed_logits[..., 2] = 2
    changed_probabilities = tf.nn.softmax(changed_logits, axis=-1)
    changed_value = objective(target, changed_probabilities)
    assert not np.isclose(changed_value.numpy(), value.numpy())

    # Check the unavailable logit's gradient against a numerical derivative.
    epsilon = 1e-3
    perturbation = np.zeros(target.shape, dtype=np.float32)
    perturbation[..., 2] = epsilon
    positive = objective(target, tf.nn.softmax(perturbation, axis=-1))
    negative = objective(target, tf.nn.softmax(-perturbation, axis=-1))
    difference = (positive - negative) / (2 * epsilon)
    np.testing.assert_allclose(difference.numpy(), logit_gradient[..., 2].item(), rtol=0.005, atol=1e-5)


def test_regions_head_is_softmax():
    """Tie the coupling check to the actual model output activation."""
    tf.keras.backend.clear_session()
    model = networks.mt_r_net_3d((16, 16, 16), 1, 11, [1, 2, 4, 8, 16], with_classifier=True)
    activation = model.get_layer("regions").activation
    assert tf.keras.activations.serialize(activation) == "softmax"
    tf.keras.backend.clear_session()


def test_empty_dice_and_masked_classes_keep_historical_reductions():
    """Retain unusual smoothing and class averaging instead of fixing them silently."""
    empty = tf.zeros((1, 1, 1, 1, 3))
    ignored = -tf.ones_like(empty)
    probabilities = tf.ones_like(empty) / 3

    assert losses.diceLoss()(empty, empty).numpy() == pytest.approx(-1)
    assert losses.batchDiceLoss()(empty, empty).numpy() == pytest.approx(-1)
    objective = losses.diceCELoss(gamma=0.8, domain_adaptation=True)
    assert objective(ignored, probabilities).numpy() == pytest.approx(-0.2)
    metric = metrics.dice_coefficient(domain_adaptation=True, exclude_background=False)
    assert metric(ignored, probabilities).numpy() == pytest.approx(1)


@pytest.mark.parametrize("factory", [losses.batchDiceLoss, losses.batchTverskyLoss, metrics.dice_coefficient])
def test_graph_accepts_dynamic_class_dimension(factory):
    """Integer loop bounds work even when channel count is known only at runtime."""
    signature = [tf.TensorSpec((None, None, None, None, None), tf.float32)] * 2
    objective = factory()
    compiled = tf.function(objective, input_signature=signature)
    target = tf.constant(BASELINES["inputs"]["target"], dtype=tf.float32)
    predicted = tf.constant(BASELINES["inputs"]["probabilities"], dtype=tf.float32)
    expected = objective(target, predicted)
    actual = compiled(target, predicted)
    np.testing.assert_allclose(actual.numpy(), expected.numpy(), rtol=1e-6, atol=1e-7)
