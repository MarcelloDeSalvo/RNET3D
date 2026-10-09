"""Test the separate raw-logit partial-label objective and its gradient contract."""

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from src.evaluation.losses import diceCELoss, diceCELoss_gradient_fix, masked_softmax_gradient_fix


@pytest.mark.parametrize("batch_wise", [False, True])
@pytest.mark.parametrize("focal_gamma", [None, 2.0])
@pytest.mark.parametrize("compiled", [False, True])
def test_unavailable_logits_have_zero_gradient_and_no_effect(batch_wise, focal_gamma, compiled):
    """Check exact zero gradients and robustness to extreme ignored logits."""
    target = tf.constant([[[[[1., 0., -1.], [0., 1., -1.]]]]])
    values = np.array([[[[[0.3, -0.2, 0.8], [-0.3, 0.6, 0.1]]]]], dtype=np.float32)
    logits = tf.Variable(values)
    objective = diceCELoss_gradient_fix(gamma=0.8, batch_wise=batch_wise, focal_gamma=focal_gamma)
    if compiled:
        objective = tf.function(objective)

    with tf.GradientTape() as tape:
        baseline = objective(target, logits)
    gradient = tape.gradient(baseline, logits).numpy()
    np.testing.assert_array_equal(gradient[..., 2], 0)
    assert np.any(np.abs(gradient[..., :2]) > 1e-4)

    for ignored_value in (-10000., 2., 10000.):
        changed = values.copy()
        changed[..., 2] = ignored_value
        actual = objective(target, changed)
        np.testing.assert_array_equal(actual.numpy(), baseline.numpy())


@pytest.mark.parametrize("batch_wise", [False, True])
@pytest.mark.parametrize("focal_gamma", [None, 2.0])
def test_fully_annotated_values_and_gradients_match_existing_loss(batch_wise, focal_gamma):
    """The new normalization/reduction equals the old one on fully known voxels."""
    rng = np.random.default_rng(7)
    indexes = rng.integers(0, 3, size=(2, 2, 2, 2))
    target = tf.one_hot(indexes, depth=3)
    logits = tf.Variable(rng.normal(size=target.shape).astype(np.float32))
    original = diceCELoss(gamma=0.8, batch_wise=batch_wise, focal_gamma=focal_gamma)
    fixed = diceCELoss_gradient_fix(gamma=0.8, batch_wise=batch_wise, focal_gamma=focal_gamma)

    with tf.GradientTape() as tape:
        probabilities = tf.nn.softmax(logits, axis=-1)
        expected = original(target, probabilities)
    expected_gradient = tape.gradient(expected, logits)

    with tf.GradientTape() as tape:
        actual = fixed(target, logits)
    actual_gradient = tape.gradient(actual, logits)

    np.testing.assert_allclose(actual.numpy(), expected.numpy(), rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(actual_gradient.numpy(), expected_gradient.numpy(), rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("batch_wise", [False, True])
@pytest.mark.parametrize("target_values", [[-1., -1., -1.], [0., 0., -1.]])
def test_no_supervised_positive_target_gives_zero_loss_and_gradient(batch_wise, target_values):
    """All-ignored and negative-only unknown voxels do not invent supervision."""
    target = tf.constant([[[[target_values]]]])
    logits = tf.Variable(tf.ones_like(target))
    objective = diceCELoss_gradient_fix(batch_wise=batch_wise)

    with tf.GradientTape() as tape:
        value = objective(target, logits)
    gradient = tape.gradient(value, logits)

    assert value.numpy() == 0
    np.testing.assert_array_equal(gradient.numpy(), 0)


@pytest.mark.parametrize("batch_wise", [False, True])
def test_ignored_sample_does_not_dilute_loss(batch_wise):
    """Unavailable entries are excluded from reductions rather than counted as zeros."""
    target = tf.constant([[[[[1., 0., -1.]]]]])
    logits = tf.constant([[[[[0.2, -0.1, 0.7]]]]])
    ignored = -tf.ones_like(target)
    mixed_target = tf.concat([target, ignored], axis=0)
    mixed_logits = tf.concat([logits, logits], axis=0)
    objective = diceCELoss_gradient_fix(batch_wise=batch_wise)

    expected = objective(target, logits)
    actual = objective(mixed_target, mixed_logits)
    np.testing.assert_allclose(actual.numpy(), expected.numpy(), atol=1e-7)


def test_mixed_annotation_patterns_are_masked_per_sample():
    """A channel ignored in one sample can still receive supervision in another."""
    target = tf.constant([[[[[1., 0., -1.]]]], [[[[0., 0., 1.]]]]])
    logits = tf.Variable(tf.zeros_like(target))
    objective = diceCELoss_gradient_fix(batch_wise=True)

    with tf.GradientTape() as tape:
        value = objective(target, logits)
    gradient = tape.gradient(value, logits).numpy()

    assert gradient[0, 0, 0, 0, 2] == 0
    assert abs(gradient[1, 0, 0, 0, 2]) > 1e-4


def test_masked_softmax_is_finite_and_normalized():
    """Keep normalization stable even when an ignored logit dominates by 10000."""
    target = tf.constant([[1., 0., -1.], [-1., -1., -1.]])
    logits = tf.constant([[1., 2., 10000.], [10000., -10000., 2.]])
    probabilities = masked_softmax_gradient_fix(target, logits).numpy()

    assert np.isfinite(probabilities).all()
    np.testing.assert_allclose(probabilities[0].sum(), 1)
    assert probabilities[0, 2] == 0
    np.testing.assert_array_equal(probabilities[1], 0)


@pytest.mark.parametrize("dtype", [tf.float16, tf.float32, tf.float64])
def test_supported_floating_dtypes(dtype):
    target = tf.constant([[[[[1., 0., -1.]]]]], dtype=dtype)
    logits = tf.Variable(tf.zeros_like(target))
    objective = diceCELoss_gradient_fix()

    with tf.GradientTape() as tape:
        value = objective(target, logits)
    gradient = tape.gradient(value, logits).numpy()

    assert np.isfinite(value.numpy())
    assert np.isfinite(gradient).all()
    assert gradient[..., 2].item() == 0
