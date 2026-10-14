"""Test the separate raw-logit partial-label objective and its gradient contract."""

import json
from pathlib import Path

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from src.evaluation.losses import diceCELoss, diceCELoss_gradient_fix, masked_softmax_gradient_fix


V1_BASELINES = json.loads(
    (Path(__file__).parent / "fixtures" / "partial_label_v1_baselines.json").read_text()
)["records"]


@pytest.mark.parametrize("baseline", V1_BASELINES)
def test_version_one_matches_frozen_original_values_and_gradients(baseline):
    """Restore commit 7a94419's objective, including background/unknown cases."""
    target = tf.constant(baseline["targets"], dtype=tf.float32)
    logits = tf.Variable(baseline["logits"], dtype=tf.float32)
    objective = diceCELoss_gradient_fix(
        gamma=0.8, batch_wise=baseline["batch_wise"],
        focal_gamma=baseline["focal_gamma"], version=1,
    )
    with tf.GradientTape() as tape:
        value = objective(target, logits)
    gradient = tape.gradient(value, logits)
    np.testing.assert_allclose(value.numpy(), baseline["value"], rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(gradient.numpy(), baseline["gradient"], rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("version", [0, 4, True, "1"])
def test_invalid_objective_versions_are_rejected(version):
    with pytest.raises(ValueError, match="version"):
        diceCELoss_gradient_fix(version=version)


@pytest.mark.parametrize("version", [1, 2])
@pytest.mark.parametrize("batch_wise", [False, True])
@pytest.mark.parametrize("focal_gamma", [None, 2.0])
def test_autograph_conversion_succeeds_and_preserves_values_and_gradients(version, batch_wise, focal_gamma):
    """Require actual conversion: tf.function alone can silently fall back."""
    objective = diceCELoss_gradient_fix(
        gamma=0.8, version=version, batch_wise=batch_wise, focal_gamma=focal_gamma,
    )
    converted = tf.autograph.to_graph(objective)
    compiled = tf.function(converted, autograph=False)
    target = tf.constant([[[[[0., 1., -1.], [1., 0., -1.]]]]])
    logits = tf.Variable([[[[[0.3, -0.2, 0.8], [-0.3, 0.6, 0.1]]]]])

    with tf.GradientTape() as tape:
        expected = objective(target, logits)
    expected_gradient = tape.gradient(expected, logits)
    with tf.GradientTape() as tape:
        actual = compiled(target, logits)
    actual_gradient = tape.gradient(actual, logits)

    np.testing.assert_allclose(actual.numpy(), expected.numpy(), rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(actual_gradient.numpy(), expected_gradient.numpy(), rtol=1e-6, atol=1e-7)
    np.testing.assert_array_equal(actual_gradient.numpy()[..., 2], 0)


@pytest.mark.parametrize("batch_wise", [False, True])
@pytest.mark.parametrize("focal_gamma", [None, 2.0])
@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("version", [1, 2])
def test_unavailable_logits_have_zero_gradient_and_no_effect(batch_wise, focal_gamma, compiled, version):
    """Check exact zero gradients and robustness to extreme ignored logits."""
    target = tf.constant([[[[[0., 1., -1.], [1., 0., -1.]]]]])
    values = np.array([[[[[0.3, -0.2, 0.8], [-0.3, 0.6, 0.1]]]]], dtype=np.float32)
    logits = tf.Variable(values)
    objective = diceCELoss_gradient_fix(gamma=0.8, batch_wise=batch_wise, focal_gamma=focal_gamma, version=version)
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
def test_foreground_objective_matches_known_overlap_and_finite_difference(batch_wise, focal_gamma):
    """Two voxels have foreground probabilities 0.8/0.2 and one foreground target."""
    target = tf.constant([[[[[0., 1.], [1., 0.]]]]])
    probabilities = np.array([[[[[0.2, 0.8], [0.8, 0.2]]]]], dtype=np.float32)
    values = np.log(probabilities)
    logits = tf.Variable(values)
    smooth = 1e-5
    objective = diceCELoss_gradient_fix(gamma=0.8, smooth=smooth, batch_wise=batch_wise, focal_gamma=focal_gamma)

    dice = 1 - (2 * 0.8 + smooth) / (1 + 0.8 + 0.2 + smooth)
    ce = -np.log(0.8)
    if focal_gamma is not None:
        ce *= (1 - 0.8) ** focal_gamma
    expected = 0.8 * ce + 0.2 * dice

    with tf.GradientTape() as tape:
        actual = objective(target, logits)
    gradient = tape.gradient(actual, logits).numpy()
    np.testing.assert_allclose(actual.numpy(), expected, rtol=1e-6, atol=1e-7)

    epsilon = 1e-3
    for voxel in (0, 1):
        for channel in (0, 1):
            upper = values.copy()
            lower = values.copy()
            upper[0, 0, 0, voxel, channel] += epsilon
            lower[0, 0, 0, voxel, channel] -= epsilon
            derivative = (objective(target, upper).numpy() - objective(target, lower).numpy()) / (2 * epsilon)
            np.testing.assert_allclose(derivative, gradient[0, 0, 0, voxel, channel], rtol=0.01, atol=3e-5)


@pytest.mark.parametrize("batch_wise", [False, True])
@pytest.mark.parametrize("target_values", [[-1., -1., -1.], [1., -1., -1.]])
def test_no_foreground_annotation_gives_zero_loss_and_gradient(batch_wise, target_values):
    """Background alone is never an explicit optimization target."""
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


@pytest.mark.parametrize("batch_wise", [False, True])
def test_known_foreground_negatives_penalize_false_positives_without_background_ce(batch_wise):
    """A zero at an annotated ROI is a known negative even if its anatomy is unknown."""
    target = tf.constant([[[[[0., 1., -1.], [0., 0., -1.]]]]])
    clean = tf.constant([[[[[0., 2., 100.], [2., 0., 100.]]]]])
    false_positive = tf.Variable([[[[[0., 2., 100.], [0., 2., 100.]]]]])
    dice = diceCELoss_gradient_fix(gamma=0, batch_wise=batch_wise)
    ce = diceCELoss_gradient_fix(gamma=1, batch_wise=batch_wise)

    with tf.GradientTape() as tape:
        value = dice(target, false_positive)
    gradient = tape.gradient(value, false_positive).numpy()
    assert value.numpy() > dice(target, clean).numpy()
    assert gradient[0, 0, 0, 1, 1] > 0
    assert gradient[0, 0, 0, 1, 0] < 0
    np.testing.assert_array_equal(gradient[..., 2], 0)
    np.testing.assert_allclose(ce(target, false_positive), ce(target, clean))


@pytest.mark.parametrize("batch_wise", [False, True])
def test_background_targets_are_omitted_but_the_background_logit_competes(batch_wise):
    foreground = tf.constant([[[[[0., 1., -1.]]]]])
    background = tf.constant([[[[[1., 0., -1.]]]]])
    unknown_anatomy = tf.constant([[[[[0., 0., -1.]]]]])
    logits = tf.Variable(tf.zeros_like(foreground))
    ce = diceCELoss_gradient_fix(gamma=1, batch_wise=batch_wise)
    dice = diceCELoss_gradient_fix(gamma=0, batch_wise=batch_wise)

    with tf.GradientTape() as tape:
        value = ce(foreground, logits)
    gradient = tape.gradient(value, logits).numpy().reshape(-1)
    assert gradient[0] > 0
    assert gradient[1] < 0
    assert gradient[2] == 0
    assert ce(background, logits).numpy() == 0
    assert ce(unknown_anatomy, logits).numpy() == 0
    np.testing.assert_array_equal(dice(background, logits), dice(unknown_anatomy, logits))


def test_annotated_empty_roi_has_finite_nonnegative_loss_and_negative_supervision():
    target = tf.constant([[[[[1., 0., -1.]]]]])
    logits = tf.Variable(tf.zeros_like(target))
    objective = diceCELoss_gradient_fix(gamma=0)
    with tf.GradientTape() as tape:
        value = objective(target, logits)
    gradient = tape.gradient(value, logits).numpy().reshape(-1)
    assert 0 <= value.numpy() <= 1
    assert gradient[1] > 0
    assert gradient[0] < 0
    assert gradient[2] == 0
    perfect_background = tf.constant([[[[[1000., -1000., 0.]]]]])
    assert objective(target, perfect_background).numpy() == 0


def test_ce_retains_mean_over_available_foreground_entries():
    target = tf.constant([[[[[0., 1., 0., -1.]]]]])
    logits = tf.zeros_like(target)
    objective = diceCELoss_gradient_fix(gamma=1)
    np.testing.assert_allclose(objective(target, logits), np.log(3) / 2, rtol=1e-6)


def test_foreground_policy_is_intentionally_different_from_legacy_complete_loss():
    target = tf.constant([[[[[0., 1., 0.], [1., 0., 0.]]]]])
    logits = tf.zeros_like(target)
    old = diceCELoss(gamma=0.8, batch_wise=True)
    new = diceCELoss_gradient_fix(gamma=0.8, batch_wise=True)
    assert not np.isclose(new(target, logits).numpy(), old(target, tf.nn.softmax(logits)).numpy())


def test_background_cannot_be_removed_from_foreground_normalization():
    target = tf.constant([[[[[-1., 1., 0.]]]]])
    objective = diceCELoss_gradient_fix()
    with pytest.raises(tf.errors.InvalidArgumentError, match="Background must remain available"):
        objective(target, tf.zeros_like(target))
