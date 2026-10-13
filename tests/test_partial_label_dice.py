"""Check training Dice against known overlap and mixed annotation coverage."""

import numpy as np
import pytest
import tensorflow as tf

from src.evaluation.metrics import PartialLabelDice, dice_coefficient


def test_fully_annotated_dice_matches_soft_dice_and_pools_batches():
    labels = tf.constant([0, 1, 2, 1, 0, 2, 1, 2])
    target = tf.reshape(tf.one_hot(labels, 3), (2, 2, 1, 2, 3))
    logits = tf.random.stateless_normal(target.shape, seed=(3, 9))
    probabilities = tf.nn.softmax(logits, axis=-1)
    legacy_dice = dice_coefficient(mode="soft")
    expected = legacy_dice(target, probabilities)

    metric = PartialLabelDice(num_classes=3)
    metric.update_state(target[:1], logits[:1])
    metric.update_state(target[1:], logits[1:])
    np.testing.assert_allclose(metric.result(), expected, rtol=1e-6)

    metric.reset_state()
    assert metric.result().numpy() == 0
    metric.update_state(target, logits)
    np.testing.assert_allclose(metric.result(), expected, rtol=1e-6)


def test_unavailable_classes_are_excluded_but_known_roi_negatives_count():
    # Class 2 is unavailable; the final voxel is a known negative for class 1.
    target = tf.constant([[[[[0., 1., -1.], [1., 0., -1.], [0., 0., -1.]]]]])
    probabilities = tf.constant([[[[[0.1, 0.8, 0.1], [0.6, 0.2, 0.2], [0., 1., 0.]]]]])
    metric = PartialLabelDice(num_classes=3, from_logits=False)
    metric.update_state(target, probabilities)
    expected = (2 * 0.8 + metric.smooth) / (1 + 0.8 + 0.2 + 1 + metric.smooth)
    np.testing.assert_allclose(metric.result(), expected, rtol=1e-6)

    ignored = tf.fill(target.shape, -1.)
    metric.update_state(ignored, probabilities)
    np.testing.assert_allclose(metric.result(), expected, rtol=1e-6)


def test_classes_are_active_if_annotated_in_any_sample():
    target = tf.constant([[[[[0., 1., -1.]]]], [[[[0., -1., 1.]]]]])
    probabilities = tf.constant([[[[[0.1, 0.8, 0.1]]]], [[[[0.2, 0.2, 0.6]]]]])
    metric = PartialLabelDice(num_classes=3, from_logits=False)
    metric.update_state(target, probabilities)
    first = (2 * 0.8 + metric.smooth) / (1 + 0.8 + metric.smooth)
    second = (2 * 0.6 + metric.smooth) / (1 + 0.6 + metric.smooth)
    np.testing.assert_allclose(metric.result(), (first + second) / 2, rtol=1e-6)


@pytest.mark.parametrize("shape", [(), (2,), (2, 1, 1, 1), (2, 1, 1, 1, 1)])
def test_zero_weights_remove_supervision(shape):
    target = tf.ones((2, 1, 1, 1, 3))
    metric = PartialLabelDice(num_classes=3)
    weights = tf.zeros(shape)
    metric.update_state(target, target, sample_weight=weights)
    assert metric.result().numpy() == 0


def test_sample_weights_exclude_one_sample():
    target = tf.constant([[[[[0., 1., 0.]]]], [[[[0., 0., 1.]]]]])
    probabilities = tf.constant([[[[[0., 1., 0.]]]], [[[[0., 1., 0.]]]]])
    metric = PartialLabelDice(num_classes=3, from_logits=False)
    metric.update_state(target, probabilities, sample_weight=tf.constant([1., 0.]))
    np.testing.assert_allclose(metric.result(), 1.)


def test_unsupervised_epoch_is_finite_and_graph_execution_serializes():
    metric = PartialLabelDice(num_classes=3)
    target = tf.fill((1, 2, 1, 1, 3), -1.)
    logits = tf.constant([[[[[1000., -1000., 0.]]], [[[0., 1000., -1000.]]]]])
    update = tf.function(metric.update_state)
    update(target, logits)
    assert metric.result().numpy() == 0

    restored = tf.keras.metrics.deserialize(tf.keras.metrics.serialize(metric))
    assert isinstance(restored, PartialLabelDice)
    assert restored.get_config() == metric.get_config()
    restored.update_state(target, logits)
    assert restored.result().numpy() == 0
