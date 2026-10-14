"""Verify direct known-negative supervision and unchanged ignored-logit gradients."""

import json
from dataclasses import replace

import nibabel as nib
import numpy as np
import pytest
import tensorflow as tf

from test_experiments import experiment
from src.evaluation.losses import diceCELoss_gradient_fix
from src.experiments.config import DatasetSpec
from src.experiments.data import prepare_data


@pytest.mark.parametrize('batch_wise', [False, True])
@pytest.mark.parametrize('compiled', [False, True])
def test_known_empty_lesion_has_strong_false_positive_gradient(batch_wise, compiled):
    target = tf.constant([[[[[1., 0., -1.]]]]])
    logits = tf.Variable([[[[[0., 20., 1000.]]]]])
    objective = diceCELoss_gradient_fix(version=3, gamma=1.0, batch_wise=batch_wise)
    if compiled:
        converted = tf.autograph.to_graph(objective)
        objective = tf.function(converted, autograph=False)
    with tf.GradientTape() as tape:
        loss = objective(target, logits)
    gradient = tape.gradient(loss, logits).numpy()
    np.testing.assert_allclose(loss.numpy(), np.logaddexp(0., 20.), rtol=1e-6)
    assert gradient[..., 1].item() > 0.99
    assert gradient[..., 0].item() < -0.99
    assert gradient[..., 2].item() == 0.0
    changed = tf.constant([[[[[0., 20., -1000.]]]]])
    np.testing.assert_array_equal(objective(target, changed).numpy(), loss.numpy())


@pytest.mark.parametrize('dtype', [tf.float16, tf.float32, tf.float64])
@pytest.mark.parametrize('values', [[0., -1000., 10000.], [0., 1000., -10000.], [0., 0., 0.]])
def test_extreme_logits_have_finite_values_and_gradients(dtype, values):
    target = tf.constant([[[[[1., 0., -1.]]]]], dtype=dtype)
    logits = tf.Variable([[[[values]]]], dtype=dtype)
    objective = diceCELoss_gradient_fix(version=3)
    with tf.GradientTape() as tape:
        loss = objective(target, logits)
    gradient = tape.gradient(loss, logits).numpy()
    assert np.isfinite(loss.numpy()).all()
    assert np.isfinite(gradient).all()
    assert gradient[..., 2].item() == 0.0


@pytest.mark.parametrize('values', [[-1., -1., -1.], [1., -1., -1.]])
def test_no_foreground_supervision_has_zero_loss_and_gradient(values):
    target = tf.constant([[[[values]]]])
    logits = tf.Variable(tf.zeros_like(target))
    with tf.GradientTape() as tape:
        loss = diceCELoss_gradient_fix(version=3)(target, logits)
    gradient = tape.gradient(loss, logits)
    assert loss.numpy() == 0
    np.testing.assert_array_equal(gradient.numpy(), 0)


def test_positive_and_negative_entries_are_balanced_per_available_roi():
    target = tf.constant([[[[[0., 1., -1.], [1., 0., -1.]]]]])
    logits = tf.constant([[[[[0., 2., 99.], [0., 3., -99.]]]]])
    loss = diceCELoss_gradient_fix(version=3, gamma=1.0, positive_weight=0.3)(target, logits)
    expected = 0.3 * np.logaddexp(0., -2.) + 0.7 * np.logaddexp(0., 3.)
    np.testing.assert_allclose(loss.numpy(), expected, rtol=1e-6)


def test_multiclass_bce_matches_one_vs_rest_and_finite_difference():
    target = tf.constant([[[[[0., 1., 0., -1.], [1., 0., 0., -1.]]]]], dtype=tf.float64)
    values = np.array([[[[[.3, .7, -.2, 99.], [-.8, .3, .4, -99.]]]]])
    logits = tf.Variable(values)
    objective = diceCELoss_gradient_fix(version=3, gamma=1.0)
    with tf.GradientTape() as tape:
        loss = objective(target, logits)
    gradient = tape.gradient(loss, logits).numpy()
    for voxel in [0, 1]:
        for channel in range(3):
            upper = values.copy()
            lower = values.copy()
            upper[0, 0, 0, voxel, channel] += 1e-5
            lower[0, 0, 0, voxel, channel] -= 1e-5
            derivative = (objective(target, upper).numpy() - objective(target, lower).numpy()) / 2e-5
            np.testing.assert_allclose(derivative, gradient[0, 0, 0, voxel, channel], rtol=1e-5, atol=1e-7)
    np.testing.assert_array_equal(gradient[..., 3], 0)


def test_verified_sham_absence_becomes_an_available_empty_lesion_target(experiment):
    folder = experiment.data.root / 'D1' / 'SUBJ_0000'
    path = folder / 'metadata.json'
    metadata = json.loads(path.read_text())
    metadata.update(condition='SHAM', has_lesion_annotation=False, annotated_regions=['cl-CC'])
    path.write_text(json.dumps(metadata))
    mask_path = folder / 'Labels.nii.gz'
    mask = nib.load(mask_path)
    labels = mask.get_fdata()
    labels[labels == 1] = 0
    nib.save(nib.Nifti1Image(labels, mask.affine), mask_path)
    loss = replace(experiment.loss, gradient_fix_version=3)
    unchanged = prepare_data(replace(experiment, loss=loss))
    assert 1 in unchanged.loader.get_subject(('D1_T1w', 'SUBJ_0000')).unavailable_labels
    dataset = DatasetSpec(sham_lesion_absent=True)
    config = replace(experiment, loss=loss, data=replace(experiment.data, datasets=(dataset,)))
    data = prepare_data(config)
    assert 1 not in data.loader.get_subject(('D1_T1w', 'SUBJ_0000')).unavailable_labels
    _, targets = data.make_generator(['D1/SUBJ_0000'], training=False)[0]
    np.testing.assert_array_equal(targets[0][..., 1], 0)
    assert 2 in data.loader.get_subject(('D1_T1w', 'SUBJ_0001')).unavailable_labels


def test_verified_absence_rejects_lesion_positive_sham_masks(experiment):
    path = experiment.data.root / 'D1' / 'SUBJ_0000' / 'metadata.json'
    metadata = json.loads(path.read_text())
    metadata['condition'] = 'SHAM'
    path.write_text(json.dumps(metadata))
    data = replace(experiment.data, datasets=(DatasetSpec(sham_lesion_absent=True),))
    with pytest.raises(ValueError, match='conflicts'):
        prepare_data(replace(experiment, data=data))
