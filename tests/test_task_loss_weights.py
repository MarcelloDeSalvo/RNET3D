"""Verify task weights affect the total objective without changing head losses."""

from dataclasses import replace

import numpy as np
import pytest
import tensorflow as tf

from test_experiments import experiment
from src.experiments.config import ExperimentConfig, LossConfig, ModelConfig
from src.experiments import training


@pytest.mark.parametrize('objective', ['dice_ce', 'dice_ce_gradient_fix'])
@pytest.mark.parametrize('classifier', [False, True])
def test_weighted_total_and_disabled_brain_gradient(experiment, monkeypatch, objective, classifier):
    def tiny_network(config):
        """Keep real compilation/losses while avoiding an expensive encoder."""
        inputs = tf.keras.Input((4, 4, 4, 1))
        logits = tf.keras.layers.Conv3D(3, 1)(inputs)
        regions = tf.keras.layers.Activation('softmax', name='regions')(logits)
        brain = tf.keras.layers.Conv3D(1, 1, activation='sigmoid', name='brain_mask')(inputs)
        outputs = [regions, brain]
        if config.model.classifier:
            pooled = tf.keras.layers.GlobalAveragePooling3D()(inputs)
            condition = tf.keras.layers.Dense(1, activation='sigmoid', name='classifier')(pooled)
            outputs.append(condition)
        return tf.keras.Model(inputs, outputs)

    monkeypatch.setattr(training, 'build_network', tiny_network)
    loss = LossConfig(regions=objective, regions_weight=2.0, brain_weight=0.0, classifier_weight=0.25)
    model_config = ModelConfig(classifier=classifier)
    config = replace(experiment, loss=loss, model=model_config)
    model = training.build_model(config)
    inputs = np.ones((2, 4, 4, 4, 1), dtype=np.float32)
    labels = np.ones((2, 4, 4, 4), dtype=np.int32)
    regions = tf.one_hot(labels, 3).numpy()
    if objective == 'dice_ce_gradient_fix':
        regions[..., 2] = -1
    targets = [regions, np.ones_like(inputs)]
    if classifier:
        targets.append(np.asarray([[1.0], [0.0]], dtype=np.float32))

    metrics = model.test_on_batch(inputs, targets, return_dict=True)
    roi_head = model.output_names[0]
    expected = 2.0 * metrics[f'{roi_head}_loss']
    if classifier:
        expected += 0.25 * metrics['classifier_loss']
    np.testing.assert_allclose(metrics['loss'], expected, rtol=1e-6)
    assert metrics['brain_mask_loss'] > 0  # Still observable even when weight is zero.

    tensor_targets = [tf.convert_to_tensor(target) for target in targets]
    with tf.GradientTape() as tape:
        predictions = model(inputs, training=False)
        value = model.compiled_loss(tensor_targets, predictions)
    gradients = tape.gradient(value, model.get_layer('brain_mask').trainable_variables)
    for gradient in gradients:
        np.testing.assert_array_equal(gradient.numpy(), np.zeros_like(gradient.numpy()))


@pytest.mark.parametrize('weight', [-1.0, float('nan'), float('inf'), True, '1'])
def test_invalid_task_weights_fail_before_build(experiment, weight):
    config = replace(experiment, loss=replace(experiment.loss, brain_weight=weight))
    with pytest.raises(ValueError, match='Task loss weights'):
        config.validate()


def test_inactive_classifier_cannot_supply_the_only_positive_weight():
    settings = LossConfig(regions_weight=0.0, brain_weight=0.0, classifier_weight=1.0)
    with pytest.raises(ValueError, match='active task'):
        settings.validate_weights(with_classifier=False)
    settings.validate_weights(with_classifier=True)


def test_older_configs_keep_equal_task_weights(experiment):
    values = experiment.to_dict()
    for field in ('regions_weight', 'brain_weight', 'classifier_weight'):
        values['loss'].pop(field)
    loaded = ExperimentConfig.from_dict(values)
    assert loaded.loss.regions_weight == loaded.loss.brain_weight == loaded.loss.classifier_weight == 1.0
