"""Native objective fidelity and one-pass full-volume brain/region measurements."""

import json
from dataclasses import replace

import nibabel as nib
import numpy as np
import pandas as pd
import pytest
import tensorflow as tf

from test_experiments import experiment
from src.experiments.data import prepare_data
from src.experiments.evaluation import evaluate_patches
from src.experiments.training import build_model
from src.evaluation.inference import (
    ExperimentVolumeEvaluator, PredictionResult, SavedModelEvaluationConfig,
    evaluate_saved_model, load_saved_evaluation,
)


def test_native_patch_results_match_keras_and_include_every_tail_patch(experiment):
    """The new export retains the compiled loss and metric definitions exactly."""
    data = prepare_data(experiment)
    model = build_model(experiment)
    animals = ['D1/SUBJ_0002', 'D1/SUBJ_0003']
    generator = data.make_generator(animals, training=False)
    expected = model.evaluate(generator, verbose=0, workers=0, return_dict=True)
    before = [weight.numpy().copy() for weight in model.weights]
    actual = evaluate_patches(model, data, animals)
    for metric, value in expected.items():
        exported = actual.loc[actual.Metric == metric, 'Value'].item()
        np.testing.assert_allclose(exported, value, rtol=1e-6)
    assert actual.PatchCount.eq(3).all()
    assert actual.ScanCount.eq(3).all()
    assert actual.AnimalCount.eq(2).all()
    assert actual.Aggregation.eq('cohort_pooled').all()
    assert int(model.optimizer.iterations.numpy()) == 0
    for weight, previous in zip(model.weights, before):
        np.testing.assert_array_equal(weight.numpy(), previous)


def test_native_evaluation_cannot_silently_use_an_incompatible_partial_objective(experiment):
    config = replace(experiment, loss=replace(experiment.loss, regions='dice_ce'))
    data = prepare_data(config, require_training_loss=False)
    with pytest.raises(ValueError, match='saved gradient-fixed objective'):
        evaluate_patches(object(), data, ['D1/SUBJ_0001'])


def test_saved_weights_export_patch_and_brain_metrics_without_fitting(experiment, tmp_path, monkeypatch):
    """Checkpoint-only evaluation uses the native graph and a separate workbook."""
    model = build_model(experiment)
    checkpoint = tmp_path / 'best.weights.h5'
    model.save_weights(str(checkpoint))
    config_file = tmp_path / 'experiment.json'
    config_file.write_text(json.dumps(experiment.to_dict()))
    monkeypatch.setattr(tf.keras.Model, 'fit', lambda *_a, **_k: pytest.fail('Must never fit'))
    settings = SavedModelEvaluationConfig(
        experiment_file=config_file, model_file=checkpoint, output_dir=tmp_path / 'evaluation',
        subject_ids=('D1/SUBJ_0003',), checkpoint_kind='weights', role='test', evaluate_patches=True,
    )
    result = evaluate_saved_model(settings)
    assert not result.patch_metrics.empty
    assert result.patch_metrics.PatchCount.eq(2).all()
    assert set(result.metrics.Head) == {'regions', 'brain_mask'}
    assert set(result.metrics.EvaluationStage) == {'full_volume'}
    assert (result.output_dir / 'evaluation_metrics.xlsx').is_file()
    sheets = pd.ExcelFile(result.output_dir / 'evaluation_metrics.xlsx').sheet_names
    assert sheets[:2] == ['patch_metrics', 'full_volume_metrics']
    assert {'region_set_summary', 'region_coverage', 'subgroup_summary', 'participant_flow'} <= set(sheets)
    assert (result.output_dir / 'evaluation_patches.json').is_file()
    reopened = load_saved_evaluation(result.output_dir)
    pd.testing.assert_frame_equal(reopened.patch_metrics, result.patch_metrics, check_dtype=False)


def test_postprocessed_brain_metrics_use_brain_gt_and_share_region_inference(experiment, tmp_path, monkeypatch):
    data = prepare_data(experiment)
    scan = data.loader.get_subject(('D1_T1w', 'SUBJ_0000'))
    reference = np.zeros(scan.image.shape, dtype=np.uint8)
    reference[2:6, 2:6, 2:6] = 1
    scan.brain_mask = nib.Nifti1Image(reference, scan.image.affine)
    predicted = reference.copy()
    predicted[2:4, 2:6, 2:6] = 0
    calls = []

    def predict(self, image, with_brain_mask):
        calls.append(image.shape)
        regions = np.zeros(image.shape, dtype=np.int16)
        return PredictionResult(roi=regions, brain_mask=predicted)

    class Model:
        def predict(self, inputs, verbose=0):
            return None

    monkeypatch.setattr('src.evaluation.inference.RandomCroppingPrediction.predict', predict)
    evaluator = ExperimentVolumeEvaluator(Model(), data, ['D1/SUBJ_0000'],
                                         {'evaluation_dir': tmp_path / 'volume', 'save_predictions': False})
    metrics = evaluator.evaluate()
    brain = metrics.loc[metrics.Head == 'brain_mask'].set_index('Metric')
    assert len(calls) == 1
    np.testing.assert_allclose(brain.loc['Dice', 'Value'], 2 / 3)
    np.testing.assert_allclose(brain.loc['Sensitivity', 'Value'], 0.5)
    np.testing.assert_allclose(brain.loc['ReferenceVolume', 'Value'], 64 * 0.1 ** 3, rtol=1e-6)
    assert brain.Label.eq(1).all()
    assert brain.Region.eq('Brain').all()
