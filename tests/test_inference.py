"""Sliding-window inference and repeatable, path-optional evaluation checks."""

import matplotlib
matplotlib.use('Agg')
import numpy as np
import nibabel as nib
import pandas as pd
import pytest

from src.evaluation.inference import RandomCroppingPrediction, FullVolumeEvaluation
from src.utils.loader import Rodent, RodentDatasets


class ToyModel:
    """Patch-position-dependent probabilities expose overlap-weighting changes."""
    def __init__(self, mode='multi'):
        self.mode = mode
        self.calls = 0
    def predict(self, patch, verbose=False):
        self.calls += 1
        spatial = patch.shape[1:4]
        ramp = np.indices(spatial).sum(axis=0) / (sum(spatial) - 3)
        score = np.clip(0.25 * patch[0, ..., 0] + ramp * 0.7, 0, 1)
        roi = np.stack([1 - score, score], axis=-1)[None]
        brain = score[None, ..., None]
        classifier = np.array([[score.mean()]])
        if self.mode == 'single':
            return roi
        if self.mode == 'dict':
            return dict(regions=roi, brain_mask=brain, classifier=classifier)
        return [roi, brain, classifier]


@pytest.mark.parametrize('shape', [(2, 3, 2), (4, 4, 4), (5, 6, 7)])
@pytest.mark.parametrize('mode', ['multi', 'dict', 'single'])
def test_output_heads_padding_and_original_shape(shape, mode):
    model = ToyModel(mode)
    predictor = RandomCroppingPrediction(model, (4, 4, 4), 2, stride=2)
    volume = np.ones(shape)
    result = predictor.random_cropping_inference(volume)
    assert result['roi'].shape == shape
    assert result['roi'].dtype == np.float64
    assert set(np.unique(result['roi'])).issubset({0, 1})
    np.testing.assert_array_equal(volume, 1)


def test_historical_duplicate_edge_patches_and_head_equivalence():
    volume = np.ones((4, 4, 4))
    results = []
    for mode in ('multi', 'dict'):
        model = ToyModel(mode)
        predictor = RandomCroppingPrediction(model, (4, 4, 4), 2, stride=2)
        results.append(predictor.random_cropping_inference(volume, True, True))
        assert model.calls == 8  # Existing 2 x 2 x 2 repeated-edge schedule.
    for head in ('roi', 'brain_mask', 'classifier'):
        np.testing.assert_array_equal(results[0][head], results[1][head])


def test_classifier_uses_patch_votes_and_strict_tie_rule():
    predictor = RandomCroppingPrediction(None, (4, 4, 4), 2, stride=2)
    votes, counts = predictor.initialize_classifier_arrays()
    for probability in (0.51, 0.51, 0.0):
        predictor.update_classifier_arrays(votes, counts, np.array([probability]))
    assert predictor.calculate_final_classifier_output(votes, counts) is True
    predictor.update_classifier_arrays(votes, counts, np.array([0.5]))
    assert predictor.calculate_final_classifier_output(votes, counts) is False


@pytest.mark.parametrize('stride', [0, -1, 5, 1.5])
def test_invalid_or_gapped_stride_fails_early(stride):
    with pytest.raises(ValueError, match='stride'):
        RandomCroppingPrediction(None, (4, 4, 4), 2, stride=stride)


def test_missing_requested_head_has_clear_error():
    predictor = RandomCroppingPrediction(ToyModel('single'), (4, 4, 4), 2, stride=2)
    with pytest.raises(ValueError, match='brain_mask'):
        predictor.random_cropping_inference(np.ones((4, 4, 4)), with_brain_mask=True)


@pytest.fixture
def evaluation_data(tmp_path):
    loader = RodentDatasets()
    for number in range(2):
        image = nib.Nifti1Image(np.ones((4, 4, 4)), np.eye(4))
        image_path = tmp_path / f'{number}.nii.gz'
        nib.save(image, image_path)
        scan_id = f'SUBJ_000{number}'
        record = Rodent('T1w', scan_id, image_path, image_path, image_path,
                        subject_id=scan_id, metadata={'dataset': 'D1', 'original_id': str(number)})
        loader._register([('D1_T1w', record)])
    return loader


class IdentityPreprocessor:
    def preprocess(self, image, masks):
        return image, masks


def make_evaluator(loader, path=None, callback=None):
    config = dict(input_shape=(4, 4, 4), dataset_path='unused', num_classes=2)
    return FullVolumeEvaluation(ToyModel(), loader.get_subjects_list(), config, loader,
                                [2, 4], IdentityPreprocessor(), roi_postprocess_callback=callback,
                                path=path, postprocessed_num_classes=2)


def test_evaluation_without_callbacks_or_output_path_is_repeatable(evaluation_data):
    evaluator = make_evaluator(evaluation_data)
    first = evaluator.evaluate(with_brain_mask=False)
    second = evaluator.evaluate(with_brain_mask=False)
    pd.testing.assert_frame_equal(first, second)
    assert evaluator.num_classes == 2
    assert set(first['Subject']) == {'SUBJ_0000', 'SUBJ_0001'}
    assert set(first['Modality']) == {'D1_T1w'}
    assert len(first) == 24  # Two scans, two strides, six metrics.
    assert 'Mean' in first and 'Class: 1' in first


def test_postprocessing_does_not_change_model_channel_count(evaluation_data):
    evaluator = make_evaluator(evaluation_data, callback=lambda mask: mask)
    actual = evaluator.evaluate(with_brain_mask=False)
    assert evaluator.num_classes == 2
    assert len(actual) == 24


def test_brain_evaluation_then_roi_evaluation_keeps_class_counts(evaluation_data):
    evaluator = make_evaluator(evaluation_data)
    brain = evaluator.evaluate(with_brain_mask=True, evaluate_brain_mask=True)
    assert 'Class: 1' not in brain
    roi = evaluator.evaluate(with_brain_mask=False)
    assert 'Class: 1' in roi
    assert evaluator.num_classes == 2


def test_reports_do_not_mutate_input_tables(evaluation_data, tmp_path, monkeypatch):
    import matplotlib.pyplot as plt
    monkeypatch.setattr(plt, 'show', lambda: None)
    evaluator = make_evaluator(evaluation_data, path=tmp_path / 'new_reports')
    result = evaluator.evaluate(with_brain_mask=False)
    assert len(result) == 24
    assert (tmp_path / 'new_reports' / 'final_metrics.xlsx').is_file()
    assert (tmp_path / 'new_reports' / 'evaluation.log').is_file()
    assert evaluator.logger.log.closed
    before = result.copy(deep=True)
    figure, axis = plt.subplots()
    evaluator.plot_results(result, 'test', axis)
    pd.testing.assert_frame_equal(result, before)
    plt.close(figure)
    assert not plt.get_fignums()


def test_singleton_spatial_axis_is_preserved_in_metric_encoding(evaluation_data, monkeypatch):
    evaluator = make_evaluator(evaluation_data)
    subject = evaluation_data.get_subject(evaluation_data.get_subjects_list()[0])
    singleton_image = nib.Nifti1Image(np.ones((4, 4, 1)), np.eye(4))
    monkeypatch.setattr(evaluator, 'preprocess_data', lambda image, gt: (singleton_image, singleton_image))
    captured = []
    original = evaluator.compute_metrics_per_class
    def inspect_encoding(reference, prediction, subject_id):
        captured.append(reference.shape)
        assert reference.shape == prediction.shape == (1, 4, 4, 1, 2)
        return original(reference, prediction, subject_id)
    monkeypatch.setattr(evaluator, 'compute_metrics_per_class', inspect_encoding)
    predictor = RandomCroppingPrediction(ToyModel(), (4, 4, 4), 2, stride=2)
    evaluator.with_brain_mask = False
    evaluator.compute_metrics_for_subject(subject, predictor)
    assert len(captured) == 1


def test_real_keras_single_roi_head():
    import tensorflow as tf
    inputs = tf.keras.layers.Input((4, 4, 4, 1))
    probabilities = tf.keras.layers.Activation('sigmoid')(inputs)
    model = tf.keras.Model(inputs, probabilities)
    predictor = RandomCroppingPrediction(model, (4, 4, 4), 1, stride=4)
    result = predictor.random_cropping_inference(np.zeros((4, 4, 4)))
    np.testing.assert_array_equal(result['roi'], 1)
