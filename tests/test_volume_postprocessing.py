"""Verify original IDs, historical operations, source GT and notebook integration."""

import json
from dataclasses import replace
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from test_experiments import experiment
from src.experiments.config import DatasetSpec, InferenceConfig, LabelConfig, PostprocessingConfig
from src.experiments.data import prepare_data
from src.evaluation.inference import (
    ExperimentVolumeEvaluator, PredictionResult, load_prediction_comparison,
    best_checkpoint_settings, evaluate_saved_model, prediction_comparison_figure, saved_prediction_manifest,
)
from src.evaluation.postprocessing import FullVolumePostprocessor, OutputLabels


def merged_phantom():
    """Two lateral ventricles and a midline with a lesion on the low-x side."""
    raw = np.zeros((16, 16, 16), dtype=np.int16)
    raw[7, 4:13, 4:15] = 3
    raw[3:5, 6:10, 6:10] = 2
    raw[10:12, 6:10, 6:10] = 2
    raw[1:4, 1:4, 1:4] = 1
    truth = raw.copy()
    truth[raw == 3] = 21
    truth[raw == 2] = 3
    truth[3:5, 6:10, 6:10] = 13
    return raw, truth


def test_split_remap_and_brain_morphology_match_historical_operations():
    raw, expected = merged_phantom()
    original = raw.copy()
    brain = np.zeros(raw.shape, dtype=np.int16)
    brain[2:14, 2:14, 2:14] = 1
    brain[5, 5, 5] = 0
    brain[0, 0, 0] = 1
    settings = PostprocessingConfig(
        split_ventricles=True, clean_brain_mask=True,
        holes_max_area=2, object_min_area=8,
    )
    processor = FullVolumePostprocessor(LabelConfig(), settings, original_label_ids=True)
    result = processor.process(raw, brain)
    np.testing.assert_array_equal(result.regions, expected)
    np.testing.assert_array_equal(raw, original)
    assert result.brain_mask[5, 5, 5] == 1
    assert result.brain_mask[0, 0, 0] == 0
    assert brain[5, 5, 5] == 0 and brain[0, 0, 0] == 1
    assert set(np.unique(result.regions)) == {0, 1, 3, 13, 21}


def test_mousefull_remap_keeps_predicted_sides_and_never_masks_rois_with_brain():
    mapping = {0: 0, 1: 1, 2: 2, 3: 3, 5: 4, 6: 5, 12: 6, 13: 7, 15: 8, 16: 9, 21: 10}
    names = tuple(str(source) for source in mapping)
    labels = LabelConfig(mapping=mapping, names=names)
    raw = np.arange(11, dtype=np.int16).reshape((11, 1, 1))
    brain = np.ones(raw.shape, dtype=np.int16)
    processor = FullVolumePostprocessor(
        labels, PostprocessingConfig(clean_brain_mask=True), original_label_ids=True,
    )
    result = processor.process(raw, brain)
    np.testing.assert_array_equal(result.regions.ravel(), list(mapping))
    assert not result.brain_mask.any()  # Smaller than the explicit historical threshold.
    assert result.regions[7, 0, 0] == 13
    with pytest.raises(ValueError, match="four-class"):
        OutputLabels.resolve(labels, split_ventricles=True, original_label_ids=True)


def test_merged_labels_cannot_be_inverted_without_separation():
    with pytest.raises(ValueError, match="separating merged"):
        OutputLabels.resolve(LabelConfig(), split_ventricles=False, original_label_ids=True)

    labels = LabelConfig(mapping={0: 0, 1: 1, 2: 0, 13: 2}, names=('Background', 'Lesion', 'Ipsi ventricle'))
    output = OutputLabels.resolve(labels, split_ventricles=False, original_label_ids=True)
    assert output.prediction_mapping == {0: 0, 1: 1, 2: 13}
    assert output.ground_truth_mapping[2] == 0  # Explicit extra-anatomy projection stays background.


def test_missing_midline_is_reported_instead_of_silently_assigning_sides():
    raw, _ = merged_phantom()
    raw[raw == 3] = 0
    processor = FullVolumePostprocessor(
        LabelConfig(), PostprocessingConfig(split_ventricles=True), original_label_ids=True,
    )
    with pytest.raises(ValueError, match="three"):
        processor.process(raw, None)


def test_config_roundtrip_and_older_settings(experiment):
    config = replace(experiment, inference=InferenceConfig(
        original_label_ids=True,
        postprocessing=PostprocessingConfig(clean_brain_mask=True, holes_max_area=4),
    ))
    restored = type(config).from_dict(config.to_dict())
    assert restored == config
    old = config.to_dict()
    del old['inference']['postprocessing']
    del old['inference']['original_label_ids']
    restored = type(config).from_dict(old)
    assert restored.inference.postprocessing == PostprocessingConfig()


def test_evaluator_preserves_true_gt_sides_metrics_geometry_and_viewer(experiment, tmp_path, monkeypatch):
    """One scan, predicted once: correct source IDs in metrics, files and native controls."""
    raw, truth = merged_phantom()
    datasets = (DatasetSpec(available_labels=(0, 1, 3, 13, 21)),)
    settings = PostprocessingConfig(split_ventricles=True, clean_brain_mask=True, object_min_area=8)
    inference = InferenceConfig(enabled=True, stride=16, original_label_ids=True, postprocessing=settings)
    config = replace(experiment, labels=LabelConfig(), inference=inference,
                     data=replace(experiment.data, datasets=datasets))
    data = prepare_data(config)
    key = ('D1_T1w', 'SUBJ_0000')
    scan = data.loader.get_subject(key)
    scan.labels = nib.Nifti1Image(truth, scan.image.affine)
    calls = []

    def predict(self, image, with_brain_mask):
        calls.append(image.shape)
        return PredictionResult(roi=raw, brain_mask=np.ones(raw.shape, dtype=np.int16))

    monkeypatch.setattr('src.evaluation.inference.RandomCroppingPrediction.predict', predict)

    class WarmupModel:
        def predict(self, values, verbose=0):
            return None

    folder = tmp_path / 'run' / 'fold_0' / 'evaluations' / 'test' / 'regions'
    evaluator = ExperimentVolumeEvaluator(WarmupModel(), data, ['D1/SUBJ_0000'],
                                         {'evaluation_dir': folder, 'save_predictions': True})
    table = evaluator.evaluate()
    assert len(calls) == 1
    assert set(table['Label']) == {1, 3, 13, 21}
    assert np.all(table.loc[table['Metric'] == 'Dice', 'Value'] == 1)
    path = folder / 'predictions' / 'D1_T1w' / 'SUBJ_0000' / 'regions.nii.gz'
    saved = nib.load(path)
    np.testing.assert_array_equal(saved.get_fdata(), truth)
    np.testing.assert_allclose(saved.affine, scan.image.affine)
    assert saved.get_data_dtype() == np.dtype('int16')
    comparison = load_prediction_comparison(data, key, path)
    np.testing.assert_array_equal(comparison.ground_truth, comparison.prediction)
    manifest = saved_prediction_manifest(data, tmp_path / 'run')
    figure = prediction_comparison_figure(data, manifest, region=13, max_slices=2)
    assert figure.data[-1].name == '13: Ipsi ventricle'

    # Annotated sides can differ from predicted lesion-side assignment: never postprocess GT.
    scan.labels = nib.Nifti1Image(np.where(truth == 13, 3, np.where(truth == 3, 13, truth)), scan.image.affine)
    comparison = load_prediction_comparison(data, key, path)
    assert not np.array_equal(comparison.ground_truth, comparison.prediction)

    # Availability is independent for the two source sides after separating a merged class.
    partial_spec = (DatasetSpec(available_labels=(0, 1, 3, 21)),)
    partial_config = replace(config, data=replace(config.data, datasets=partial_spec))
    partial_data = prepare_data(partial_config)
    partial_scan = partial_data.loader.get_subject(key)
    partial_scan.labels = nib.Nifti1Image(truth, partial_scan.image.affine)
    partial_evaluator = ExperimentVolumeEvaluator(
        WarmupModel(), partial_data, ['D1/SUBJ_0000'],
        {'evaluation_dir': tmp_path / 'partial', 'save_predictions': False},
    )
    partial_table = partial_evaluator.evaluate()
    assert set(partial_table.loc[partial_table['Label'] == 13, 'Status']) == {'unavailable'}
    assert 'unavailable' not in set(partial_table.loc[partial_table['Label'] == 3, 'Status'])


def test_every_training_notebook_has_explicit_full_volume_and_checkpoint_settings():
    root = Path(__file__).resolve().parents[1] / 'notebook'
    for path in root.glob('0*.ipynb'):
        notebook = json.loads(path.read_text(encoding='utf-8'))
        sources = [''.join(cell['source']) for cell in notebook['cells'] if cell['cell_type'] == 'code']
        for index, source in enumerate(sources):
            compile(source, f'{path.name}:{index}', 'exec')
        if path.name.startswith('09_'):
            assert any('ORIGINAL_LABEL_IDS: bool = True' in source for source in sources)
            assert any('postprocessing=postprocessing_settings' in source for source in sources)
            assert any('USE_VENTRICLE_CENTROIDS: bool = True' in source for source in sources)
            continue
        config = next(source for source in sources if 'inference_settings = InferenceConfig(' in source)
        checkpoint = next(source for source in sources if 'RUN_CHECKPOINT_EVALUATION: bool' in source)
        assert 'save_predictions=True' in config and 'original_label_ids=True' in config
        assert 'clean_brain_mask=True' in config
        assert 'original_label_ids=True, postprocessing=checkpoint_postprocessing' in checkpoint
        split = not path.name.startswith(('05_', '06_', '07f_'))
        assert f'split_ventricles={split}' in config
        assert f'split_ventricles={split}' in checkpoint
        if split:
            assert 'use_centroids=True' in config
            assert 'use_centroids=True' in checkpoint
        assert 'regions_weight=1.0, brain_weight=1.0, classifier_weight=1.0' in config
        if path.name.startswith('01_'):
            assert 'RESUME_RUN: Path | None = None' in config
            assert 'SPLIT_FILE: Path | None = None' in config
            assert any('run_training(prepared, split, resume_run=RESUME_RUN, callbacks=callbacks)' in source for source in sources)


def test_old_checkpoint_evaluation_can_override_export_and_postprocessing(experiment, tmp_path):
    """An older run exports sparse source IDs without changing checkpoint artifacts."""
    from test_checkpoint_evaluation import artifact_hashes, write_run
    from src.experiments.training import build_model

    labels = LabelConfig(mapping={0: 0, 1: 1, 13: 2}, names=('Background', 'Lesion', 'Ipsi ventricle'))
    datasets = (DatasetSpec(available_labels=(0, 1, 13)),)
    config = replace(experiment, labels=labels, data=replace(experiment.data, datasets=datasets))
    source = build_model(config)
    head = source.get_layer('regions_logits').input._keras_history.layer
    weights = [np.zeros_like(value) for value in head.get_weights()]
    weights[-1] = np.asarray([-20, -20, 20], dtype=weights[-1].dtype)
    head.set_weights(weights)
    run = tmp_path / 'old_run'
    fold = write_run(config, run)
    source.save_weights(str(fold / 'best.weights.h5'))
    before = artifact_hashes(run)
    postprocessing = PostprocessingConfig(clean_brain_mask=True)
    settings = best_checkpoint_settings(run, original_label_ids=True, postprocessing=postprocessing)
    result = evaluate_saved_model(settings)
    assert result.data.config.inference.postprocessing == postprocessing
    assert result.data.config.inference.original_label_ids is True
    assert artifact_hashes(run) == before
    manifest = saved_prediction_manifest(result.data, result.output_dir)
    for path in manifest['PredictionPath']:
        assert set(np.unique(nib.load(path).get_fdata())) == {13}
        comparison = load_prediction_comparison(result.data, ('D1_T1w', Path(path).parent.name), Path(path))
        assert comparison.label_names[13] == 'Ipsi ventricle'
    metadata = json.loads((result.output_dir / 'evaluation.json').read_text(encoding='utf-8'))
    assert metadata['source_experiment']['inference']['original_label_ids'] is False
    assert metadata['evaluation_experiment']['inference']['original_label_ids'] is True
