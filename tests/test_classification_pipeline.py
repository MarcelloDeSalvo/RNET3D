"""Classifier/lesion independence, animal counts and auditable uncertainty."""

import json
from dataclasses import replace

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

from test_experiments import experiment
from src.evaluation.classification import classification_report, healthy_lesion_report
from src.evaluation.inference import ExperimentVolumeEvaluator, PredictionResult, RandomCroppingPrediction
from src.experiments.data import prepare_data
from src.experiments.evaluation import export_evaluation_tables


def decisions(animals=16, *, role='test', fold=0, seed=42):
    """Two positive scans per independent TBI animal for the exact-CI example."""
    return pd.DataFrame([
        {'Variant': '3drnet', 'Seed': seed, 'Fold': fold, 'DatasetRole': role,
         'Subject': f'animal-{animal}', 'Scan': f'scan-{animal}-{scan}', 'Dataset': 'D2',
         'Reference': 1, 'Prediction': 1}
        for animal in range(animals) for scan in range(2)
    ])


def test_classifier_intervals_count_animals_not_longitudinal_scans():
    report = classification_report(decisions())
    summary = report.summary.iloc[0]
    assert summary.Animals == 16 and summary.Scans == 32
    assert summary.accuracy == 1
    np.testing.assert_allclose(summary.accuracy_ci_lower, 0.7940928, rtol=1e-6)
    assert np.isnan(summary.specificity)
    assert report.confusion.Count.sum() == 16


def test_scan_vote_ties_remain_sham_and_conflicting_reference_is_rejected():
    scans = decisions(1)
    scans.loc[0, 'Prediction'] = 0
    report = classification_report(scans)
    assert report.animals.Prediction.item() == 0
    scans.loc[0, 'Reference'] = 0
    with pytest.raises(ValueError, match='conflicting SHAM/TBI'):
        classification_report(scans)


def test_holdout_pooling_is_disjoint_and_repeated_model_estimates_stay_separate():
    first = decisions(2, role='holdout', fold=0)
    second = decisions(2, role='holdout', fold=1)
    second['Subject'] = 'other-' + second['Subject']
    report = classification_report(pd.concat([first, second], ignore_index=True))
    pooled = report.summary.loc[report.summary.Scope == 'pooled_holdout']
    assert pooled.Animals.item() == 4
    repeated = pd.concat([decisions(2, fold=0), decisions(2, fold=1)], ignore_index=True)
    report = classification_report(repeated)
    assert report.summary.Animals.tolist() == [2, 2]
    assert report.summary.Scope.eq('fold').all()
    duplicate_holdout = pd.concat([first, decisions(2, role='holdout', fold=1)], ignore_index=True)
    with pytest.raises(ValueError, match='exactly one fold'):
        classification_report(duplicate_holdout)
    with pytest.raises(ValueError, match='Duplicate classifier'):
        classification_report(pd.concat([first, first], ignore_index=True))


@pytest.mark.parametrize('probability,expected', [(0.5, False), (0.6, True)])
def test_patch_vote_fraction_is_preserved_without_extra_prediction_calls(probability, expected):
    class Model:
        calls = 0

        def predict(self, inputs, verbose=0):
            self.calls += 1
            roi = np.zeros((1, 2, 2, 2, 2), dtype=np.float32)
            roi[..., 0] = 1
            brain = np.ones((1, 2, 2, 2, 1), dtype=np.float32)
            classifier = np.array([[probability]], dtype=np.float32)
            return [roi, brain, classifier]

    model = Model()
    predictor = RandomCroppingPrediction(model, patch_size=(2, 2, 2), num_classes=2, stride=2)
    result = predictor.predict(np.zeros((2, 2, 2)), with_brain_mask=True, with_classifier=True)
    assert result.classifier is expected
    assert result.classifier_positive_fraction == int(expected)
    assert result.classifier_patch_count == model.calls


@pytest.mark.parametrize('probability', [np.nan, -0.1, 1.1])
def test_invalid_classifier_outputs_cannot_be_reported_as_negative_votes(probability):
    class Model:
        def predict(self, inputs, verbose=0):
            roi = np.zeros((1, 2, 2, 2, 2), dtype=np.float32)
            roi[..., 0] = 1
            return [roi, None, np.array([[probability]])]

    predictor = RandomCroppingPrediction(Model(), patch_size=(2, 2, 2), num_classes=2, stride=2)
    with pytest.raises(ValueError, match='Classifier probabilities'):
        predictor.predict(np.zeros((2, 2, 2)), with_classifier=True)


def test_correct_sham_classifier_does_not_remove_lesion_false_positives(experiment, tmp_path, monkeypatch):
    folder = experiment.data.root / 'D1' / 'SUBJ_0000'
    metadata_file = folder / 'metadata.json'
    metadata = json.loads(metadata_file.read_text())
    metadata.update(condition='SHAM', has_lesion_annotation=False)
    metadata_file.write_text(json.dumps(metadata))
    mask_file = folder / 'Labels.nii.gz'
    image = nib.load(mask_file)
    truth = image.get_fdata()
    truth[truth == 1] = 0
    nib.save(nib.Nifti1Image(truth, image.affine), mask_file)
    config = replace(experiment, model=replace(experiment.model, architecture='3drnet', classifier=True))
    data = prepare_data(config)
    calls = []

    def predict(self, image, with_brain_mask, with_classifier):
        calls.append(with_classifier)
        labels = np.zeros(image.shape, dtype=np.int16)
        labels[2:4, 2:4, 2:4] = 1
        return PredictionResult(roi=labels, brain_mask=np.ones(image.shape), classifier=False,
                                classifier_positive_fraction=0.25, classifier_patch_count=4)

    class Model:
        def predict(self, inputs, verbose=0):
            return None

    monkeypatch.setattr('src.evaluation.inference.RandomCroppingPrediction.predict', predict)
    evaluator = ExperimentVolumeEvaluator(Model(), data, ['D1/SUBJ_0000'],
                                         {'evaluation_dir': tmp_path / 'volume', 'save_predictions': False})
    metrics = evaluator.evaluate().assign(Variant='3drnet', Seed=42, Fold=0, DatasetRole='test')
    scans = evaluator.classifier_predictions.assign(Variant='3drnet', Seed=42, Fold=0, DatasetRole='test')
    assert calls == [True]
    assert scans.Reference.item() == scans.Prediction.item() == 0
    assert classification_report(scans).summary.accuracy.item() == 1
    animals, summary = healthy_lesion_report(metrics)
    np.testing.assert_allclose(animals.MeanLesionVolumeMM3.item(), 0.008, rtol=1e-6)
    assert summary.FalsePositiveAnimals.item() == 1
    assert summary.FalsePositiveAnimalRate.item() == 1
    output = tmp_path / 'export'
    export_evaluation_tables(pd.DataFrame(), metrics, output, classifier_predictions=scans)
    assert (output / 'classifier_summary.csv').is_file()
    assert (output / 'sham_lesion_summary.csv').is_file()
    assert 'classifier_confusion' in pd.ExcelFile(output / 'evaluation_metrics.xlsx').sheet_names
    assert healthy_lesion_report(metrics.drop(columns='Condition'))[0].empty


def test_healthy_diagnostics_never_infer_health_from_an_empty_tbi_lesion():
    metrics = decisions(1).rename(columns={'Prediction': 'Value'})
    metrics = metrics.assign(Condition='TBI', Region='Lesion', Metric='PredictionVolume', Status='reference_empty')
    assert healthy_lesion_report(metrics)[0].empty
    metrics['Condition'] = 'SHAM'
    metrics['Status'] = 'unavailable'
    assert healthy_lesion_report(metrics)[0].empty
