"""Physical full-volume metrics and explicit empty/annotation policies."""

import numpy as np
import pytest

from src.evaluation.metrics import binary_segmentation_metrics, segmentation_metrics_table
from src.evaluation.k_fold import SegmentationEvaluator, SegmentationVolume


def test_anisotropic_single_voxel_distance_and_volume():
    reference = np.zeros((4, 4, 4), dtype=np.uint8)
    prediction = reference.copy()
    reference[1, 1, 1] = 1
    prediction[2, 1, 1] = 1
    result = binary_segmentation_metrics(reference, prediction, (2, 3, 4))
    assert result['Dice'] == result['IoU'] == 0
    assert result['HD95'] == result['ASSD'] == 2
    assert result['ReferenceVolume'] == result['PredictionVolume'] == 24
    assert result['AVE'] == result['RVE'] == 0
    assert result['Sensitivity'] == result['Precision'] == 0


def test_overlap_and_signed_volume_error():
    reference = np.zeros((4, 4, 4), dtype=np.uint8)
    reference[0, 0, 0] = 1
    prediction = reference.copy()
    prediction[1, 0, 0] = 1
    result = binary_segmentation_metrics(reference, prediction, (2, 1, 1))
    assert result['Dice'] == pytest.approx(2 / 3)
    assert result['IoU'] == 0.5
    assert result['Sensitivity'] == 1
    assert result['Precision'] == 0.5
    assert result['AVE'] == 2
    assert result['RVE'] == 100
    assert result['ASSD'] == 0.5
    assert result['HD95'] == pytest.approx(1.9)


@pytest.mark.parametrize('reference_present,prediction_present', [(False, False), (True, False), (False, True)])
def test_empty_surface_distances_are_never_zero(reference_present, prediction_present):
    reference = np.zeros((2, 2, 2), dtype=np.uint8)
    prediction = reference.copy()
    reference[0, 0, 0] = int(reference_present)
    prediction[0, 0, 0] = int(prediction_present)
    result = binary_segmentation_metrics(reference, prediction, (1, 1, 1))
    assert np.isnan(result['HD95']) and np.isnan(result['ASSD'])
    if not reference_present:
        assert np.isnan(result['RVE']) and np.isnan(result['Sensitivity'])
    if not prediction_present:
        assert np.isnan(result['Precision'])
    if not reference_present and not prediction_present:
        assert result['Dice'] == 1
    else:
        assert result['Dice'] == 0


def test_full_mask_has_surface_on_image_boundary():
    mask = np.ones((3, 3, 3), dtype=np.uint8)
    result = binary_segmentation_metrics(mask, mask, (1, 1, 1))
    assert result['Dice'] == 1
    assert result['HD95'] == result['ASSD'] == 0


@pytest.mark.parametrize('spacing', [(0, 1, 1), (np.nan, 1, 1), (1, 1)])
def test_invalid_geometry_is_rejected(spacing):
    mask = np.zeros((2, 2, 2))
    with pytest.raises(ValueError, match='spacing'):
        binary_segmentation_metrics(mask, mask, spacing)


def test_unavailable_regions_are_not_scored_as_absent():
    mask = np.zeros((2, 2, 2), dtype=np.int16)
    rows = segmentation_metrics_table(mask, mask, (1, 1, 1), {1: 'lesion', 3: 'ventricle'},
                                      subject='D1/S1', scan='S1', dataset='D1', available_labels=[1])
    unavailable = rows[rows['Label'] == 3]
    assert unavailable['Value'].isna().all()
    assert set(unavailable['Status']) == {'unavailable'}
    available = rows[rows['Label'] == 1]
    assert set(available['Status']) == {'both_empty'}
    assert available.loc[available['Metric'] == 'Dice', 'Value'].item() == 1
    assert set(rows['Subject']) == {'D1/S1'}
    assert rows.loc[rows['Metric'] == 'HD95', 'Unit'].unique().tolist() == ['mm']


def test_segmentation_evaluator_preserves_timepoints():
    mask = np.ones((2, 2, 2), dtype=np.int16)
    def predict(model, subject):
        for scan in ('day7', 'day14'):
            yield SegmentationVolume(reference=mask, prediction=mask, spacing_mm=(1, 1, 1), scan=scan, dataset='D2')
    evaluator = SegmentationEvaluator(None, ['animal'], {}, predict_subject=predict, regions={1: 'lesion'})
    table = evaluator.evaluate()
    assert len(table) == 20
    assert set(table['Scan']) == {'day7', 'day14'}
    assert set(table['Subject']) == {'animal'}


def test_segmentation_evaluator_rejects_missing_subject_predictions():
    evaluator = SegmentationEvaluator(None, ['animal'], {}, predict_subject=lambda model, subject: [], regions={1: 'lesion'})
    with pytest.raises(ValueError, match='No full-volume'):
        evaluator.evaluate()


def test_physical_metrics_use_full_sheared_affine_without_changing_orthogonal_results():
    a = np.zeros((5, 5, 5), dtype=np.uint8)
    b = a.copy()
    a[1, 1, 1] = 1
    b[2, 2, 1] = 1
    affine = np.array([[1, 0.5, 0, 0], [0, 1, 0, 0], [0, 0, 2, 0], [0, 0, 0, 1]], dtype=float)
    spacing = np.linalg.norm(affine[:3, :3], axis=0)
    result = binary_segmentation_metrics(a, b, spacing, affine_mm=affine)
    assert result["HD95"] == pytest.approx(np.sqrt(1.5 ** 2 + 1))
    assert result["ASSD"] == pytest.approx(result["HD95"])
    assert result["ReferenceVolume"] == pytest.approx(2)
    orthogonal = np.diag([1, 2, 3, 1])
    old = binary_segmentation_metrics(a, b, (1, 2, 3))
    new = binary_segmentation_metrics(a, b, (1, 2, 3), affine_mm=orthogonal)
    for key in old:
        assert new[key] == pytest.approx(old[key], nan_ok=True)
