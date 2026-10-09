"""Small morphology and ventricle-label regression/edge-case checks."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pytest
from skimage import morphology

from src.evaluation.postprocessing import ipsi_contra_division_callback, morphology_refinement_callback


def roi_mask(lesion=True):
    mask = np.zeros((15, 17, 19), dtype=np.uint8)
    mask[7, 4:13, 4:15] = 3
    mask[3:5, 6:10, 6:10] = 2
    mask[10:12, 6:10, 6:10] = 2
    if lesion:
        mask[1:4, 1:4, 1:4] = 1
    return mask


@pytest.mark.parametrize('holes', [False, True])
@pytest.mark.parametrize('objects', [False, True])
def test_morphology_preserves_operation_order_and_input(holes, objects):
    mask = np.zeros((7, 7, 7), dtype=np.float32)
    mask[1:6, 1:6, 1:6] = 1
    mask[3, 3, 3] = 0
    mask[0, 0, 0] = 1
    before = mask.copy()
    expected = mask.astype(bool)
    if holes:
        expected = morphology.remove_small_holes(expected, area_threshold=2)
    if objects:
        expected = morphology.remove_small_objects(expected, min_size=3)
    callback = morphology_refinement_callback(holes, objects, 2, 3)
    actual = callback(mask)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(mask, before)
    if holes or objects:
        assert actual.dtype == np.float64
    else:
        assert actual is mask


@pytest.mark.parametrize('centroids', [False, True])
def test_lesion_side_and_labels_preserved(centroids):
    mask = roi_mask()
    before = mask.copy()
    callback = ipsi_contra_division_callback(use_centroids=centroids)
    actual = callback(mask)
    assert np.all(actual[3:5, 6:10, 6:10] == 3)
    assert np.all(actual[10:12, 6:10, 6:10] == 2)
    assert np.all(actual[mask == 3] == 4)
    assert np.all(actual[mask == 1] == 1)
    np.testing.assert_array_equal(mask, before)
    assert actual.dtype == mask.dtype


def test_left_fallback_is_opposite_right():
    mask = roi_mask(lesion=False)
    right = ipsi_contra_division_callback(default_hemisphere='right')(mask)
    left = ipsi_contra_division_callback(default_hemisphere='left')(mask)
    ventricles = mask == 2
    np.testing.assert_array_equal(right[ventricles] + left[ventricles], 5)
    assert np.all(right[3:5, 6:10, 6:10] == 3)


def test_tiny_lesion_uses_fallback_even_with_centroids():
    mask = roi_mask(lesion=False)
    expected = ipsi_contra_division_callback(default_hemisphere='left')(mask)
    mask[1, 1, 1] = 1
    actual = ipsi_contra_division_callback(default_hemisphere='left', use_centroids=True)(mask)
    np.testing.assert_array_equal(actual[mask == 2], expected[mask == 2])
    assert actual[1, 1, 1] == 1


@pytest.mark.parametrize('count', [0, 1, 2])
def test_insufficient_third_ventricle_gives_clear_error(count):
    mask = np.zeros((5, 5, 5), dtype=np.uint8)
    mask[0, 0, 0] = 2
    mask[2, 2, :count] = 3
    with pytest.raises(ValueError, match='three third-ventricle'):
        ipsi_contra_division_callback()(mask)


def test_no_lateral_ventricles_needs_no_plane():
    mask = np.zeros((3, 3, 3), dtype=np.uint8)
    mask[1, 1, 1] = 3
    actual = ipsi_contra_division_callback()(mask)
    assert actual[1, 1, 1] == 4
    assert mask[1, 1, 1] == 3


def test_plotting_between_ten_and_twenty_thousand_points(monkeypatch):
    mask = np.zeros((5, 110, 110), dtype=np.uint8)
    mask[2, :, :] = 3
    mask[0, 40:50, 40:50] = 2
    monkeypatch.setattr(plt, 'show', lambda: None)
    result = ipsi_contra_division_callback(visualize_pca=True)(mask)
    assert result.shape == mask.shape
    assert not plt.get_fignums()


@pytest.mark.parametrize('kwargs', [{'default_hemisphere': 'bad'}, {'lesion_voxels_threshold': -1}])
def test_invalid_split_settings_fail_early(kwargs):
    with pytest.raises(ValueError):
        ipsi_contra_division_callback(**kwargs)
