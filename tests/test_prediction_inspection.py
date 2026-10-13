"""Check scan discovery, geometry, partial annotations and native viewer controls."""

import base64
from io import BytesIO
from dataclasses import replace
from urllib.parse import quote

import nibabel as nib
import numpy as np
from PIL import Image
import pytest

from test_experiments import experiment
from src.experiments.data import prepare_data
from src.evaluation.inference import (
    load_prediction_comparison, prediction_comparison_figure, saved_prediction_manifest,
)


def save_prediction(data, run, key, *, role="test", fold=0, values=None):
    """Save output IDs on an input scan's grid, as the evaluation exporter does."""
    scan = data.loader.get_subject(key)
    if values is None:
        source = scan.labels.get_fdata()
        values = np.zeros(scan.image.shape, dtype=np.int16)
        for source_id, channel in data.config.labels.mapping.items():
            values[source == source_id] = channel
    group, scan_id = key
    folder = run / f"fold_{fold}" / "evaluations" / role / "regions" / "predictions"
    folder = folder / quote(group, safe="") / quote(scan_id, safe="")
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "regions.nii.gz"
    nib.save(nib.Nifti1Image(values, scan.image.affine), path)
    return path


def test_manifest_preserves_roles_folds_and_longitudinal_identity(experiment, tmp_path):
    data = prepare_data(experiment)
    run = tmp_path / "run"
    keys = data.subjects.scan_keys(["D1/SUBJ_0003"])
    for key in keys:
        save_prediction(data, run, key, fold=1)
    save_prediction(data, run, ("D1_T1w", "SUBJ_0000"), role="validation")
    table = saved_prediction_manifest(data, run)
    assert set(table["Subject"]) == {"D1/SUBJ_0003"}
    assert set(table["Scan"]) == {"SUBJ_0003_7days", "SUBJ_0003_14days"}
    assert set(table["Fold"]) == {1}
    assert set(table["DatasetRole"]) == {"test"}
    assert table.index.name == "ViewIndex"
    assert len(saved_prediction_manifest(data, run, role="validation")) == 1
    assert saved_prediction_manifest(data, run, role="holdout").empty
    with pytest.raises(ValueError, match="Select"):
        saved_prediction_manifest(data, run, role="train")


def test_comparison_maps_gt_and_reorients_all_inputs(experiment, tmp_path):
    data = prepare_data(experiment)
    key = ("D1_T1w", "SUBJ_0000")
    scan = data.loader.get_subject(key)
    # A permuted reference requires every input to follow the same axis change.
    orientation = np.asarray([[1, -1], [2, 1], [0, 1]])
    reference = scan.image.as_reoriented(orientation)
    reference_path = tmp_path / "reference.nii.gz"
    nib.save(reference, reference_path)
    preprocessing = replace(experiment.preprocessing, reference=reference_path)
    data.config = replace(experiment, preprocessing=preprocessing)
    path = save_prediction(data, tmp_path / "run", key)
    comparison = load_prediction_comparison(data, key, path)
    assert comparison.orientation == nib.aff2axcodes(reference.affine)
    np.testing.assert_array_equal(comparison.ground_truth, comparison.prediction)
    expected = scan.image.as_reoriented(orientation).get_fdata()
    np.testing.assert_allclose(comparison.image, expected)
    assert comparison.unavailable_names == ()


def test_missing_gt_is_hidden_while_predictions_remain_visible(experiment, tmp_path):
    data = prepare_data(experiment)
    key = ("D1_T1w", "SUBJ_0001")
    scan = data.loader.get_subject(key)
    source = scan.labels.get_fdata()
    source[0, 0, 0] = 2  # Present in a file, but not declared annotated.
    source[0, 0, 1] = 7  # Outside the task vocabulary: maps to background.
    scan.labels = nib.Nifti1Image(source, scan.image.affine)
    path = save_prediction(data, tmp_path / "run", key)
    comparison = load_prediction_comparison(data, key, path)
    assert comparison.unavailable_names == ("CC",)
    assert comparison.ground_truth[0, 0, 0] == -1
    assert comparison.prediction[0, 0, 0] == 2
    assert comparison.ground_truth[0, 0, 1] == 0
    np.testing.assert_array_equal(scan.labels.get_fdata(), source)


def test_comparison_rejects_misaligned_or_invalid_saved_masks(experiment, tmp_path):
    data = prepare_data(experiment)
    key = ("D1_T1w", "SUBJ_0000")
    path = save_prediction(data, tmp_path / "run", key)
    prediction = nib.load(path)
    affine = prediction.affine.copy()
    affine[0, 3] += 1
    nib.save(nib.Nifti1Image(prediction.get_fdata(), affine), path)
    with pytest.raises(ValueError, match="physical grid"):
        load_prediction_comparison(data, key, path)

    values = np.full((16, 16, 16), 7, dtype=np.int16)
    path = save_prediction(data, tmp_path / "run", key, values=values)
    with pytest.raises(ValueError, match="output class IDs"):
        load_prediction_comparison(data, key, path)


def test_native_controls_switch_scans_planes_and_slices(experiment, tmp_path):
    data = prepare_data(experiment)
    run = tmp_path / "run"
    for key in (("D1_T1w", "SUBJ_0000"), ("D1_T1w", "SUBJ_0001")):
        save_prediction(data, run, key)
    table = saved_prediction_manifest(data, run)
    figure = prediction_comparison_figure(data, table, max_slices=4, max_pixels=8)
    assert len(figure.frames) == 2 * 3 * 4
    assert len(figure.layout.updatemenus[0].buttons) == 6
    assert len(figure.layout.sliders[0].steps) == 4
    assert "unannotated GT regions: none" in figure.layout.title.text
    assert "unannotated GT regions: CC" in figure.layout.updatemenus[0].buttons[3].args[1]["title.text"]
    frame_names = {frame.name for frame in figure.frames}
    for button in figure.layout.updatemenus[0].buttons:
        slider = button.args[1]["sliders"][0]
        assert all(step["args"][0][0] in frame_names for step in slider["steps"])
    assert [trace.name for trace in figure.data[3:]] == ["1: Lesion", "2: CC"]
    for trace in figure.data[:3]:
        encoded = trace.source.split(",", 1)[1]
        image = Image.open(BytesIO(base64.b64decode(encoded)))
        assert image.size == (8, 8)

    detailed = prediction_comparison_figure(data, table, scan_indices=(1,), region=2, max_slices=None)
    assert len(detailed.frames) == 3 * 16
    assert len(detailed.data) == 4
    assert detailed.data[-1].name == "2: CC"
    assert detailed.to_json()  # Renderable by the native notebook renderer.


@pytest.mark.parametrize("settings", [{"scan_indices": (5,)}, {"region": 0}, {"opacity": -1}, {"max_slices": 0}])
def test_invalid_viewer_selections_are_explicit(experiment, tmp_path, settings):
    data = prepare_data(experiment)
    run = tmp_path / "run"
    save_prediction(data, run, ("D1_T1w", "SUBJ_0000"))
    table = saved_prediction_manifest(data, run)
    with pytest.raises(ValueError):
        prediction_comparison_figure(data, table, **settings)
