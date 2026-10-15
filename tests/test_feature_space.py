"""Scan features, longitudinal aggregation and metadata-free exploratory projections."""

import ast
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest
import tensorflow as tf

from test_experiments import experiment
from test_attention import saved_run
from src.evaluation.feature_space import (
    ScanFeatures, encoder_feature_table, extract_scan_features, feature_space_figure, project_features,
)
from src.preprocessing.preprocessor import Preprocessor


def encoder():
    """Two known channels feeding a decoder, without an auxiliary classifier."""
    inputs = tf.keras.Input((4, 4, 4, 1))
    features = tf.keras.layers.Conv3D(2, 1, use_bias=False, name="bottleneck")(inputs)
    outputs = tf.keras.layers.Conv3DTranspose(1, 1)(features)
    model = tf.keras.Model(inputs, outputs)
    model.get_layer("bottleneck").set_weights([np.array([1, 2], dtype=np.float32).reshape((1, 1, 1, 1, 2))])
    return model


@pytest.fixture
def features():
    """Five scans from four animals, including one repeated animal."""
    manifest = pd.DataFrame({
        "Subject": ["D1/A", "D1/A", "D1/B", "D3/C", "D3/D"],
        "Scan": ["A-1", "A-2", "B", "C", "D"], "Dataset": ["D1", "D1", "D1", "D3", "D3"],
        "Condition": ["TBI", "TBI", "TBI", "SHAM", "SHAM"],
        "Timepoint": ["7d", "14d", "7d", "7d", "14d"], "DatasetRole": ["test"] * 5,
    })
    vectors = np.array([[1, 2, 0], [3, 4, 0], [1, 0, 2], [0, 1, 4], [1, 0, 5]], dtype=np.float32)
    return ScanFeatures(values=vectors, manifest=manifest, layer="bottleneck")


def test_extracts_exact_spatial_channel_means_once_per_scan_without_model_mutation(tmp_path, monkeypatch):
    model = encoder()
    table = encoder_feature_table(model)
    assert table.loc[table.Default, "Layer"].item() == "bottleneck"
    paths = []
    for index in range(3):
        path = tmp_path / f"{index}.nii.gz"
        nib.save(nib.Nifti1Image(np.full((4, 4, 4), index, dtype=np.float32), np.eye(4)), path)
        paths.append(str(path))
    manifest = pd.DataFrame({"Subject": ["D1/A", "D1/B", "D1/C"], "Scan": ["A", "B", "C"],
                             "Dataset": ["D1"] * 3, "ImagePath": paths})
    before = [value.copy() for value in model.get_weights()]
    monkeypatch.setattr(tf.keras.Model, "fit", lambda *args, **kwargs: pytest.fail("No fitting"))
    result = extract_scan_features(model, manifest, Preprocessor(steps=[]))
    np.testing.assert_array_equal(result.values, [[0, 0], [1, 2], [2, 4]])
    pd.testing.assert_frame_equal(result.manifest, manifest)
    for expected, actual in zip(before, model.get_weights()):
        np.testing.assert_array_equal(expected, actual)
    with pytest.raises(ValueError, match="each scan once"):
        extract_scan_features(model, pd.concat([manifest, manifest]), Preprocessor([]))
    missing = manifest.copy()
    missing.loc[0, "Subject"] = " "
    with pytest.raises(ValueError, match="must not be empty"):
        extract_scan_features(model, missing, Preprocessor([]))


def test_animal_averaging_preserves_counts_and_lists_longitudinal_metadata(features):
    result = project_features(features, aggregation="animal", standardize=False)
    assert len(result.points) == 4
    assert result.parameters["Animals"] == 4 and result.parameters["SelectedScans"] == 5
    animal = result.points.loc[result.points.Subject == "D1/A"].iloc[0]
    assert animal.ScanCount == 2 and animal.Timepoint == "14d | 7d"
    averaged = ScanFeatures(values=np.array([[2, 3, 0], [1, 0, 2], [0, 1, 4], [1, 0, 5]]),
                            manifest=result.points.drop(columns=["Component1", "Component2"]), layer=features.layer)
    expected = project_features(averaged, aggregation="scan", standardize=False)
    np.testing.assert_allclose(result.points[["Component1", "Component2"]], expected.points[["Component1", "Component2"]])
    assert sum(result.explained_variance) <= 1.000001


def test_metadata_colors_never_enter_projection_and_pca_is_repeatable(features):
    first = project_features(features, aggregation="scan")
    changed = features.manifest.copy()
    changed["Condition"] = ["other", "SHAM", "new", "TBI", "TBI"]
    second = project_features(ScanFeatures(features.values, changed, features.layer), aggregation="scan")
    np.testing.assert_array_equal(first.points[["Component1", "Component2"]], second.points[["Component1", "Component2"]])
    assert len(first.points) == 5 and first.parameters["Animals"] == 4
    figure = feature_space_figure(first)
    assert "4 animals / 5 scans" in figure.layout.title.text
    assert "PC1" in figure.layout.xaxis.title.text
    assert all("Subject" in trace.hovertemplate for trace in figure.data)


@pytest.mark.parametrize("column", ["Condition", "DatasetRole", "Dataset"])
def test_conflicting_animal_metadata_is_not_silently_assigned(features, column):
    features.manifest.loc[1, column] = "conflict"
    with pytest.raises(ValueError, match=f"conflicting {column}"):
        project_features(features)


def test_invalid_projection_inputs_have_actionable_errors(features):
    with pytest.raises(ValueError, match="three distinct"):
        project_features(ScanFeatures(features.values[:2], features.manifest.iloc[:2], features.layer))
    with pytest.raises(ValueError, match="identical"):
        project_features(ScanFeatures(np.ones_like(features.values), features.manifest, features.layer))
    with pytest.raises(ValueError, match="finite vectors"):
        project_features(ScanFeatures(features.values * np.nan, features.manifest, features.layer))
    with pytest.raises(ValueError, match="integer projection seed"):
        project_features(features, seed=-1)


def test_optional_umap_is_unsupervised_and_records_actual_neighbor_count(features, monkeypatch):
    calls = {}
    class FakeUMAP:
        def __init__(self, **kwargs):
            calls.update(kwargs)

        def fit_transform(self, values):
            calls["values"] = values.copy()
            return values[:, :2]

    monkeypatch.setitem(sys.modules, "umap", SimpleNamespace(UMAP=FakeUMAP))
    result = project_features(features, method="umap", n_neighbors=10, seed=17)
    assert calls["n_neighbors"] == result.parameters["NNeighbors"] == 3
    assert calls["random_state"] == 17
    assert calls["values"].shape == (4, 3) and result.explained_variance is None
    monkeypatch.setitem(sys.modules, "umap", None)
    with pytest.raises(ImportError, match="UMAP is optional"):
        project_features(features, method="umap")


@pytest.mark.parametrize("use_csv", [False, True])
def test_feature_notebook_runs_on_saved_weights_and_retains_longitudinal_scans(experiment, tmp_path, monkeypatch, use_csv):
    folder = tmp_path / "run"
    saved_run(experiment, folder)
    split = {"train": ["D1/SUBJ_0000", "D1/SUBJ_0001"], "validation": ["D1/SUBJ_0002"],
             "test": ["D1/SUBJ_0003"]}
    (folder / "splits.json").write_text(json.dumps(split))
    manifest_path = tmp_path / "inputs.csv"
    if use_csv:
        rows = []
        for path in sorted(experiment.data.root.rglob("N4.nii.gz")):
            relative = path.relative_to(experiment.data.root)
            subject = "/".join(relative.parts[:2])
            rows.append({"Subject": subject, "Scan": "/".join(relative.parts[1:-1]), "Dataset": "D1",
                         "ImagePath": str(path.relative_to(tmp_path)), "Condition": "TBI"})
        pd.DataFrame(rows).to_csv(manifest_path, index=False)
    shown = []
    monkeypatch.setattr(go.Figure, "show", lambda figure, **kwargs: shown.append(kwargs))
    monkeypatch.setattr(tf.keras.Model, "fit", lambda *args, **kwargs: pytest.fail("No fitting"))
    monkeypatch.setattr(tf.keras.Model, "compile", lambda *args, **kwargs: pytest.fail("No compilation"))
    path = Path(__file__).resolve().parents[1] / "notebook/check_feature_space.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    replacements = {"RUN_DIR": f"Path({str(folder)!r})", "DATA_ROOT": f"Path({str(experiment.data.root)!r})",
                    "DATASETS": "('D1',)", "RUN_FEATURE_EXTRACTION": "True", "MAX_ANIMALS_PER_DATASET": "4"}
    if use_csv:
        replacements["CSV_MANIFEST"] = f"Path({str(manifest_path)!r})"
    namespace = {}
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        tree = ast.parse("".join(cell["source"]))
        for node in tree.body:
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id in replacements:
                node.value = ast.parse(replacements[node.target.id], mode="eval").body
        ast.fix_missing_locations(tree)
        exec(compile(tree, str(path), "exec"), namespace)
    assert len(namespace["manifest"]) == 5
    assert namespace["scan_features"].values.shape == (5, 32)
    projection = namespace["projection"]
    assert len(projection.points) == 4 and projection.points.ScanCount.sum() == 5
    assert set(projection.points.DatasetRole) == {"train", "validation", "test"}
    assert shown == [{"renderer": "vscode"}]
    assert not (folder / "projection.csv").exists()
