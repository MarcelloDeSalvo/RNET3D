"""Exploratory encoder-feature extraction and unsupervised animal/scan projections."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import nibabel as nib
import numpy as np
import pandas as pd

from ..preprocessing.preprocessor import Preprocessor
from .attention import classifier_feature_table


@dataclass(frozen=True)
class ScanFeatures:
    """One spatially pooled vector per manifest scan, from one fixed model/layer."""

    values: np.ndarray
    manifest: pd.DataFrame
    layer: str


@dataclass(frozen=True)
class FeatureProjection:
    """Exploratory coordinates and the exact unsupervised projection settings."""

    points: pd.DataFrame
    parameters: dict[str, Any]
    explained_variance: tuple[float, float] | None = None


def encoder_feature_table(model: Any) -> pd.DataFrame:
    """List encoder ancestors of the bottleneck, without requiring a classifier.

    Classifier models use the existing pre-global-pooling feature. Other current
    and legacy U-Nets use the input to their first transposed decoder convolution.
    This avoids selecting a decoder or deep-supervision projection as bottleneck.
    """
    import tensorflow as tf

    if "classifier" in model.output_names:
        return classifier_feature_table(model)

    preferred: str | None = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Conv3DTranspose):
            preferred = layer.input._keras_history.layer.name
            break
    if preferred is None:
        raise ValueError("Cannot infer an encoder bottleneck for this model")

    branch = tf.keras.Model(inputs=model.inputs, outputs=model.get_layer(preferred).output)
    rows: list[dict[str, Any]] = []
    for layer in branch.layers:
        tensors = tf.nest.flatten(layer.output)
        if len(tensors) == 1 and len(tensors[0].shape) == 5 and not isinstance(layer, tf.keras.layers.InputLayer):
            rows.append({"Layer": layer.name, "Type": type(layer).__name__, "Shape": tuple(tensors[0].shape),
                         "Default": layer.name == preferred})
    return pd.DataFrame(rows)


def extract_scan_features(
    model: Any, manifest: pd.DataFrame, pipeline: Preprocessor, *, layer: str | None = None,
) -> ScanFeatures:
    """Read each selected MRI once, preprocess deterministically and pool XYZ.

    The caller supplies the visible inference pipeline and scan selection.
    No metadata labels enter the network. The manifest must provide explicit
    dataset-scoped Subject IDs, Scan, Dataset and ImagePath; duplicates are rejected.
    Only one patch is held in memory and dropout/BN use training=False.
    """
    import tensorflow as tf

    required = ["Subject", "Scan", "Dataset", "ImagePath"]
    if manifest.empty or not set(required).issubset(manifest) or manifest[required].isna().any().any():
        raise ValueError("Provide selected scans with Subject, Scan, Dataset and ImagePath")
    if manifest[required].astype(str).apply(lambda column: column.str.strip().eq("")).any().any():
        raise ValueError("Scan identities and image paths must not be empty")
    if manifest.duplicated(["Subject", "Dataset", "Scan"]).any():
        raise ValueError("Select each scan once")
    if len(model.inputs) != 1 or len(model.input_shape) != 5:
        raise ValueError("Feature extraction requires one 3D model input")

    candidates = encoder_feature_table(model)
    if layer is None:
        layer = candidates.loc[candidates["Default"], "Layer"].iloc[0]
    if layer not in set(candidates["Layer"]):
        raise ValueError("Choose a spatial encoder layer from encoder_feature_table")

    probe = tf.keras.Model(inputs=model.inputs, outputs=model.get_layer(layer).output)
    vectors: list[np.ndarray] = []
    expected = tuple(model.input_shape[1:4])
    channels = int(model.input_shape[-1])

    for record in manifest.to_dict(orient="records"):
        source = nib.load(str(record["ImagePath"]))
        image = pipeline.preprocess(imgs=source)
        values = image.get_fdata(dtype=np.float32)
        if values.ndim != 3 or not np.isfinite(values).all():
            raise ValueError(f"{record['Subject']}: require a finite 3D preprocessed MRI")
        if any(size is not None and size != actual for size, actual in zip(expected, values.shape)):
            raise ValueError(f"Expected patch shape {expected}; the supplied pipeline must crop/pad accordingly")

        batch = np.repeat(values[None, ..., None], channels, axis=-1)
        feature_map = probe(batch, training=False)
        vector = tf.reduce_mean(tf.cast(feature_map[0], tf.float32), axis=(0, 1, 2)).numpy()
        if not np.isfinite(vector).all():
            raise ValueError("Encoder features contain nonfinite values")
        vectors.append(vector)

    return ScanFeatures(values=np.stack(vectors), manifest=manifest.reset_index(drop=True).copy(), layer=layer)


def project_features(
    features: ScanFeatures, *, aggregation: Literal["animal", "scan"] = "animal",
    method: Literal["pca", "umap"] = "pca", standardize: bool = True,
    seed: int = 42, n_neighbors: int = 10, min_dist: float = 0.1,
) -> FeatureProjection:
    """Project unlabeled feature vectors; metadata is attached only after fitting.

    Animal means give each selected animal one point regardless of scan count.
    Varying metadata is listed explicitly rather than assigned an arbitrary first
    value. PCA uses deterministic full SVD. UMAP is optional and never installed
    automatically. Its seed/neighbor count are recorded. Neither separation nor
    apparent clusters establish classification accuracy or causal feature learning.
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    values = np.asarray(features.values, dtype=float)
    manifest = features.manifest.reset_index(drop=True).copy()
    if values.ndim != 2 or len(values) != len(manifest) or values.shape[1] < 2 or not np.isfinite(values).all():
        raise ValueError("Provide matching scan metadata and finite vectors with at least two features")
    if aggregation not in ("animal", "scan") or method not in ("pca", "umap"):
        raise ValueError("Select animal/scan aggregation and pca/umap projection")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("Use a nonnegative integer projection seed")
    if "Subject" not in manifest or manifest["Subject"].isna().any():
        raise ValueError("Provide explicit subject identities")

    if aggregation == "animal":
        rows: list[dict[str, Any]] = []
        vectors: list[np.ndarray] = []
        for subject, group in manifest.groupby("Subject", sort=True):
            row: dict[str, Any] = {"Subject": subject, "ScanCount": len(group)}
            for column in manifest:
                if column == "Subject":
                    continue
                entries = sorted(set(group[column].fillna("unknown").astype(str).replace("", "unknown")))
                if column in ("Condition", "DatasetRole", "Dataset") and len(entries) > 1:
                    raise ValueError(f"{subject}: conflicting {column} across scans; check animal identity/roles")
                row[column] = " | ".join(entries)
            rows.append(row)
            vectors.append(values[group.index].mean(axis=0))
        manifest = pd.DataFrame(rows)
        values = np.stack(vectors)
    else:
        manifest["ScanCount"] = 1

    if len(values) < 3:
        raise ValueError("Select at least three distinct animals/scans for an exploratory 2D projection")
    if not np.any(np.ptp(values, axis=0) > 0):
        raise ValueError("All feature vectors are identical; choose another layer or distinct inputs")
    if standardize:
        values = StandardScaler().fit_transform(values)

    parameters: dict[str, Any] = {
        "Layer": features.layer, "Pooling": "spatial_mean_XYZ", "Aggregation": aggregation,
        "Method": method, "Standardize": standardize, "Seed": seed,
        "Points": len(values), "Animals": manifest["Subject"].nunique(), "SelectedScans": len(features.manifest),
    }
    explained: tuple[float, float] | None = None
    if method == "pca":
        reducer = PCA(n_components=2, svd_solver="full")
        coordinates = reducer.fit_transform(values)
        explained = tuple(float(value) for value in reducer.explained_variance_ratio_)
    else:
        if len(values) < 4 or not isinstance(n_neighbors, int) or n_neighbors < 2 or not 0 <= min_dist <= 1:
            raise ValueError("UMAP needs at least four points, neighbors >=2 and min_dist between zero and one")
        try:
            from umap import UMAP
        except ImportError as error:
            raise ImportError("UMAP is optional; use method=pca unless compatible umap-learn is already installed") from error
        used_neighbors = min(n_neighbors, len(values) - 1)
        reducer = UMAP(n_components=2, n_neighbors=used_neighbors, min_dist=min_dist, random_state=seed, init="random")
        coordinates = reducer.fit_transform(values)
        parameters.update(NNeighbors=used_neighbors, MinDist=min_dist, Initialization="random")

    if not np.isfinite(coordinates).all():
        raise ValueError("The projection produced nonfinite coordinates")
    manifest["Component1"] = coordinates[:, 0]
    manifest["Component2"] = coordinates[:, 1]
    return FeatureProjection(points=manifest, parameters=parameters, explained_variance=explained)


def feature_space_figure(projection: FeatureProjection, *, color: str = "Condition", symbol: str = "Dataset") -> Any:
    """Make a native Plotly scatter with metadata colors and explicit point counts."""
    import plotly.express as px

    points = projection.points.copy()
    if color not in points or symbol not in points:
        raise ValueError("Choose color and symbol columns present in the projection metadata")
    for column in (color, symbol):
        points[column] = points[column].fillna("unknown").replace("", "unknown").astype(str)
    labels: dict[str, str] = {"Component1": "Component 1", "Component2": "Component 2"}
    if projection.explained_variance is not None:
        labels = {"Component1": f"PC1 ({projection.explained_variance[0]:.1%})",
                  "Component2": f"PC2 ({projection.explained_variance[1]:.1%})"}
    hover = [column for column in ("Subject", "Scan", "Dataset", "DatasetRole", "Condition", "Modality", "Timepoint", "ScanCount") if column in points]
    title = (f"{projection.parameters['Method'].upper()} · {projection.parameters['Aggregation']} points · "
             f"{projection.parameters['Animals']} animals / {projection.parameters['SelectedScans']} scans")
    return px.scatter(
        points, x="Component1", y="Component2", color=color, symbol=symbol,
        hover_data=hover, labels=labels, title=title, template="plotly_white",
    )
