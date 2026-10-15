"""Explicit paired-reader masks: native-grid overlap, boundaries and volume ICC."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence
import hashlib
import json

import nibabel as nib
import numpy as np
import pandas as pd

from .metrics import segmentation_metrics_table
from .statistics import AgreementConfig, agreement_table
from .volumes import image_affine_mm


@dataclass(frozen=True)
class AnnotationPair:
    """Two independent readers on the same physical scan and label vocabulary.

    None availability means every selected region was fully delineated. An
    available empty structure is valid; a missing annotation must be declared.
    """

    subject: str
    scan: str
    dataset: str
    rater_a: Path
    rater_b: Path
    available_a: tuple[int, ...] | None = None
    available_b: tuple[int, ...] | None = None


def load_annotation_pairs(manifest_file: Path) -> list[AnnotationPair]:
    """Read explicit mask pairs from CSV; resolve relative paths beside the CSV.

    Required columns: Subject, Scan, Dataset, RaterA, RaterB. Optional
    AvailableA/AvailableB cells are JSON lists of available source label IDs.
    Files are paired by the manifest, never by guessed filenames.
    """
    path: Path = Path(manifest_file).resolve()
    table: pd.DataFrame = pd.read_csv(path)
    columns: list[str] = ["Subject", "Scan", "Dataset", "RaterA", "RaterB"]
    if not set(columns).issubset(table) or table.empty or table[columns].isna().any().any():
        raise ValueError("Provide nonempty Subject/Scan/Dataset/RaterA/RaterB columns")
    pairs: list[AnnotationPair] = []
    for row in table.to_dict(orient="records"):
        availability: dict[str, tuple[int, ...] | None] = {}
        for name in ("AvailableA", "AvailableB"):
            values: tuple[int, ...] | None = None
            if name in row and pd.notna(row[name]):
                declared = json.loads(row[name])
                if not isinstance(declared, list) or any(type(value) is not int or value < 0 for value in declared):
                    raise ValueError("Reader availability must be a JSON list of nonnegative integer IDs")
                if len(set(declared)) != len(declared):
                    raise ValueError("Reader availability must not repeat label IDs")
                values = tuple(declared)
            availability[name] = values
        paths: list[Path] = []
        for name in ("RaterA", "RaterB"):
            mask_path: Path = Path(row[name])
            if not mask_path.is_absolute():
                mask_path = path.parent / mask_path
            paths.append(mask_path.resolve())
        pairs.append(AnnotationPair(
            subject=str(row["Subject"]), scan=str(row["Scan"]), dataset=str(row["Dataset"]),
            rater_a=paths[0], rater_b=paths[1],
            available_a=availability["AvailableA"], available_b=availability["AvailableB"],
        ))

    return pairs


@dataclass
class InterRaterReport:
    """Scan metrics, animal metrics/volumes, volume agreement and paired sources."""

    scan_metrics: pd.DataFrame
    animal_metrics: pd.DataFrame
    metric_summary: pd.DataFrame
    volume_pairs: pd.DataFrame
    volume_agreement: pd.DataFrame
    sources: pd.DataFrame
    settings: dict[str, Any]

    def export(self, output_dir: Path) -> Path:
        """Export CSV/Excel in a fresh folder, verifying measured masks are unchanged."""
        for record in self.sources.to_dict(orient="records"):
            for name in ("RaterA", "RaterB"):
                path: Path = Path(record[name])
                digest: str = hashlib.sha256(path.read_bytes()).hexdigest()
                if digest != record[name + "SHA256"]:
                    raise ValueError("A reader mask changed after measurement; rerun the report")
        folder: Path = Path(output_dir)
        folder.mkdir(parents=True, exist_ok=False)
        tables: dict[str, pd.DataFrame] = {
            "scan_metrics": self.scan_metrics, "animal_metrics": self.animal_metrics,
            "metric_summary": self.metric_summary, "volume_pairs": self.volume_pairs,
            "volume_agreement": self.volume_agreement, "sources": self.sources,
        }
        with pd.ExcelWriter(folder / "inter_rater.xlsx") as writer:
            for name, table in tables.items():
                table.to_csv(folder / f"{name}.csv", index=False)
                table.to_excel(writer, sheet_name=name, index=False)
        config_path: Path = folder / "analysis.json"
        config_path.write_text(json.dumps(self.settings, indent=2), encoding="utf-8")

        return folder


def _paired_masks(pair: AnnotationPair, spatial_unit: str | None) -> tuple[Any, Any, np.ndarray]:
    """Require matching physical grids; never silently interpolate reader labels."""
    a = nib.load(str(pair.rater_a))
    b = nib.load(str(pair.rater_b))
    affine_a: np.ndarray = image_affine_mm(a, spatial_unit=spatial_unit)
    affine_b: np.ndarray = image_affine_mm(b, spatial_unit=spatial_unit)
    if len(a.shape) != 3 or a.shape != b.shape or not np.allclose(affine_a, affine_b, rtol=0, atol=1e-5):
        raise ValueError(f"{pair.subject}/{pair.scan}: reader masks must share the same physical grid")

    return a, b, affine_a


def inter_rater_report(
    pairs: Sequence[AnnotationPair], regions: Mapping[int, str], *,
    spatial_unit: str | None = None, agreement: AgreementConfig = AgreementConfig(unit="mm3"),
) -> InterRaterReport:
    """Measure paired labels, averaging longitudinal scans within animals for ICC.

    Both readers must use the same IDs and independently annotate selected
    regions throughout the volume. Availability is intersected explicitly.
    Empty-mask boundary distances remain undefined. A reader is not treated as
    biological truth: sensitivity/precision and signed bias are directional A→B.
    """
    if not pairs or not regions or 0 in regions:
        raise ValueError("Select explicit reader pairs and foreground label IDs")
    identities: list[tuple[str, str, str]] = [(pair.dataset, pair.subject, pair.scan) for pair in pairs]
    invalid_identity: bool = any(not all(isinstance(value, str) and value.strip() for value in identity) for identity in identities)
    if len(set(identities)) != len(identities) or invalid_identity:
        raise ValueError("Reader pairs require unique nonempty dataset/subject/scan identities")
    tables: list[pd.DataFrame] = []
    sources: list[dict[str, Any]] = []
    for pair in pairs:
        if Path(pair.rater_a).resolve() == Path(pair.rater_b).resolve():
            raise ValueError("Select two separate reader mask files, not the same file twice")
        for declared in (pair.available_a, pair.available_b):
            if declared is not None:
                if any(type(label) is not int or label < 0 for label in declared) or len(set(declared)) != len(declared):
                    raise ValueError("Reader availability requires unique nonnegative integer IDs")
        a, b, affine = _paired_masks(pair, spatial_unit)
        values_a: np.ndarray = np.asarray(a.dataobj)
        values_b: np.ndarray = np.asarray(b.dataobj)
        available: set[int] = set(regions)
        if pair.available_a is not None:
            available &= set(pair.available_a)
        if pair.available_b is not None:
            available &= set(pair.available_b)
        spacing: np.ndarray = np.linalg.norm(affine[:3, :3], axis=0)
        shared_labels: list[int] = sorted(available)
        table = segmentation_metrics_table(
            reference=values_a, prediction=values_b, spacing_mm=spacing, affine_mm=affine,
            regions=regions, subject=pair.subject, scan=pair.scan, dataset=pair.dataset,
            available_labels=shared_labels,
        )
        tables.append(table)
        source: dict[str, Any] = {
            "Dataset": pair.dataset, "Subject": pair.subject, "Scan": pair.scan,
            "AvailableA": json.dumps(pair.available_a), "AvailableB": json.dumps(pair.available_b),
        }
        for name, path in (("RaterA", pair.rater_a), ("RaterB", pair.rater_b)):
            resolved: Path = Path(path).resolve()
            source[name] = str(resolved)
            source[name + "SHA256"] = hashlib.sha256(resolved.read_bytes()).hexdigest()
        sources.append(source)

    metrics: pd.DataFrame = pd.concat(tables, ignore_index=True)
    keys: list[str] = ["Dataset", "Region", "Label", "Metric", "Unit", "Subject"]
    valid: pd.DataFrame = metrics.loc[np.isfinite(metrics["Value"])].copy()
    animals: pd.DataFrame = valid.groupby(keys, dropna=False).agg(
        Value=("Value", "mean"), Scans=("Scan", "nunique"),
    ).reset_index()
    summary: pd.DataFrame = animals.groupby(keys[:-1], dropna=False).agg(
        Animals=("Subject", "nunique"), Mean=("Value", "mean"), SD=("Value", "std"),
        Minimum=("Value", "min"), Maximum=("Value", "max"),
    ).reset_index()
    selected: pd.DataFrame = metrics.loc[metrics["Metric"].isin(["ReferenceVolume", "PredictionVolume"])]
    volume_keys: list[str] = ["Dataset", "Region", "Label", "Subject", "Scan"]
    scans: pd.DataFrame = selected.pivot(index=volume_keys, columns="Metric", values="Value").reset_index()
    scans = scans.rename(columns={"ReferenceVolume": "RaterA", "PredictionVolume": "RaterB"})
    scans = scans.dropna(subset=["RaterA", "RaterB"])
    volumes: pd.DataFrame = scans.groupby(volume_keys[:-1], dropna=False).agg(
        RaterA=("RaterA", "mean"), RaterB=("RaterB", "mean"), Scans=("Scan", "nunique"),
    ).reset_index()
    settings = replace(agreement, subject_col="Subject", reference_col="RaterA", prediction_col="RaterB", unit="mm3", group_cols=())
    agreements: list[dict[str, Any]] = []
    for key, group in volumes.groupby(["Dataset", "Region", "Label"], dropna=False):
        row: dict[str, Any] = dict(zip(["Dataset", "Region", "Label"], key))
        row.update({"n": len(group), "Status": "too_few_animals"})
        if len(group) >= 3:
            row.update(agreement_table(group, settings).iloc[0].to_dict())
            row["Status"] = "ok"
        agreements.append(row)
    from dataclasses import asdict

    provenance: dict[str, Any] = {
        "regions": dict(regions), "spatial_unit_override": spatial_unit,
        "agreement": asdict(settings), "statistical_unit": "animal mean scan measurement",
        "grid": "shared original physical grid; no interpolation",
        "independence": "reader independence/experience/blinding must be documented by the study",
    }
    return InterRaterReport(
        scan_metrics=metrics, animal_metrics=animals, metric_summary=summary, volume_pairs=volumes,
        volume_agreement=pd.DataFrame(agreements), sources=pd.DataFrame(sources), settings=provenance,
    )


def inter_rater_slice_figure(
    pair: AnnotationPair, *, reference: Path, axis: int = 0, position: float = 0.5,
    spatial_unit: str | None = None,
) -> Any:
    """Show both labels in the shared notebook reference orientation with Plotly."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    if axis not in (0, 1, 2) or not np.isfinite(position) or not 0 <= position <= 1:
        raise ValueError("Choose axis 0/1/2 and a relative slice position between zero and one")
    a, b, _ = _paired_masks(pair, spatial_unit=spatial_unit)
    target = nib.load(str(reference))
    source_orientation = nib.orientations.io_orientation(a.affine)
    target_orientation = nib.orientations.io_orientation(target.affine)
    transform = nib.orientations.ornt_transform(source_orientation, target_orientation)
    arrays: list[np.ndarray] = []
    for image in (a, b):
        values: np.ndarray = np.asarray(image.dataobj)
        values = nib.orientations.apply_orientation(values, transform)
        arrays.append(values)
    index: int = round(position * (arrays[0].shape[axis] - 1))
    maximum: int = max(int(values.max()) for values in arrays)
    figure = make_subplots(rows=1, cols=2, subplot_titles=("Reader A", "Reader B"))
    for column, values in enumerate(arrays, start=1):
        slice_data: np.ndarray = np.take(values, index, axis=axis).T
        trace = go.Heatmap(z=slice_data, zmin=0, zmax=maximum, colorscale="Viridis", showscale=column == 2)
        figure.add_trace(trace, row=1, col=column)
        figure.update_yaxes(autorange="reversed", row=1, col=column)
    figure.update_layout(title=f"{pair.subject} / {pair.scan}: paired annotations")

    return figure
