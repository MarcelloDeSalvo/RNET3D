"""Native-grid annotated volumes and descriptive ventricle symmetry reports."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
import json

import nibabel as nib
import numpy as np
import pandas as pd


def image_affine_mm(image: nib.spatialimages.SpatialImage, *, spatial_unit: str | None = None) -> np.ndarray:
    """Convert a copied physical affine to mm without editing image geometry.

    A unit override is required when the NIfTI header does not declare units.
    It changes measurement interpretation only; neither image nor mask is edited.
    """
    unit: str = image.header.get_xyzt_units()[0]
    if spatial_unit is not None:
        unit = spatial_unit
    factors: dict[str, float] = {"mm": 1.0, "meter": 1000.0, "micron": 0.001}
    if unit not in factors:
        raise ValueError("Declare spatial_unit='mm', 'meter' or 'micron' for unknown image units")
    affine: np.ndarray = np.asarray(image.affine, dtype=float)
    determinant: float = float(abs(np.linalg.det(affine[:3, :3])))
    if not np.isfinite(affine).all() or not np.isfinite(determinant) or determinant <= 0:
        raise ValueError("Volume measurement requires a finite, invertible affine")

    result: np.ndarray = affine.copy()
    result[:3] *= factors[unit]
    return result


def voxel_volume_mm3(image: nib.spatialimages.SpatialImage, *, spatial_unit: str | None = None) -> float:
    """Measure physical voxel volume using the full affine, including shear."""
    affine: np.ndarray = image_affine_mm(image, spatial_unit=spatial_unit)
    return float(abs(np.linalg.det(affine[:3, :3])))


@dataclass(frozen=True)
class VolumeLabels:
    """Original annotation IDs; independent of training channel order."""

    lesion: int = 1
    ipsilateral: int = 13
    contralateral: int = 3

    def __post_init__(self) -> None:
        """Require distinct foreground IDs before inspecting annotations."""
        values: tuple[int, ...] = (self.lesion, self.ipsilateral, self.contralateral)
        if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in values):
            raise ValueError("Volume label IDs must be positive integers")
        if len(set(values)) != len(values):
            raise ValueError("Lesion and ventricle label IDs must be distinct")


@dataclass
class DatasetVolumeReport:
    """All scan measurements, animal means and condition-specific distributions."""

    scans: pd.DataFrame
    animals: pd.DataFrame
    summary: pd.DataFrame
    settings: dict[str, Any] = field(default_factory=dict)

    def export(self, output_dir: Path) -> Path:
        """Save full-precision tables to a fresh folder without changing the dataset."""
        folder: Path = Path(output_dir)
        folder.mkdir(parents=True, exist_ok=False)
        tables: dict[str, pd.DataFrame] = {
            "scan_volumes": self.scans, "animal_volumes": self.animals, "volume_summary": self.summary,
        }
        with pd.ExcelWriter(folder / "dataset_volumes.xlsx") as writer:
            for name, table in tables.items():
                table.to_csv(folder / f"{name}.csv", index=False)
                table.to_excel(writer, sheet_name=name, index=False)
        settings_path: Path = folder / "analysis.json"
        settings_path.write_text(json.dumps(self.settings, indent=2), encoding="utf-8")

        return folder


def dataset_volume_report(
    dataset_dir: Path, *, labels: VolumeLabels = VolumeLabels(),
    image_name: str = "N4", mask_name: str = "Labels", spatial_unit: str | None = None,
) -> DatasetVolumeReport:
    """Inspect every annotated subject/scan in one normalized Dx directory.

    Availability follows metadata, never the presence of positive voxels. SHAM
    lesion absence is known, but missing ventricular annotations stay missing.
    The correction is lesion + ipsilateral - contralateral, without clipping;
    negative estimates are retained and flagged. Longitudinal scans are averaged
    within each animal before descriptive condition-specific summaries.
    """
    from ..experiments.data import annotation_values
    from .k_fold import SubjectIndex
    from ..utils.loader import RodentDatasets

    folder: Path = Path(dataset_dir).resolve()
    loader = RodentDatasets()
    loader.add_normalized_dataset(
        dataset_path=folder.parent, dataset=folder.name, image_name=image_name,
        roi_mask_name=mask_name, brain_mask_name=None,
    )
    index = SubjectIndex(loader)
    rows: list[dict[str, Any]] = []
    measurements: dict[str, int] = {
        "LesionMm3": labels.lesion, "IpsiVentricleMm3": labels.ipsilateral,
        "ContraVentricleMm3": labels.contralateral,
    }
    for key in loader.get_subjects_list():
        scan = loader.get_subject(key)
        available: set[int] = annotation_values(scan)
        values: np.ndarray = np.asarray(scan.labels.dataobj)
        if values.ndim != 3 or not np.isfinite(values).all() or np.any(values != np.rint(values)) or np.any(values < 0):
            raise ValueError(f"{scan.id}: annotation masks must contain nonnegative integer 3D labels")
        if scan.get_sham() and np.any(values == labels.lesion):
            raise ValueError(f"{scan.id}: verified SHAM scan contains lesion-positive voxels")
        voxel_mm3: float = voxel_volume_mm3(scan.labels, spatial_unit=spatial_unit)
        row: dict[str, Any] = {
            "Subject": index.subject_for_scan(key), "Scan": scan.id, "Dataset": folder.name,
            "Condition": scan.metadata["condition"], "Modality": scan.modality,
            "Timepoint": scan.timepoint or "unspecified", "Subgroup": scan.metadata.get("subgroup", "unknown"),
            "MaskFile": str(Path(scan.labels_path).resolve()), "VoxelVolumeMm3": voxel_mm3,
        }
        for column, label in measurements.items():
            value: float = float("nan")
            if label in available:
                value = float(np.count_nonzero(values == label) * voxel_mm3)
            row[column] = value

        ipsi: float = row["IpsiVentricleMm3"]
        contra: float = row["ContraVentricleMm3"]
        ratio: float = float("nan")
        ratio_status: str = "unavailable"
        if np.isfinite(ipsi) and np.isfinite(contra):
            ratio_status = "zero_contralateral_volume"
            if contra > 0:
                ratio = ipsi / contra
                ratio_status = "ok"
        corrected: float = row["LesionMm3"] + ipsi - contra
        row.update({
            "IpsiContraRatio": ratio, "RatioStatus": ratio_status,
            "LesionUncorrectedMm3": row["LesionMm3"], "LesionCorrectedMm3": corrected,
            "NegativeCorrectedVolume": bool(np.isfinite(corrected) and corrected < 0),
        })
        rows.append(row)

    if not rows:
        raise ValueError("No normalized scans found in the selected Dx dataset")
    scans: pd.DataFrame = pd.DataFrame(rows)
    columns: list[str] = [*measurements, "IpsiContraRatio", "LesionUncorrectedMm3", "LesionCorrectedMm3"]
    keys: list[str] = ["Dataset", "Condition", "Subject"]
    animal_means: pd.DataFrame = scans.groupby(keys, dropna=False)[columns].mean().reset_index()
    counts: pd.DataFrame = scans.groupby(keys, dropna=False).agg(Scans=("Scan", "nunique")).reset_index()
    animals: pd.DataFrame = animal_means.merge(counts, on=keys, validate="one_to_one")
    for column in columns:
        valid_name: str = column + "Scans"
        valid_counts: pd.DataFrame = scans.groupby(keys, dropna=False)[column].count().reset_index(name=valid_name)
        animals = animals.merge(valid_counts, on=keys, validate="one_to_one")
    long: pd.DataFrame = animals.melt(id_vars=keys, value_vars=columns, var_name="Measurement", value_name="Value")
    groups: list[str] = ["Dataset", "Condition", "Measurement"]
    summary: pd.DataFrame = long.groupby(groups, dropna=False).agg(
        ObservedAnimals=("Subject", "nunique"), FiniteAnimals=("Value", "count"),
        Mean=("Value", "mean"), SD=("Value", "std"), Median=("Value", "median"),
        Minimum=("Value", "min"), Maximum=("Value", "max"),
    ).reset_index()

    settings: dict[str, Any] = {
        "dataset_dir": str(folder), "labels": asdict(labels), "spatial_unit_override": spatial_unit,
        "image_name": image_name, "mask_name": mask_name, "statistical_unit": "animal mean scan measurement",
        "ratio": "mean of defined scan-wise ipsilateral/contralateral ratios within each animal",
        "correction": "lesion + ipsilateral ventricle - contralateral ventricle; no clipping",
    }
    return DatasetVolumeReport(scans=scans, animals=animals, summary=summary, settings=settings)


def ventricle_ratio_figure(report: DatasetVolumeReport) -> Any:
    """Display animal mean ratios by condition; no normality or symmetry test."""
    import plotly.express as px

    table: pd.DataFrame = report.animals.loc[np.isfinite(report.animals["IpsiContraRatio"])].copy()
    figure = px.histogram(
        table, x="IpsiContraRatio", color="Condition", marginal="rug", barmode="overlay",
        hover_data=["Subject", "Scans"], title="Annotated ipsilateral / contralateral ventricle volume",
    )
    figure.add_vline(x=1.0, line_dash="dash")

    return figure
