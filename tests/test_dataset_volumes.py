"""Native physical annotation volumes and animal-level ventricle ratios."""

import json

import nibabel as nib
import numpy as np
import pytest

from src.evaluation.volumes import dataset_volume_report, voxel_volume_mm3


def write_subject(root, name, *, condition="SHAM", timepoints=(), missing_contra=False):
    folder = root / "D3" / name
    folder.mkdir(parents=True)
    annotation = ["il-V", "cl-V", "Les"]
    if missing_contra:
        annotation.remove("cl-V")
    metadata = {"dataset": "D3", "normalized_id": name, "original_id": name,
                "condition": condition, "modality": "T1w", "timepoints": list(timepoints),
                "annotated_regions": annotation, "has_lesion_annotation": condition != "SHAM"}
    (folder / "metadata.json").write_text(json.dumps(metadata))
    for index, timepoint in enumerate(timepoints or [None]):
        target = folder
        if timepoint is not None:
            target = folder / timepoint
            target.mkdir()
        data = np.zeros((5, 5, 5), dtype=np.int16)
        data.flat[:2 + index] = 13
        data.flat[10:14] = 3
        if condition == "TBI":
            data.flat[20:23] = 1
        affine = np.diag([0.1, 0.2, 0.3, 1.0])
        affine[0, 1] = 0.05  # Native shear does not change the determinant here.
        for name in ("N4", "Labels"):
            image = nib.Nifti1Image(data, affine)
            image.header.set_xyzt_units("mm")
            nib.save(image, target / f"{name}.nii.gz")

def test_dataset_volumes_use_original_labels_and_do_not_turn_missing_into_zero(tmp_path):
    write_subject(tmp_path, "a", timepoints=("7days", "30days"))
    write_subject(tmp_path, "b", missing_contra=True)
    write_subject(tmp_path, "c", condition="TBI")
    report = dataset_volume_report(tmp_path / "D3")
    assert len(report.scans) == 4 and len(report.animals) == 3
    animal = report.animals.loc[report.animals.Subject.eq("D3/a")].iloc[0]
    assert animal.Scans == 2
    assert animal.IpsiContraRatio == pytest.approx((0.5 + 0.75) / 2)
    assert animal.LesionUncorrectedMm3 == 0
    assert animal.LesionCorrectedMm3 == pytest.approx(-1.5 * 0.006)
    missing = report.scans.loc[report.scans.Subject.eq("D3/b")].iloc[0]
    assert np.isnan(missing.ContraVentricleMm3) and np.isnan(missing.LesionCorrectedMm3)
    assert missing.RatioStatus == "unavailable"
    sham = report.summary.loc[report.summary.Condition.eq("SHAM") & report.summary.Measurement.eq("IpsiContraRatio")].iloc[0]
    assert sham.ObservedAnimals == 2 and sham.FiniteAnimals == 1
    assert report.scans.loc[report.scans.Subject.eq("D3/a"), "NegativeCorrectedVolume"].all()
    folder = report.export(tmp_path / "volumes")
    assert (folder / "dataset_volumes.xlsx").is_file()
    with pytest.raises(FileExistsError):
        report.export(folder)

def test_voxel_volume_requires_units_and_converts_meters():
    image = nib.Nifti1Image(np.zeros((2, 2, 2)), np.diag([0.001, 0.002, 0.003, 1]))
    with pytest.raises(ValueError, match="spatial_unit"):
        voxel_volume_mm3(image)
    image.header.set_xyzt_units("meter")
    assert voxel_volume_mm3(image) == pytest.approx(6)

def test_zero_ventricle_is_available_and_ratio_is_undefined(tmp_path):
    write_subject(tmp_path, "a")
    path = tmp_path / "D3/a/Labels.nii.gz"
    original = nib.load(path)
    values = original.get_fdata().astype(np.int16)
    values[values == 3] = 0
    image = nib.Nifti1Image(values, original.affine, original.header)
    nib.save(image, path)
    report = dataset_volume_report(tmp_path / "D3")
    row = report.scans.iloc[0]
    assert row.ContraVentricleMm3 == 0 and np.isnan(row.IpsiContraRatio)
    assert row.RatioStatus == "zero_contralateral_volume"
    assert row.LesionCorrectedMm3 == pytest.approx(row.IpsiVentricleMm3)
