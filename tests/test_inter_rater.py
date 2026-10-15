"""Paired-reader identity, native geometry, missing annotations and animal ICC."""

import json

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

from src.evaluation.inter_rater import AnnotationPair, inter_rater_report, inter_rater_slice_figure, load_annotation_pairs
from src.evaluation.statistics import AgreementConfig


def write_mask(path, data, affine):
    image = nib.Nifti1Image(data, affine)
    image.header.set_xyzt_units("mm")
    nib.save(image, path)


def test_inter_rater_uses_animal_not_longitudinal_scan_count(tmp_path):
    pairs = []
    for number in range(3):
        for timepoint in (7, 30):
            values = np.zeros((6, 6, 6), dtype=np.uint8)
            values.flat[:number + 1] = 1
            a = tmp_path / f"a_{number}_{timepoint}.nii.gz"
            b = tmp_path / f"b_{number}_{timepoint}.nii.gz"
            write_mask(a, values, np.eye(4))
            write_mask(b, values, np.eye(4))
            pairs.append(AnnotationPair(f"rat{number}", f"{number}_{timepoint}", "D2", a, b))
    settings = AgreementConfig(bootstrap_samples=100)
    report = inter_rater_report(pairs, {1: "Lesion"}, agreement=settings)
    dice = report.metric_summary.loc[report.metric_summary.Metric.eq("Dice")].iloc[0]
    assert dice.Animals == 3 and dice.Mean == 1
    assert report.volume_pairs.Scans.eq(2).all()
    assert report.volume_agreement.n.item() == 3
    assert report.volume_agreement.icc_a1.item() == pytest.approx(1)
    target = report.export(tmp_path / "rater_report")
    assert (target / "inter_rater.xlsx").is_file()
    assert json.loads((target / "analysis.json").read_text())["statistical_unit"] == "animal mean scan measurement"
    with pytest.raises(FileExistsError):
        report.export(target)
    with pytest.raises(ValueError, match="unique"):
        inter_rater_report([pairs[0], pairs[0]], {1: "Lesion"})


def test_reader_manifest_relative_paths_and_explicit_unavailable_labels(tmp_path):
    values = np.zeros((4, 4, 4), dtype=np.uint8)
    values[1, 1, 1] = 1
    a = tmp_path / "a.nii.gz"
    b = tmp_path / "b.nii.gz"
    write_mask(a, values, np.eye(4))
    write_mask(b, values, np.eye(4))
    path = tmp_path / "pairs.csv"
    pd.DataFrame([{"Subject": "mouse", "Scan": "scan", "Dataset": "D1", "RaterA": a.name,
                   "RaterB": b.name, "AvailableA": "[1, 3]", "AvailableB": "[1]"}]).to_csv(path, index=False)
    pairs = load_annotation_pairs(path)
    assert pairs[0].rater_a == a
    report = inter_rater_report(pairs, {1: "Lesion", 3: "Ventricle"}, agreement=AgreementConfig(bootstrap_samples=100))
    assert report.scan_metrics.loc[report.scan_metrics.Region.eq("Ventricle"), "Status"].eq("unavailable").all()
    assert report.volume_agreement.Status.item() == "too_few_animals"
    moved = np.eye(4)
    moved[0, 3] = 1
    write_mask(b, values, moved)
    with pytest.raises(ValueError, match="physical grid"):
        inter_rater_report(pairs, {1: "Lesion"})


def test_reader_report_accepts_shared_native_shear_and_verifies_saved_masks(tmp_path):
    values = np.zeros((4, 4, 4), dtype=np.uint8)
    values[1, 1, 1] = 1
    a = tmp_path / "a.nii.gz"
    b = tmp_path / "b.nii.gz"
    affine = np.eye(4)
    affine[0, 1] = 0.1
    write_mask(a, values, affine)
    write_mask(b, values, affine)
    pair = AnnotationPair("a", "a_scan", "D1", a, b)
    report = inter_rater_report([pair], {1: "Lesion"})
    assert report.scan_metrics.loc[report.scan_metrics.Metric.eq("HD95"), "Value"].item() == 0
    figure = inter_rater_slice_figure(pair, reference=a, axis=0, position=1/3)
    assert len(figure.data) == 2
    np.testing.assert_array_equal(figure.data[0].z, figure.data[1].z)
    assert np.asarray(figure.data[0].z).sum() == 1
    values[2, 1, 1] = 1
    write_mask(b, values, affine)
    with pytest.raises(ValueError, match="changed after measurement"):
        report.export(tmp_path / "changed")
