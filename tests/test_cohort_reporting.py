"""Native annotated volumes, explicit anatomy means and cohort/scan accounting."""


import numpy as np
import pandas as pd
import pytest

from src.evaluation.reporting import (
    attach_scan_metadata, evaluation_report, participant_flow, split_scan_manifest, structure_size_figure,
)


@pytest.fixture
def cohort_metrics():
    rows = []
    for subject, condition in (("a", "SHAM"), ("b", "TBI")):
        for region, dice, head, volume in (("Lesion", 0.9, "regions", 2),
                                           ("Cortex", 0.4, "regions", 1),
                                           ("Brain", 1.0, "brain_mask", 20)):
            missing = subject == "b" and region == "Cortex"
            for metric, value, unit in (("Dice", dice, "fraction"), ("ReferenceVolume", volume, "mm3"),
                                        ("PredictionVolume", volume, "mm3")):
                if missing:
                    value = np.nan
                rows.append({"Subject": subject, "Scan": subject + "_scan", "Dataset": "D1",
                             "Condition": condition, "Variant": "reference", "Seed": 42, "Fold": 0,
                             "DatasetRole": "test", "Head": head, "Region": region, "Metric": metric,
                             "Value": value, "Unit": unit, "Status": "unavailable" if missing else "ok"})
    return pd.DataFrame(rows)


def test_region_sets_separate_anatomy_brain_lesion_and_conditions(cohort_metrics):
    report = evaluation_report(cohort_metrics)
    sham = report.region_set_scans.loc[report.region_set_scans.Condition.eq("SHAM")].set_index("RegionSet")
    assert sham.loc["anatomy_only", "Value"] == pytest.approx(0.4)
    assert sham.loc["all_roi", "Value"] == pytest.approx(0.65)
    assert sham.loc["all_roi_and_brain", "Value"] == pytest.approx((0.9 + 0.4 + 1) / 3)
    assert sham.loc["anatomy_and_brain", "Value"] == pytest.approx(0.7)
    tbi = report.region_set_scans.loc[report.region_set_scans.Condition.eq("TBI")].set_index("RegionSet")
    assert np.isnan(tbi.loc["anatomy_only", "Value"])
    assert tbi.loc["anatomy_only", "ExpectedRegions"] == 1
    assert tbi.loc["anatomy_only", "FiniteRegions"] == 0
    assert not tbi.loc["anatomy_only", "CompleteRegionSet"]
    assert set(report.region_set_summary.Condition) == {"SHAM", "TBI"}
    assert report.region_coverage.loc[report.region_coverage.Status.eq("unavailable"), "Finite"].eq(0).all()
    assert not report.structure_size_metrics.empty
    assert len(structure_size_figure(report, region="Cortex").data) == 1


def test_scan_metadata_preserves_explicit_fused_flags_and_rejects_conflicts(cohort_metrics):
    manifest = pd.DataFrame([
        {"Subject": "a", "Scan": "a_scan", "Dataset": "D1", "Condition": "SHAM", "Modality": "T1w", "FusedBoundary": False},
        {"Subject": "b", "Scan": "b_scan", "Dataset": "D1", "Condition": "TBI", "Modality": "T2w", "FusedBoundary": True},
    ])
    report = evaluation_report(cohort_metrics, manifest=manifest)
    fused = report.subgroup_summary.loc[report.subgroup_summary.GroupBy.eq("FusedBoundary")]
    assert set(fused.GroupValue) == {"False", "True"}
    assert report.participant_flow.loc[report.participant_flow.Scope.eq("dataset_total"), "Animals"].item() == 2
    bad = manifest.copy()
    bad.loc[0, "Condition"] = "TBI"
    with pytest.raises(ValueError, match="Conflicting"):
        attach_scan_metadata(cohort_metrics, bad)
    with pytest.raises(ValueError, match="consistent record"):
        attach_scan_metadata(cohort_metrics, pd.concat([manifest, bad]))


def test_participant_and_split_flows_count_animals_and_scans_separately():
    manifest = pd.DataFrame([
        {"Subject": "rat1", "Scan": "r1_7", "Dataset": "D2", "Modality": "T1w", "Timepoint": "7days"},
        {"Subject": "rat1", "Scan": "r1_30", "Dataset": "D2", "Modality": "T1w", "Timepoint": "30days"},
        {"Subject": "rat2", "Scan": "r2_7", "Dataset": "D2", "Modality": "T1w", "Timepoint": "7days"},
    ])
    summary = participant_flow(manifest)
    total = summary.loc[summary.Scope.eq("dataset_total")].iloc[0]
    assert total.Animals == 2 and total.Scans == 3
    split = [{"train": ["rat1"], "validation": [], "test": ["rat2"]}]
    selection = split_scan_manifest(manifest, split)
    assert set(selection.DatasetRole) == {"train", "holdout"}
    assert len(selection.loc[selection.DatasetRole.eq("train")]) == 2
    fixed = split_scan_manifest(manifest, {"train": ["rat1"], "test": ["rat2"]})
    assert "test" in set(fixed.DatasetRole)


def test_model_repetitions_do_not_inflate_summary_animals(cohort_metrics):
    repeated = cohort_metrics.copy()
    repeated["Seed"] = 43
    report = evaluation_report(pd.concat([cohort_metrics, repeated], ignore_index=True))
    assert report.region_set_summary.Animals.eq(1).all()
    assert report.evaluation_flow.loc[report.evaluation_flow.Scope.eq("dataset_total"), "Animals"].eq(2).all()
    with pytest.raises(ValueError, match="Duplicate"):
        evaluation_report(pd.concat([cohort_metrics, cohort_metrics]))


def test_source_ids_preserve_background_and_lesion_semantics_with_short_names(cohort_metrics):
    table = cohort_metrics.copy()
    labels = {"Lesion": 1, "Cortex": 6, "Brain": 1}
    table["Label"] = table["Region"].map(labels)
    table.loc[table.Region.eq("Lesion"), "Region"] = "Les"
    background = table.loc[table.Region.eq("Les")].copy()
    background["Region"] = "BG"
    background["Label"] = 0
    background["Value"] = 1
    report = evaluation_report(pd.concat([table, background], ignore_index=True))
    sham = report.region_set_scans.loc[report.region_set_scans.Condition.eq("SHAM")].set_index("RegionSet")
    assert sham.loc["anatomy_only", "Value"] == pytest.approx(0.4)
    assert sham.loc["anatomy_and_brain", "Value"] == pytest.approx(0.7)
    assert "BG" not in set(report.region_coverage.Region)
