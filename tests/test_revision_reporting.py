"""Reporting preserves statistical units, failures, timing provenance and inputs."""

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.evaluation.revision import (
    CostSource, MetricSource, RevisionConfig, cost_report, export_revision,
    failure_case_table, load_metric_sources, load_patch_sources, revision_report,
)


@pytest.fixture
def metrics(tmp_path):
    """Three independent animals with model repetitions and missing/empty GT."""
    rows = []
    for animal in range(3):
        for seed in (11, 12):
            values = {"Dice": 0.2 + animal / 4, "ReferenceVolume": animal + 1.0,
                      "PredictionVolume": animal + 1.5}
            for metric, value in values.items():
                unit = "mm3"
                if metric == "Dice":
                    unit = "unitless"
                rows.append({
                    "Variant": "3drnet", "Seed": seed, "Fold": 0, "DatasetRole": "test",
                    "Subject": f"D1/SUBJ_{animal}", "Scan": f"scan_{animal}", "Dataset": "D1",
                    "Region": "Lesion", "Metric": metric, "Value": value, "Unit": unit,
                    "Status": "ok", "EvaluationStage": "full_volume", "Head": "regions",
                })
    rows.extend([
        {**rows[0], "Region": "CC", "Value": np.nan, "Status": "unavailable"},
        {**rows[0], "Region": "Ventricle", "Value": 0.0, "Status": "prediction_empty"},
        {**rows[0], "Region": "Other", "Value": 1.0, "Status": "both_empty"},
        {**rows[0], "Region": "Brain", "Value": 0.95, "Head": "brain_mask"},
    ])
    path = tmp_path / "metrics.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return MetricSource(path=path)


@pytest.fixture
def training_cost(tmp_path):
    """Two recorded folds with distinct epoch counts on a known runtime."""
    path = tmp_path / "fold_timings.csv"
    pd.DataFrame([
        {"fold": 0, "training_seconds": 100.0, "evaluation_seconds": 20.0,
         "epochs_completed": 10, "parameter_count": 1234, "best_epoch": 8},
        {"fold": 1, "training_seconds": 200.0, "evaluation_seconds": 30.0,
         "epochs_completed": 12, "parameter_count": 1234, "best_epoch": 10},
    ]).to_csv(path, index=False)
    metadata = tmp_path / "metadata.json"
    metadata.write_text(json.dumps({"devices": ["/physical_device:GPU:0"], "precision": "float32"}))
    return CostSource(path=path, name="3drnet", seed=42, metadata_file=metadata)


@pytest.fixture
def patch_source(tmp_path):
    """Native pooled loss and region Dice with an incomplete final batch retained."""
    path = tmp_path / "patch_metrics.csv"
    rows = []
    for role in ("validation", "test"):
        for metric in ("loss", "regions_mean_dice"):
            rows.append({"DatasetRole": role, "Head": "regions", "Region": "Regions",
                         "Metric": metric, "Value": 0.8, "Aggregation": "cohort_pooled",
                         "EvaluationStage": "patch", "AnimalCount": 3, "ScanCount": 3,
                         "PatchCount": 9})
    pd.DataFrame(rows).to_csv(path, index=False)
    return MetricSource(path=path, name="3drnet", seed=42, fold=0)


def test_reporting_keeps_native_patch_and_timing_units_out_of_animal_bootstrap(metrics, patch_source, training_cost):
    config = RevisionConfig(metrics=("Dice",), bootstrap_samples=100)
    report = revision_report([metrics], config, patch_sources=[patch_source], cost_sources=[training_cost])
    lesion = report.summary.loc[report.summary.Region.eq("Lesion")].iloc[0]
    assert lesion.Animals == 3
    assert lesion.Mean == pytest.approx(0.45)
    assert report.patch_metrics.PatchCount.eq(9).all()
    assert set(report.patch_metrics.DatasetRole) == {"validation", "test"}
    coverage = report.coverage.loc[report.coverage.Region.eq("Lesion") & report.coverage.Metric.eq("Dice")].iloc[0]
    assert coverage.ObservedAnimals == coverage.ObservedScans == coverage.FiniteAnimals == 3
    assert coverage.RawRows == 6
    missing = report.coverage.loc[report.coverage.Region.eq("CC")].iloc[0]
    assert missing.UnavailableRows == 1 and missing.FiniteAnimals == 0
    assert "Brain" in set(report.summary.Region)
    assert report.costs.measurements.Kind.eq("training").all()


def test_failure_review_preserves_seed_fold_and_does_not_remove_bad_scores(metrics):
    raw = load_metric_sources([metrics])
    candidates = failure_case_table(raw)
    assert set(candidates.ReviewReason) == {"low_dice", "prediction_empty"}
    assert not candidates.Region.isin(["CC", "Other", "Brain"]).any()
    assert set(candidates.loc[candidates.Region.eq("Lesion"), "Seed"]) == {11, 12}
    assert set(candidates.loc[candidates.Region.eq("Lesion"), "Value"]) == {0.2, 0.45}
    assert len(raw) == 22
    fewer = failure_case_table(raw, dice_threshold=0.1)
    assert len(fewer) == 1  # Empty prediction still needs review.


@pytest.mark.parametrize("threshold", [-0.1, 1.1, np.nan, np.inf])
def test_review_rejects_invalid_thresholds(metrics, threshold):
    with pytest.raises(ValueError, match="threshold"):
        RevisionConfig(failure_dice_threshold=threshold)
    with pytest.raises(ValueError, match="threshold"):
        failure_case_table(load_metric_sources([metrics]), dice_threshold=threshold)


def test_cost_summary_uses_recorded_folds_and_hardware(training_cost):
    result = cost_report([training_cost])
    assert len(result.measurements) == 10
    training = result.summary.loc[result.summary.Metric.eq("training_seconds")].iloc[0]
    assert training.Recorded == 2 and training.Mean == 150
    assert training.Device == "/physical_device:GPU:0"
    assert training.Precision == "float32" and training.ProvenanceStatus == "recorded"
    assert len(result.sources) == 2
    with pytest.raises(ValueError, match="Duplicate cost"):
        cost_report([training_cost, training_cost])


def test_inference_cost_uses_its_actual_evaluation_runtime_and_keeps_roles_separate(tmp_path):
    path = tmp_path / "inference_timings.csv"
    pd.DataFrame([
        {"Subject": "D1/a", "Scan": "a", "PreprocessingSeconds": 1,
         "InferenceSeconds": 8, "PostprocessingSeconds": 2},
        {"Subject": "D1/b", "Scan": "b", "PreprocessingSeconds": 1,
         "InferenceSeconds": 12, "PostprocessingSeconds": 2},
    ]).to_csv(path, index=False)
    metadata = tmp_path / "evaluation.json"
    metadata.write_text(json.dumps({"runtime": {"gpu_devices": ["evaluation_GPU"], "precision": "mixed_float16"}}))
    sources = [
        CostSource(path=path, name="3drnet", seed=42, fold=0, role="test", dataset="D1", metadata_file=metadata),
        CostSource(path=path, name="3drnet", seed=42, fold=0, role="external", dataset="D4"),
    ]
    result = cost_report(sources)
    summary = result.summary.loc[result.summary.Metric.eq("InferenceSeconds")]
    assert len(summary) == 2
    assert summary.Mean.eq(10).all()
    external = summary.loc[summary.DatasetRole.eq("external")].iloc[0]
    assert external.Device == external.Precision == "unknown"
    assert external.ProvenanceStatus == "missing_hardware_metadata"
    test = summary.loc[summary.DatasetRole.eq("test")].iloc[0]
    assert test.Device == "evaluation_GPU" and test.Precision == "mixed_float16"


@pytest.mark.parametrize("value", [-1, np.inf, np.nan])
def test_cost_rejects_invalid_durations(training_cost, value):
    table = pd.read_csv(training_cost.path)
    table.loc[0, "training_seconds"] = value
    table.to_csv(training_cost.path, index=False)
    with pytest.raises(ValueError, match="finite and nonnegative"):
        cost_report([training_cost])


def test_missing_parameter_counts_are_not_reported_as_zero(training_cost):
    table = pd.read_csv(training_cost.path)
    table["parameter_count"] = 0
    table.to_csv(training_cost.path, index=False)
    result = cost_report([training_cost])
    params = result.summary.loc[result.summary.Metric.eq("parameter_count")].iloc[0]
    assert params.Recorded == 0 and np.isnan(params.Mean)
    assert result.measurements.loc[result.measurements.Metric.eq("parameter_count"), "Status"].eq("not_recorded").all()


def test_patch_contract_rejects_duplicates_and_scan_metrics(patch_source):
    with pytest.raises(ValueError, match="duplicate"):
        load_patch_sources([patch_source, patch_source])
    table = pd.read_csv(patch_source.path)
    table["Aggregation"] = "scan"
    table.to_csv(patch_source.path, index=False)
    with pytest.raises(ValueError, match="cohort_pooled"):
        load_patch_sources([patch_source])


def test_revision_exports_separate_workbook_sheets_and_exact_source_hashes(metrics, patch_source, training_cost, tmp_path):
    config = RevisionConfig(bootstrap_samples=100)
    report = revision_report([metrics], config, patch_sources=[patch_source], cost_sources=[training_cost])
    target = export_revision(report, tmp_path / "revision", config)
    sheets = pd.ExcelFile(target / "revision_tables.xlsx").sheet_names
    assert {"native_patch_metrics", "metric_summary", "metric_coverage", "failure_review", "cost_summary", "source_files"} <= set(sheets)
    sources = pd.read_csv(target / "source_files.csv")
    for record in sources.to_dict(orient="records"):
        assert hashlib.sha256(Path(record["Path"]).read_bytes()).hexdigest() == record["SHA256"]
    payload = json.loads((target / "analysis.json").read_text())
    assert payload["analysis"]["failure_dice_threshold"] == 0.5
    assert len(payload["source_files"]) == 4
    assert "D4" in (target / "reporting_notes.md").read_text()
    with pytest.raises(FileExistsError):
        export_revision(report, target, config)


def test_revision_refuses_changed_inputs_before_creating_output(metrics, tmp_path):
    config = RevisionConfig(bootstrap_samples=100)
    report = revision_report([metrics], config)
    with metrics.path.open("a") as stream:
        stream.write("\n")
    target = tmp_path / "changed"
    with pytest.raises(ValueError, match="changed since analysis"):
        export_revision(report, target, config)
    assert not target.exists()


def test_revision_refuses_mislabelled_analysis_settings(metrics, tmp_path):
    config = RevisionConfig(bootstrap_samples=100)
    report = revision_report([metrics], config)
    changed = replace(config, failure_dice_threshold=0.8)
    target = tmp_path / "changed_config"
    with pytest.raises(ValueError, match="configuration changed"):
        export_revision(report, target, changed)
    assert not target.exists()


def test_reference_empty_and_unexpected_undefined_measurements_need_review(metrics):
    table = load_metric_sources([metrics])
    extra = table.iloc[[0, 1]].copy()
    extra.iloc[0, extra.columns.get_loc("Status")] = "reference_empty"
    extra.iloc[1, extra.columns.get_loc("Value")] = np.nan
    result = failure_case_table(extra)
    assert set(result.ReviewReason) == {"reference_empty", "undefined_measurement"}


def test_brain_evaluator_rows_remain_available_for_review(metrics):
    table = pd.read_csv(metrics.path)
    table["Evaluator"] = "regions"
    table.loc[table.Head.eq("brain_mask"), "Evaluator"] = "brain_mask"
    table.to_csv(metrics.path, index=False)
    raw = load_metric_sources([metrics])
    assert "Brain" in set(raw.Region)


def test_combined_training_export_does_not_mix_patch_rows_into_scan_summaries(metrics):
    table = pd.read_csv(metrics.path)
    patch = {**table.iloc[0].to_dict(), "EvaluationStage": "patch", "Value": 0.0}
    pd.concat([table, pd.DataFrame([patch])], ignore_index=True).to_csv(metrics.path, index=False)
    actual = load_metric_sources([metrics])
    assert len(actual) == len(table)
    assert actual.EvaluationStage.eq("full_volume").all()


def test_revision_notebook_defaults_do_not_read_results_or_train(capsys):
    import nbformat

    path = Path(__file__).resolve().parents[1] / "notebook" / "10_revision_results.ipynb"
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    namespace = {}
    for cell in notebook.cells:
        if cell.cell_type == "code":
            exec(compile(cell.source, str(path), "exec"), namespace)
    assert namespace["report"] is None
    assert namespace["classifier_results"] is None
    assert namespace["PATCH_SOURCES"] == namespace["COST_SOURCES"] == []
