"""Fold estimands, animal uncertainty and readable paper tables from saved CSVs."""

import ast
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.evaluation.paper_tables import PaperTableConfig, build_paper_tables, paper_tables_from_folder


def metrics_frame(records, *, role="holdout"):
    """One Dice row per explicit animal/scan/fold observation."""
    rows = []
    for subject, scan, fold, value in records:
        rows.append({"Variant": "reference", "Seed": 42, "DatasetRole": role, "Dataset": "D1",
                     "Condition": "TBI", "Head": "regions", "Region": "Lesion", "Label": 1,
                     "Subject": subject, "Scan": scan, "Fold": fold, "Metric": "Dice",
                     "Value": value, "Unit": "fraction", "Status": "ok"})
    return pd.DataFrame(rows)


@pytest.fixture
def settings():
    return PaperTableConfig(bootstrap_samples=100)


def test_fold_mean_differs_from_animal_mean_when_fold_sizes_differ(settings):
    rows = metrics_frame([("a", "a", 0, 0.0), ("b", "b", 1, 1.0), ("c", "c", 1, 1.0)])
    report = build_paper_tables(rows, settings=settings, expected_folds=2)
    stats = report.statistics.loc[report.statistics.Head.eq("regions")].iloc[0]
    assert stats.Mean == 0.5 and stats.FoldSD == pytest.approx(np.sqrt(0.5))
    assert stats.AnimalMean == pytest.approx(2 / 3)
    assert stats.Animals == 3 and stats.Folds == stats.FiniteFolds == 2
    assert report.segmentation.Dice.item() == "0.500 ± 0.707"
    assert report.animal_intervals.loc[report.animal_intervals.Head.eq("regions"), "Dice"].item().startswith("0.667 [")
    assert report.fold_coverage.Complete.item()


def test_scans_are_averaged_within_animal_before_test_set_mean(settings):
    rows = metrics_frame([("rat_a", "a_7", 0, 0.0), ("rat_a", "a_30", 0, 1.0),
                          ("rat_b", "b_7", 0, 0.8)])
    report = build_paper_tables(rows, settings=settings)
    row = report.statistics.loc[report.statistics.Head.eq("regions")].iloc[0]
    assert row.Mean == pytest.approx(0.65)
    assert row.Animals == 2 and row.ObservedScans == 3
    assert np.isnan(row.FoldSD)
    assert report.segmentation.Dice.item() == "0.650"


def test_repeated_test_predictions_do_not_inflate_animal_sample_size(settings):
    rows = metrics_frame([("a", "a", 0, 0.2), ("b", "b", 0, 0.8),
                          ("a", "a", 1, 0.4), ("b", "b", 1, 0.6)], role="test")
    report = build_paper_tables(rows, settings=settings)
    row = report.statistics.loc[report.statistics.Head.eq("regions")].iloc[0]
    assert row.Animals == 2 and row.ObservedScans == 2
    assert row.AnimalMean == row.Mean == pytest.approx(0.5)
    assert row.RepeatedAnimalsAcrossFolds
    with pytest.raises(ValueError, match="multiple outer holdouts"):
        build_paper_tables(rows.assign(DatasetRole="holdout"), settings=settings)


def test_unavailable_and_undefined_values_are_missing_not_zero(settings):
    rows = metrics_frame([("a", "a", 0, 0.8), ("b", "b", 1, np.nan), ("c", "c", 2, np.nan)])
    rows.loc[rows.Subject.eq("b"), "Status"] = "unavailable"
    rows.loc[rows.Subject.eq("c"), "Status"] = "both_empty"
    boundary = rows.copy()
    boundary["Metric"] = "HD95"
    boundary["Unit"] = "mm"
    boundary["Value"] = np.nan
    boundary.loc[boundary.Subject.eq("a"), "Status"] = "prediction_empty"
    report = build_paper_tables(pd.concat([rows, boundary]), settings=settings)
    dice = report.statistics.loc[report.statistics.Head.eq("regions") & report.statistics.Metric.eq("Dice")].iloc[0]
    assert dice.Mean == 0.8 and dice.FiniteFolds == 1 and dice.Folds == 3 and dice.Animals == 1
    assert report.segmentation["HD95 (mm)"].item() == "—"
    assert report.segmentation.BoundaryAnimals.item() == 0
    bad = rows.copy()
    bad.loc[bad.Status.eq("unavailable"), "Value"] = 0
    with pytest.raises(ValueError, match="Unavailable"):
        build_paper_tables(bad, settings=settings)


def test_seeds_roles_conditions_and_variants_remain_separate(settings):
    original = metrics_frame([("a", "a", 0, 0.7)])
    other = original.assign(Variant="candidate", Seed=43, Dataset="D3", Condition="SHAM", DatasetRole="test")
    report = build_paper_tables(pd.concat([original, other]), settings=settings)
    assert len(report.segmentation) == 2
    assert set(report.segmentation.Seed) == {42, 43}
    assert set(report.segmentation.DatasetRole) == {"holdout", "test"}
    with pytest.raises(ValueError, match="Duplicate"):
        build_paper_tables(pd.concat([original, original]), settings=settings)


def test_anatomy_macros_require_complete_annotation_but_roi_rows_keep_partial_labels(settings):
    lesion = metrics_frame([("a", "a", 0, 0.9), ("b", "b", 1, 0.9)])
    anatomy = lesion.assign(Region="Cortex", Label=6, Value=0.5)
    anatomy.loc[anatomy.Subject.eq("b"), ["Value", "Status"]] = [np.nan, "unavailable"]
    report = build_paper_tables(pd.concat([lesion, anatomy]), settings=settings)
    cortex = report.segmentation.loc[report.segmentation.Region.eq("Cortex")].iloc[0]
    assert cortex.Animals == 1
    macro = report.region_sets.loc[report.region_sets.Region.eq("anatomy_only")].iloc[0]
    assert macro.Animals == 1
    assert macro.Dice == "0.500"
    assert report.region_sets.loc[report.region_sets.Region.eq("all_roi"), "Animals"].item() == 1


def test_native_patch_results_are_averaged_as_fold_cohort_measurements(settings):
    rows = metrics_frame([("a", "a", 0, 0.7), ("b", "b", 1, 0.8)])
    patch = pd.DataFrame([{"Variant": "reference", "Seed": 42, "DatasetRole": "holdout", "Dataset": "D1",
                           "Head": "regions", "Region": "Regions", "Metric": "regions_logits_loss",
                           "Unit": "unitless", "Value": value, "Fold": fold,
                           "EvaluationStage": "patch", "Aggregation": "cohort_pooled"}
                          for fold, value in enumerate((0.2, 0.4))])
    report = build_paper_tables(rows, patch_metrics=patch, settings=settings)
    assert report.patches.Mean.item() == pytest.approx(0.3)
    assert report.patches.FoldSD.item() == pytest.approx(np.sqrt(0.02))
    assert "AnimalCILower" not in report.patches


def test_classifier_exact_intervals_use_disjoint_animals_and_also_report_fold_means(settings):
    rows = metrics_frame([("a", "a", 0, 0.7), ("b", "b", 0, 0.8),
                          ("c", "c", 1, 0.7), ("d", "d", 1, 0.8)])
    decisions = rows[["Variant", "Seed", "Fold", "DatasetRole", "Dataset", "Subject", "Scan"]].copy()
    decisions["Reference"] = [0, 1, 0, 1]
    decisions["Prediction"] = [0, 1, 0, 1]
    report = build_paper_tables(rows, classifier_predictions=decisions, settings=settings)
    pooled = report.classification.loc[report.classification.Scope.eq("pooled_holdout")].iloc[0]
    assert pooled.Animals == 4 and pooled.accuracy == 1
    assert 0 < pooled.accuracy_ci_lower < 1
    average = report.classification.loc[report.classification.Scope.eq("equal_fold_mean")].iloc[0]
    assert average.Animals == 4 and average.accuracy == 1 and average.accuracy_fold_sd == 0


def test_volume_agreement_and_paired_model_comparisons_reuse_animal_units(settings):
    rows = metrics_frame([(str(i), str(i), i % 2, 0.5 + i * 0.05) for i in range(4)])
    reference = rows.assign(Metric="ReferenceVolume", Unit="mm3", Value=np.arange(1.0, 5.0))
    predicted = reference.assign(Metric="PredictionVolume", Value=np.arange(1.0, 5.0) + 0.2)
    all_rows = pd.concat([rows, reference, predicted])
    candidate = all_rows.assign(Variant="candidate")
    candidate.loc[candidate.Metric.eq("Dice"), "Value"] += 0.1
    report = build_paper_tables(pd.concat([all_rows, candidate]), settings=replace(settings, reference_variant="reference"))
    assert report.agreement.n.eq(4).all()
    assert report.agreement.mae.to_numpy() == pytest.approx([0.2, 0.2])
    assert report.comparisons.PairedAnimals.eq(4).all()
    assert report.comparisons.MeanDifference.item() == pytest.approx(0.1)
    assert report.comparisons.HolmP.item() >= report.comparisons.WilcoxonP.item()


def test_folder_reader_prefers_run_summary_and_export_verifies_inputs(tmp_path, settings):
    root = tmp_path / "run"
    (root / "summary").mkdir(parents=True)
    (root / "regions").mkdir()
    rows = metrics_frame([("a", "a", 0, 0.7), ("b", "b", 1, 0.8)])
    source = root / "summary/full_volume_metrics.csv"
    rows.drop(columns="Condition").to_csv(source, index=False)
    rows[["Subject", "Scan", "Dataset", "Condition"]].to_csv(root / "subjects.csv", index=False)
    rows.assign(Value=0).to_csv(root / "regions/full_volume_metrics.csv", index=False)
    (root / "splits.json").write_text(json.dumps([{}, {}]))
    report = paper_tables_from_folder(root, settings=settings)
    assert report.segmentation.Dice.item().startswith("0.750")
    assert report.segmentation.Condition.item() == "TBI"
    assert (root / "subjects.csv").resolve().as_posix() in {
        Path(path).as_posix() for path in report.sources.Path
    }
    assert report.fold_coverage.Complete.item()
    output = report.export(tmp_path / "paper")
    assert {"segmentation", "statistics", "fold_metrics", "classification", "agreement"} <= set(pd.ExcelFile(output / "paper_tables.xlsx").sheet_names)
    with pytest.raises(FileExistsError):
        report.export(output)
    with source.open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="changed after analysis"):
        report.export(tmp_path / "changed")


@pytest.mark.parametrize("filename", [
    "01_tbi_mouse_training.ipynb", "02_tbi_mouse_kfold.ipynb", "03_rat_finetuning.ipynb",
    "04_rat_kfold.ipynb", "05_mouse_full_training.ipynb", "06_mouse_full_kfold.ipynb",
    "07_ablation.ipynb", "07f_ablation.ipynb", "08_unet_comparison.ipynb", "09_inference_evaluation.ipynb",
])
def test_final_notebook_cell_reads_saved_results_without_fitting(filename, tmp_path):
    notebook_path = Path(__file__).resolve().parents[1] / "notebook" / filename
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    assert notebook["cells"][-1]["id"] == "paper-summary-table"
    rows = metrics_frame([("a", "a", 0, 0.7), ("b", "b", 1, 0.8)])
    if filename.startswith("04_"):
        rows["DatasetRole"] = "validation"
    rows.to_csv(tmp_path / "full_volume_metrics.csv", index=False)
    source = "".join(notebook["cells"][-1]["source"])
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id == "PAPER_RESULTS_DIR":
                node.value = ast.parse(f"Path({str(tmp_path)!r})", mode="eval").body
            elif node.target.id == "PAPER_BOOTSTRAP_SAMPLES":
                node.value = ast.Constant(100)
    ast.fix_missing_locations(tree)
    namespace = {}
    exec(compile(tree, str(notebook_path), "exec"), namespace)
    assert namespace["paper_table"].Dice.item().startswith("0.750")
    assert not (tmp_path / "paper_tables").exists()
