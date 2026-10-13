"""Validate matching, offline evaluation and independent-animal revision reports."""

import json
from dataclasses import replace

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

from test_experiments import experiment
from src.experiments import ablation
from src.experiments.config import ExperimentConfig, ModelConfig
from src.experiments.comparison import ModelVariant, comparison_plan, run_model_comparison
from src.experiments.data import make_split, prepare_data
from src.evaluation.k_fold import TrainingResult
from src.evaluation.revision import (
    MetricSource, RevisionConfig, export_revision, load_metric_sources, revision_report,
)


def model_variants(config):
    """Existing original-width options with the small synthetic fixture widths."""
    rnet = ModelConfig(filters=config.model.filters)
    unet = replace(rnet, architecture="unet", residual=False, attention=False)
    return [ModelVariant("3drnet", rnet), ModelVariant("unet", unet)]


def test_configuration_roundtrips_saved_json(experiment, tmp_path):
    path = tmp_path / "experiment.json"
    path.write_text(json.dumps(experiment.to_dict()), encoding="utf-8")
    restored = ExperimentConfig.load(path)
    assert restored == experiment
    assert restored.fingerprint() == experiment.fingerprint()
    restored.validate()


def test_model_comparison_uses_shared_runners_and_recovery(experiment, tmp_path, monkeypatch):
    data = prepare_data(experiment)
    split = make_split(data.subjects, seed=42)
    variants = model_variants(experiment)
    received = []

    def runner(prepared, selection, *, resume_run):
        received.append((prepared, selection, resume_run))
        path = prepared.config.training.start_run() if resume_run is None else resume_run
        return TrainingResult(run_dir=path)

    monkeypatch.setattr(ablation, "run_training", runner)
    folder = tmp_path / "comparison"
    run_model_comparison(data, variants, study_dir=folder, split=split, seeds=(11, 12))
    assert len(received) == 4
    for prepared, selection, resume_run in received:
        assert prepared.loader is data.loader
        assert selection is split
        assert prepared.config.preprocessing == experiment.preprocessing
        assert prepared.config.loss == experiment.loss
        assert prepared.config.training.epochs == experiment.training.epochs
        assert resume_run is None
    run_model_comparison(data, variants, study_dir=folder, split=split, seeds=(11, 12), resume=True)
    assert all(path is not None for _, _, path in received[4:])


def test_comparison_rejects_nonconventional_or_unmatched_baselines(experiment):
    variants = model_variants(experiment)
    assert comparison_plan(experiment, variants)["Architecture"].tolist() == ["3drnet", "unet"]
    changed = replace(variants[1], model=replace(variants[1].model, attention=True))
    with pytest.raises(ValueError, match="conventional"):
        comparison_plan(experiment, [variants[0], changed])
    changed = replace(variants[1], model=replace(variants[1].model, filters=(4, 8, 16, 32, 64)))
    with pytest.raises(ValueError, match="filter widths"):
        comparison_plan(experiment, [variants[0], changed])


def test_study_selection_reloads_saved_fixed_and_cv_animals(experiment, tmp_path):
    data = prepare_data(experiment)
    folder = tmp_path / "study"
    folder.mkdir()
    selection = ablation.study_selection(data, mode="fixed_split", study_dir=folder)
    selection.split.save(folder / "splits.json")
    restored = ablation.study_selection(data, mode="fixed_split", study_dir=folder, resume=True)
    assert restored.split == selection.split

    config = replace(experiment, training=replace(experiment.training, k=2))
    selection = ablation.study_selection(replace(data, config=config), mode="cross_validation", study_dir=folder)
    payload = [fold.to_dict() for fold in selection.folds]
    (folder / "splits.json").write_text(json.dumps(payload), encoding="utf-8")
    restored = ablation.study_selection(
        replace(data, config=config), mode="cross_validation", study_dir=folder, resume=True,
    )
    assert restored.folds == selection.folds


@pytest.mark.parametrize("roi_loss", ["dice_ce", "dice_ce_gradient_fix"])
def test_saved_model_evaluates_longitudinal_scans_once(experiment, tmp_path, roi_loss):
    from src.experiments.training import build_model, configure_runtime, export_model
    from src.evaluation.inference import (
        SavedModelEvaluationConfig, evaluate_saved_model, prediction_slice_figure,
    )

    experiment = replace(experiment, loss=replace(experiment.loss, regions=roi_loss))
    if roi_loss == "dice_ce":
        with pytest.raises(ValueError, match="partial annotations"):
            prepare_data(experiment)

    configure_runtime()
    model = build_model(experiment)
    model_path = tmp_path / "model.keras"
    export_model(model, model_path)
    config_path = tmp_path / "experiment.json"
    config_path.write_text(json.dumps(experiment.to_dict()), encoding="utf-8")
    settings = SavedModelEvaluationConfig(
        experiment_file=config_path, model_file=model_path, output_dir=tmp_path / "evaluation",
        subject_ids=("D1/SUBJ_0003",), role="test", variant="unet",
    )
    result = evaluate_saved_model(settings)
    assert set(result.metrics["Scan"]) == {"SUBJ_0003_7days", "SUBJ_0003_14days"}
    assert set(result.metrics["DatasetRole"]) == {"test"}
    assert len(result.timings) == len(result.manifest) == 2
    assert set(result.metrics.loc[result.metrics["Region"] == "CC", "Status"]) == {"unavailable"}
    source = experiment.data.root / "D1" / "SUBJ_0003" / "7days" / "N4.nii.gz"
    prediction = result.output_dir / "predictions" / "D1_T1w" / "SUBJ_0003_7days" / "regions.nii.gz"
    original = nib.load(source)
    restored = nib.load(prediction)
    assert original.shape == restored.shape
    np.testing.assert_array_equal(original.affine, restored.affine)
    figure = prediction_slice_figure(source, prediction)
    assert len(figure.data) == 3
    with pytest.raises(FileExistsError):
        evaluate_saved_model(settings)


@pytest.fixture
def metric_source(tmp_path):
    """Three animals, two scans, two seeds and two models with a known difference."""
    rows = []
    for variant, offset in [("3drnet", 0.0), ("unet", 0.1)]:
        for subject in range(3):
            for scan in range(2):
                for seed in (11, 12):
                    measures = {"Dice": 0.5 + subject / 10 + scan / 10 + offset,
                                "ReferenceVolume": float(subject + scan + 1),
                                "PredictionVolume": float(subject + scan + 1) + offset}
                    for metric, value in measures.items():
                        unit = "unitless"
                        if metric.endswith("Volume"):
                            unit = "mm3"
                        rows.append({"Variant": variant, "Seed": seed, "Fold": 0,
                                     "DatasetRole": "test", "Dataset": "D2", "Region": "Lesion",
                                     "Subject": f"rat{subject}", "Scan": f"rat{subject}_{scan}",
                                     "Metric": metric, "Value": value, "Unit": unit, "Status": "ok"})
    # An unannotated structure never contributes a zero to a summary.
    rows.append({**rows[0], "Region": "CC", "Value": np.nan, "Status": "unavailable"})
    path = tmp_path / "metrics.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return MetricSource(path=path)


def test_revision_aggregates_animals_not_scans_seeds_or_fold_models(metric_source, tmp_path):
    config = RevisionConfig(metrics=("Dice",), bootstrap_samples=100)
    report = revision_report([metric_source], config, comparison=("3drnet", "unet"))
    assert len(report.animals) == 6
    assert set(report.animals["Scans"]) == {2}
    assert set(report.summary["Animals"]) == {3}
    assert report.animals.loc[report.animals["Variant"] == "3drnet", "Value"].tolist() == pytest.approx([0.55, 0.65, 0.75])
    assert len(report.volumes) == 6
    assert report.volumes.loc[report.volumes["Variant"] == "3drnet", "Manual"].tolist() == [1.5, 2.5, 3.5]
    assert report.agreement["n"].tolist() == [3, 3]
    assert report.comparisons["PairedAnimals"].tolist() == [3]
    assert report.comparisons["MeanDifference"].iloc[0] == pytest.approx(0.1)
    assert report.comparisons["CILower"].iloc[0] == pytest.approx(0.1)
    assert report.comparisons["HolmP"].iloc[0] >= report.comparisons["WilcoxonP"].iloc[0]
    assert "unavailable" in set(report.statuses["Status"])
    exported = export_revision(report, tmp_path / "export", config)
    assert (exported / "animal_volume_pairs.csv").is_file()
    with pytest.raises(FileExistsError):
        export_revision(report, exported, config)


def test_revision_pairs_common_scans_and_preserves_zeros(metric_source):
    table = pd.read_csv(metric_source.path)
    drop = (table["Variant"] == "unet") & (table["Subject"] == "rat0") & (table["Scan"] == "rat0_1")
    table = table.loc[~drop]
    zeros = (table["Subject"] == "rat0") & table["Metric"].str.endswith("Volume")
    table.loc[zeros, "Value"] = 0.0
    table.to_csv(metric_source.path, index=False)
    config = RevisionConfig(metrics=("Dice",), bootstrap_samples=100)
    report = revision_report([metric_source], config, comparison=("3drnet", "unet"))
    # Compare rat0's same available scan, rather than unmatched longitudinal means.
    assert report.comparisons["MeanDifference"].iloc[0] == pytest.approx(0.1)
    assert report.comparisons["PairedScans"].iloc[0] == 5
    assert set(report.volumes.loc[report.volumes["Subject"] == "rat0", "Manual"]) == {0.0}


def test_revision_rejects_duplicates_and_changed_reference_volumes(metric_source):
    with pytest.raises(ValueError, match="Duplicate"):
        load_metric_sources([metric_source, metric_source])
    table = pd.read_csv(metric_source.path)
    changed = (table["Variant"] == "unet") & (table["Metric"] == "ReferenceVolume")
    table.loc[changed, "Value"] += 1
    table.to_csv(metric_source.path, index=False)
    with pytest.raises(ValueError, match="different reference volumes"):
        revision_report([metric_source], RevisionConfig(bootstrap_samples=100))


def test_revision_requires_roles_and_metadata_and_never_scores_unavailable(metric_source):
    config = RevisionConfig(roles=("holdout",), bootstrap_samples=100)
    with pytest.raises(ValueError, match="roles"):
        revision_report([metric_source], config)
    table = pd.read_csv(metric_source.path).drop(columns="Seed")
    table.to_csv(metric_source.path, index=False)
    with pytest.raises(ValueError, match="Seed"):
        load_metric_sources([metric_source])
    supplied = replace(metric_source, seed=42)
    table.loc[table["Status"] == "unavailable", "Value"] = 0.0
    table.to_csv(metric_source.path, index=False)
    with pytest.raises(ValueError, match="Unavailable"):
        load_metric_sources([supplied])
