"""Check matched settings, routing, recovery contracts and animal aggregation."""

import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from test_experiments import experiment
from src.experiments import ablation
from src.experiments.config import InferenceConfig, ModelConfig
from src.experiments.data import make_cv_splits, make_split, prepare_data
from src.evaluation.k_fold import FoldTiming, TrainingResult


@pytest.fixture
def prepared(experiment):
    model = ModelConfig(filters=(2, 4, 8, 16, 32))
    training = replace(experiment.training, data_multiplier=3, k=2)
    preprocessing = replace(experiment.preprocessing, augment=True, random_crop=True)
    config = replace(experiment, model=model, training=training, preprocessing=preprocessing)
    return prepare_data(config)


def test_variants_change_only_the_requested_setting(prepared):
    base = prepared.config
    variants = [ablation.AblationVariant("reference"), ablation.AblationVariant("no_attention", "attention", False)]
    plan = ablation.ablation_plan(base, variants)
    assert plan["Category"].tolist() == ["reference", "architecture"]
    changed = variants[1].configuration(base)
    assert changed.model == replace(base.model, attention=False)
    assert changed.preprocessing == base.preprocessing
    assert changed.training == base.training
    assert changed.loss == base.loss
    assert changed.labels == base.labels
    assert base.model.attention


@pytest.mark.parametrize("variant", [
    ablation.AblationVariant("same", "residual", True),
    ablation.AblationVariant("bad_bool", "attention", 0),
    ablation.AblationVariant("bad_count", "data_multiplier", True),
    ablation.AblationVariant("unknown", "deep_supervision", False),
    ablation.AblationVariant("../escape"),
])
def test_invalid_or_ineffective_variants_fail_before_running(prepared, variant):
    with pytest.raises(ValueError):
        variant.configuration(prepared.config)


def test_matched_runs_share_data_splits_seeds_and_resume_paths(prepared, tmp_path, monkeypatch):
    split = make_split(prepared.subjects, seed=42)
    variants = [ablation.AblationVariant("reference"), ablation.AblationVariant("no_attention", "attention", False)]
    calls = []

    def runner(data, selection, *, resume_run=None):
        config = data.config
        path = config.training.start_run() if resume_run is None else resume_run
        assert path.is_dir()
        calls.append((data, selection, resume_run))
        table = pd.DataFrame([{
            "Fold": 0, "DatasetRole": "test", "Subject": split.test[0], "Scan": "scan",
            "Dataset": "D1", "Region": "Lesion", "Metric": "Dice", "Value": 0.5,
            "Unit": "unitless", "Status": "ok",
        }])
        return TrainingResult(run_dir=path, evaluations={"regions": [table]}, timings=[FoldTiming(fold=0)])

    monkeypatch.setattr(ablation, "run_training", runner)
    folder = tmp_path / "study"
    result = ablation.run_ablation(prepared, variants, study_dir=folder, split=split, seeds=(11, 22))
    assert len(result.runs) == 4
    assert result.metrics["Seed"].unique().tolist() == [11, 22]
    for data, selection, resume_run in calls:
        assert data.loader is prepared.loader
        assert data.subjects is prepared.subjects
        assert selection is split
        assert data.config.split_seed == prepared.config.split_seed
        assert data.config.preprocessing == prepared.config.preprocessing
        assert resume_run is None
    assert (folder / "full_volume_metrics.csv").is_file()
    assert len(json.loads((folder / "runs.json").read_text())) == 4

    ablation.run_ablation(prepared, variants, study_dir=folder, split=split, seeds=(11, 22), resume=True)
    assert all(path is not None for _, _, path in calls[4:])
    with pytest.raises(ValueError, match="Resume study"):
        ablation.run_ablation(prepared, variants, study_dir=folder, split=split, seeds=(22, 11), resume=True)


def test_cv_routes_identical_outer_folds_to_every_variant(prepared, tmp_path, monkeypatch):
    folds = make_cv_splits(prepared.subjects.subject_ids, k=2, seed=13)
    variants = [ablation.AblationVariant("reference"), ablation.AblationVariant("no_residual", "residual", False)]
    received = []

    def runner(data, *, folds, resume_run):
        received.append(folds)
        return TrainingResult(run_dir=data.config.training.start_run())

    monkeypatch.setattr(ablation, "run_cross_validation", runner)
    ablation.run_ablation(prepared, variants, study_dir=tmp_path / "cv", mode="cross_validation", folds=folds)
    assert received == [folds, folds]


def test_animal_summary_retains_model_identity_and_ignores_missing_values(tmp_path):
    common = {"Variant": "reference", "Seed": 42, "Fold": 0, "DatasetRole": "test",
              "Subject": "rat", "Dataset": "D2", "Region": "Lesion", "Metric": "Dice", "Unit": "unitless"}
    table = pd.DataFrame([
        {**common, "Scan": "7days", "Value": 0.4},
        {**common, "Scan": "14days", "Value": 0.8},
        {**common, "Scan": "30days", "Value": np.nan},
        {**common, "Seed": 43, "Scan": "7days", "Value": 0.2},
    ])
    result = ablation.AblationResult(study_dir=tmp_path, metrics=table)
    summary = ablation.animal_metric_values(result)
    assert len(summary) == 2
    first = summary.loc[summary["Seed"] == 42]
    assert first["Value"].item() == pytest.approx(0.6)
    assert first["Scans"].item() == 2


def test_tiny_real_study_uses_core_training_and_reuses_completed_runs(prepared, tmp_path):
    """Fit two tiny architecture variants and recover the complete study."""
    preprocessing = replace(prepared.config.preprocessing, augment=False)
    inference = InferenceConfig(enabled=False, stride=16)
    config = replace(prepared.config, preprocessing=preprocessing, inference=inference)
    data = replace(prepared, config=config)
    split = make_split(data.subjects, seed=42)
    variants = [ablation.AblationVariant("reference"), ablation.AblationVariant("no_attention", "attention", False)]
    folder = tmp_path / "real_study"
    result = ablation.run_ablation(data, variants, study_dir=folder, split=split)
    assert len(result.runs) == 2
    assert result.timings.shape[0] == 2
    assert result.metrics.empty
    for run in result.runs.values():
        assert (run.run_dir / "fold_0" / "completed.json").is_file()
        assert np.isfinite(run.histories[0]["loss"]).all()

    resumed = ablation.run_ablation(data, variants, study_dir=folder, split=split, resume=True)
    for name, run in result.runs.items():
        assert resumed.runs[name].histories == run.histories
