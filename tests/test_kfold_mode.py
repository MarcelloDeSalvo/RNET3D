"""Exercise the notebook switch, full-fold selection and source checkpoint reuse."""

import json
import re
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from test_experiments import experiment
from test_pretraining_splits import cv_sources
from src.experiments.config import InferenceConfig
from src.experiments.data import prepare_data, resolve_training_selection
from src.experiments.initialization import cv_initializations_from_run
from src.experiments import training


@pytest.mark.parametrize("enabled", [False, True])
def test_switch_selects_fixed_roles_or_every_outer_fold(experiment, enabled):
    config = replace(experiment, training=replace(experiment.training, k=2))
    data = prepare_data(config)

    selected = resolve_training_selection(data, k_fold=enabled)

    if enabled:
        assert selected.split is None
        assert len(selected.folds) == 2
        held_out = [animal for fold in selected.folds for animal in fold.test_ids]
        assert sorted(held_out) == data.subjects.subject_ids
        assert selected.train_ids == tuple(selected.folds[0].train_ids)
        assert selected.validation_ids == tuple(selected.folds[0].val_ids)
        assert selected.evaluation_role == "holdout"
    else:
        assert selected.folds is None
        assert selected.split == data.make_split()
        assert selected.train_ids == selected.split.train
        assert selected.validation_ids == selected.split.validation
        assert selected.evaluation_role == "test"


def test_saved_folds_are_reused_and_wrong_mode_is_rejected(experiment, tmp_path):
    config = replace(experiment, training=replace(experiment.training, k=2))
    data = prepare_data(config)
    selected = resolve_training_selection(data, k_fold=True)
    path = tmp_path / "splits.json"
    path.write_text(json.dumps([fold.to_dict() for fold in selected.folds]))

    resumed = resolve_training_selection(data, k_fold=True, resume_run=tmp_path)
    assert resumed.folds == selected.folds
    assert resolve_training_selection(data, k_fold=True, fold_file=path).folds == selected.folds
    with pytest.raises(ValueError, match="FOLD_FILE requires"):
        resolve_training_selection(data, fold_file=path)
    with pytest.raises(ValueError, match="SPLIT_FILE belongs"):
        resolve_training_selection(data, k_fold=True, split_file=path)
    with pytest.raises(ValueError, match="must be boolean"):
        resolve_training_selection(data, k_fold=1)


def test_all_source_fold_snapshots_are_selected_without_fitting(experiment, tmp_path):
    config = replace(experiment, training=replace(experiment.training, k=2))
    data = prepare_data(config)
    _, roles = cv_sources(data, tmp_path / "source")

    sources = cv_initializations_from_run(tmp_path / "source", k=2, freeze_backbone=True)
    target = replace(data, config=replace(config, initialization_by_fold=sources))
    selected = resolve_training_selection(target, k_fold=True)

    assert set(sources) == {0, 1}
    assert all(settings.freeze_backbone for settings in sources.values())
    for index, fold in enumerate(selected.folds):
        assert fold.test_ids == roles[index]["test"]
        assert sources[index].checkpoint.parent.name == f"fold_{index}"
    sources[1].checkpoint.unlink()
    with pytest.raises(ValueError, match="no committed checkpoint for fold 1"):
        cv_initializations_from_run(tmp_path / "source", k=2)


@pytest.mark.parametrize("filename", [
    "01_tbi_mouse_training.ipynb", "03_rat_finetuning.ipynb", "05_mouse_full_training.ipynb",
])
@pytest.mark.parametrize("enabled", [False, True])
def test_training_notebook_switch_routes_the_complete_selection(experiment, filename, enabled, monkeypatch, tmp_path):
    config = replace(experiment, training=replace(experiment.training, k=2))
    data = prepare_data(config)
    root = Path(__file__).resolve().parents[1]
    cells = json.loads((root / "notebook" / filename).read_text(encoding="utf-8"))["cells"]
    sources = ["".join(cell["source"]) for cell in cells if cell["cell_type"] == "code"]
    prepare_cell = next(source for source in sources if "selection = resolve_training_selection(" in source)
    run_cell = next(source for source in sources if "result = run_training(" in source)
    context = {
        "config": config, "K_FOLD_COMPATIBLE": enabled, "RUN_TRAINING": True,
        "RESUME_RUN": None, "SPLIT_FILE": None, "FOLD_FILE": None,
        "REFERENCE_SPLITS": None, "REFERENCE_FOLD": 0,
        "USE_TBI_PRETRAINING": False, "callbacks": [],
    }
    from src.experiments import data as data_module
    monkeypatch.setattr(data_module, "prepare_data", lambda settings: data)
    exec(prepare_cell, context)
    calls = []

    def fixed(**kwargs):
        calls.append(("fixed", kwargs))
        return SimpleNamespace(run_dir=tmp_path / "fixed")

    def cv(**kwargs):
        calls.append(("cv", kwargs))
        return SimpleNamespace(run_dir=tmp_path / "cv")

    monkeypatch.setattr(training, "run_training", fixed)
    monkeypatch.setattr(training, "run_cross_validation", cv)
    # Rat fine-tuning requires a source; this routing check never loads one.
    if filename.startswith("03"):
        context["config"] = SimpleNamespace(initialization=SimpleNamespace(mode="weights"))
    exec(run_cell, context)

    assert len(calls) == 1
    mode, received = calls[0]
    assert received["callbacks"] is context["callbacks"]
    if enabled:
        assert mode == "cv"
        assert len(received["folds"]) == 2
        assert context["evaluation_role"] == "holdout"
    else:
        assert mode == "fixed"
        assert received["split"] == data.make_split()
        assert context["evaluation_role"] == "test"


def test_tiny_cv_run_saves_every_fold_for_later_pretraining(experiment):
    config = replace(
        experiment, training=replace(experiment.training, k=2),
        inference=InferenceConfig(enabled=False, stride=16),
    )
    data = prepare_data(config)
    selected = resolve_training_selection(data, k_fold=True)

    result = training.run_cross_validation(data, folds=selected.folds, callbacks=[])
    sources = cv_initializations_from_run(result.run_dir, k=2)

    assert len(result.histories) == 2
    for index, settings in sources.items():
        assert settings.checkpoint.is_file()
        assert (result.run_dir / f"fold_{index}" / "completed.json").is_file()
    saved = json.loads((result.run_dir / "splits.json").read_text(encoding="utf-8"))
    assert len(saved) == 2
    assert sorted(animal for fold in saved for animal in fold["test"]) == data.subjects.subject_ids


@pytest.mark.parametrize("filename", [
    "01_tbi_mouse_training.ipynb", "03_rat_finetuning.ipynb", "05_mouse_full_training.ipynb",
    "07_ablation.ipynb", "07f_ablation.ipynb", "08_unet_comparison.ipynb",
])
@pytest.mark.parametrize("enabled", [False, True])
def test_configuration_cell_exposes_and_applies_the_switch(experiment, filename, enabled, tmp_path):
    root = Path(__file__).resolve().parents[1]
    cells = json.loads((root / "notebook" / filename).read_text(encoding="utf-8"))["cells"]
    code = next("".join(cell["source"]) for cell in cells if "training = TrainingConfig(" in "".join(cell["source"]))
    assert "K_FOLD_COMPATIBLE: bool = False" in code
    code = code.replace("K_FOLD_COMPATIBLE: bool = False", f"K_FOLD_COMPATIBLE: bool = {enabled}")
    code = re.sub(r"DATA_ROOT: Path = [^\n]+", "DATA_ROOT: Path = dataset_root", code)
    code = re.sub(r"TBI_RUN: Path \| None = [^\n]+", "TBI_RUN: Path | None = None", code)
    context = {
        "Path": Path, "REPO_ROOT": tmp_path, "TRAINING_SEED": 42,
        "REFERENCE": experiment.preprocessing.reference, "dataset_root": experiment.data.root,
    }

    exec(code, context)

    if filename == "01_tbi_mouse_training.ipynb":
        if enabled:
            assert context["SPLIT_FILE"] is None
        else:
            baseline_file = experiment.data.root.parent / "D1_fixed_split_from_kfold_fold_0.json"
            assert context["SPLIT_FILE"] == baseline_file

    if filename.startswith(("07_", "07f_", "08_")):
        expected = "fixed_split"
        if enabled:
            expected = "cross_validation"
        assert context["MODE"] == expected
    else:
        expected_role = "test"
        if enabled:
            expected_role = "holdout"
        assert context["evaluation_role"] == expected_role
        assert context["config"].training.model_name.endswith("_kfold") == enabled


@pytest.mark.parametrize("filename", [
    "01_tbi_mouse_training.ipynb", "02_tbi_mouse_kfold.ipynb", "05_mouse_full_training.ipynb",
    "06_mouse_full_kfold.ipynb", "07_ablation.ipynb", "07f_ablation.ipynb", "08_unet_comparison.ipynb",
])
def test_mouse_notebook_defaults_do_not_resolve_pretrained_weights(experiment, filename, tmp_path, monkeypatch):
    from src.experiments.config import InitializationConfig

    path = Path(__file__).resolve().parents[1] / "notebook" / filename
    cells = json.loads(path.read_text(encoding="utf-8"))["cells"]
    code = next("".join(cell["source"]) for cell in cells if "training = TrainingConfig(" in "".join(cell["source"]))
    code = re.sub(r"DATA_ROOT: Path = [^\n]+", "DATA_ROOT: Path = dataset_root", code)

    def forbidden(*args, **kwargs):
        pytest.fail("Default mouse configurations must not load a source checkpoint")

    monkeypatch.setattr(InitializationConfig, "from_run", staticmethod(forbidden))
    context = {
        "Path": Path, "REPO_ROOT": tmp_path, "TRAINING_SEED": 42,
        "REFERENCE": experiment.preprocessing.reference, "dataset_root": experiment.data.root,
    }
    exec(code, context)
    config = context["config"]
    assert config.initialization.mode == "scratch"
    assert not config.initialization_by_fold
    if filename in ("06_mouse_full_kfold.ipynb", "07f_ablation.ipynb"):
        assert "USE_TBI_PRETRAINING" not in context


def test_notebook_05_pretraining_is_optional_and_fixed_mode_only(experiment, tmp_path, monkeypatch):
    from src.experiments.config import InitializationConfig

    path = Path(__file__).resolve().parents[1] / "notebook/05_mouse_full_training.ipynb"
    cells = json.loads(path.read_text(encoding="utf-8"))["cells"]
    code = next("".join(cell["source"]) for cell in cells if "training = TrainingConfig(" in "".join(cell["source"]))
    code = re.sub(r"DATA_ROOT: Path = [^\n]+", "DATA_ROOT: Path = dataset_root", code)
    code = re.sub(r"TBI_RUN: Path \| None = [^\n]+", "TBI_RUN: Path | None = source_run", code)
    code = code.replace("USE_TBI_PRETRAINING: bool = False", "USE_TBI_PRETRAINING: bool = True")
    checkpoint = tmp_path / "source.weights.h5"
    checkpoint.write_bytes(b"Configuration check; no model construction")
    sources = []

    def source(**kwargs):
        sources.append(kwargs)
        return InitializationConfig(mode="backbone", checkpoint=checkpoint)

    monkeypatch.setattr(InitializationConfig, "from_run", staticmethod(source))
    context = {
        "Path": Path, "REPO_ROOT": tmp_path, "TRAINING_SEED": 42, "source_run": tmp_path,
        "REFERENCE": experiment.preprocessing.reference, "dataset_root": experiment.data.root,
    }
    exec(code, context)
    assert context["config"].initialization.mode == "backbone"
    assert sources[0]["fold"] == 0
    cv_code = code.replace("K_FOLD_COMPATIBLE: bool = False", "K_FOLD_COMPATIBLE: bool = True")
    with pytest.raises(ValueError, match="optional TBI pretraining is for fixed-mode notebook 05 only"):
        exec(cv_code, context)
    assert len(sources) == 1
