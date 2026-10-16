"""Exercise shared folds, real-data drafts, dependency recovery and paper exports."""

import json
from dataclasses import replace
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

from src.experiments.revision_config import DraftConfig, RevisionSuiteConfig
from src.experiments.revision_splits import (
    make_suite_splits, project_folds, read_selection, validate_folds,
)
from src.experiments import revision_suite as suite
from src.experiments.revision_summary import export_suite_summary


@pytest.fixture
def revision_settings(tmp_path):
    """Small normalized cohorts with complete/partial anatomy and repeated rat scans."""
    root = tmp_path / "normalized"
    affine = np.diag([0.1, 0.1, 0.1, 1.0])
    rng = np.random.default_rng(23)
    foreground = [1, 2, 3, 5, 6, 12, 13, 15, 16, 21]
    all_names = ["Les", "cl-CC", "cl-V", "cl-Hipp", "cl-Ctx", "il-CC", "il-V", "il-Hipp", "il-Ctx", "tV"]
    for dataset, count in (("D1", 6), ("D3", 6), ("D2", 3), ("D4", 3)):
        for number in range(count):
            subject = f"SUBJ_{number:04d}"
            folder = root / dataset / subject
            folder.mkdir(parents=True)
            timepoints = []
            if dataset == "D2":
                timepoints = ["7days", "30days"]
            sham = dataset == "D3"
            names = ["Les", "cl-V", "il-V", "tV"]
            if dataset in ("D1", "D3") and number % 2 == 0:
                names = list(all_names)
            if sham:
                names = [name for name in names if name != "Les"]
            metadata = {
                "schema_version": 3, "dataset": dataset, "normalized_id": subject,
                "original_id": f"original-{number}", "subgroup": dataset + "-group",
                "condition": "SHAM" if sham else "TBI", "modality": "T1w",
                "species": "rat" if dataset == "D2" else "mouse",
                "annotated_regions": names, "has_lesion_annotation": not sham,
                "timepoints": timepoints,
            }
            (folder / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
            for timepoint in timepoints or [None]:
                scan_dir = folder
                if timepoint:
                    scan_dir = folder / timepoint
                    scan_dir.mkdir()
                labels = np.zeros((16, 16, 16), dtype=np.int16)
                for index, label in enumerate(foreground):
                    if sham and label == 1:
                        continue
                    x = 3 + (index % 4) * 2
                    y = 4 + (index // 4) * 3
                    labels[x:x + 2, y:y + 2, 6:9] = label
                brain = np.zeros_like(labels)
                brain[2:14, 2:14, 2:14] = 1
                arrays = {
                    "N4": rng.normal(size=labels.shape).astype(np.float32),
                    "Labels": labels, "brain_mask": brain,
                }
                for name, array in arrays.items():
                    image = nib.Nifti1Image(array, affine)
                    image.header.set_xyzt_units("mm")
                    nib.save(image, str(scan_dir / f"{name}.nii.gz"))
    reference = root / "D1" / "SUBJ_0000" / "N4.nii.gz"
    return RevisionSuiteConfig(
        data_root=root, reference=reference, output_root=tmp_path / "results",
        folds=2, bootstrap_samples=100,
        draft=DraftConfig(external_animals=3),
    )


def test_draft_rejects_a_single_bottleneck_value_before_fitting(revision_settings):
    draft = replace(revision_settings.draft, patch_size=(16, 16, 16))
    settings = replace(revision_settings, draft=draft)
    with pytest.raises(ValueError, match="more than one bottleneck voxel"):
        suite.prepare_suite(settings, mode="draft")
    assert not (revision_settings.output_root / "draft").exists()


def test_default_draft_has_finite_training_and_validation_on_available_devices(revision_settings):
    """Exercise the real loss and BatchNorm updates, including cuDNN when present."""
    import tensorflow as tf
    from src.experiments.revision_config import experiment_configuration
    from src.experiments.training import build_model

    config = experiment_configuration(
        revision_settings, "d1", output_dir=revision_settings.output_root, mode="draft",
    )
    assert config.preprocessing.patch_size == (32, 32, 32)
    devices = ["/CPU:0"]
    if tf.config.list_physical_devices("GPU"):
        devices.append("/GPU:0")
    rng = np.random.default_rng(42)
    shape = (1, *config.preprocessing.patch_size, 1)
    image = rng.normal(size=shape).astype(np.float32)
    regions = rng.integers(0, len(config.labels.names), size=shape[:-1])
    targets = {
        "regions": tf.one_hot(regions, len(config.labels.names)),
        "brain_mask": (regions > 0).astype(np.float32)[..., None],
    }

    for device in devices:
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(42)
        with tf.device(device):
            model = build_model(config)
            trained = model.train_on_batch(image, targets, return_dict=True)
            validated = model.test_on_batch(image, targets, return_dict=True)
        assert all(np.isfinite(value) for value in trained.values()), (device, trained)
        assert all(np.isfinite(value) for value in validated.values()), (device, validated)
        for layer in model.layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                assert np.prod(layer.input_shape[1:-1]) > 1
                assert np.isfinite(layer.moving_variance.numpy()).all(), (device, layer.name)
    tf.keras.backend.clear_session()


def test_folds_balance_complete_tbi_and_preserve_every_pretraining_role(revision_settings):
    plan = suite.prepare_suite(revision_settings)
    d1 = plan.data["d1"].subjects.subject_ids
    assert project_folds(plan.splits.mouse_full, d1) == plan.splits.d1
    complete = plan.data["mouse_full"].annotation_groups()
    counts = [sum(complete[animal] == "complete" for animal in fold.test_ids) for fold in plan.splits.d1]
    assert max(counts) - min(counts) <= 1
    for fold in plan.splits.rats:
        assert len(fold.train_ids) == len(fold.val_ids) == len(fold.test_ids) == 1
        scans = plan.data["rats"].subjects.scan_keys(fold.test_ids)
        assert len(scans) == 2
    assert not plan.folder.exists()


def test_existing_d1_folds_are_inherited_without_regeneration(revision_settings, tmp_path):
    original = suite.prepare_suite(revision_settings)
    path = tmp_path / "source_folds.json"
    path.write_text(json.dumps([fold.to_dict() for fold in original.splits.d1]))
    settings = replace(revision_settings, folds_file=path, split_seed=17)
    selected = suite.prepare_suite(settings)
    assert selected.splits.d1 == original.splits.d1
    assert project_folds(selected.splits.mouse_full, original.data["d1"].subjects.subject_ids) == original.splits.d1
    draft = suite.prepare_suite(settings, mode="draft")
    assert draft.splits.d1 == original.splits.d1


def test_fixed_split_anchor_is_exact_and_each_animal_is_tested_once(revision_settings, tmp_path):
    original = suite.prepare_suite(revision_settings)
    anchor = original.splits.mouse_full[0].to_dict()
    path = tmp_path / "anchor.json"
    path.write_text(json.dumps(anchor))
    selected = suite.prepare_suite(replace(revision_settings, anchor_split_file=path))
    assert selected.splits.mouse_full[0].to_dict() == anchor
    validate_folds(selected.splits.mouse_full, original.data["mouse_full"].subjects.subject_ids)


def test_provided_membership_order_does_not_create_a_false_leakage_error(revision_settings, tmp_path):
    original = suite.prepare_suite(revision_settings)
    entries = [{role: list(reversed(ids)) for role, ids in fold.to_dict().items()} for fold in original.splits.d1]
    path = tmp_path / "unsorted_folds.json"
    path.write_text(json.dumps(entries))
    selected = suite.prepare_suite(replace(revision_settings, folds_file=path))
    assert [fold.to_dict() for fold in selected.splits.d1] == entries


def test_leaked_or_two_role_split_is_rejected_before_any_output(revision_settings, tmp_path):
    original = suite.prepare_suite(revision_settings)
    values = [fold.to_dict() for fold in original.splits.d1]
    values[0]["train"].append(values[0]["test"][0])
    path = tmp_path / "leak.json"
    path.write_text(json.dumps(values))
    with pytest.raises(ValueError, match="leakage"):
        suite.prepare_suite(replace(revision_settings, folds_file=path))
    values[0].pop("test")
    path.write_text(json.dumps(values))
    with pytest.raises(ValueError, match="separate train, validation and test"):
        read_selection(path)
    assert not original.folder.exists()


def test_external_identity_overlap_is_rejected(revision_settings):
    overrides = {"D4/SUBJ_0000": "D1/SUBJ_0000"}
    with pytest.raises(ValueError, match="External D4 animals overlap"):
        suite.prepare_suite(replace(revision_settings, animal_ids=overrides))


def test_plan_only_records_contract_and_rejects_changed_resumption(revision_settings):
    folder = suite.run_revision_suite(revision_settings, plan_only=True)
    assert (folder / "experiment_plan.csv").is_file()
    assert not (folder / "experiments").exists()
    plan = pd.read_csv(folder / "experiment_plan.csv")
    assert plan.loc[plan["name"].eq("d1_reference"), "Fits"].sum() == 2
    same = suite.run_revision_suite(revision_settings, plan_only=True, resume=True)
    assert same == folder
    changed = replace(revision_settings, mouse_full_epochs=301)
    with pytest.raises(ValueError, match="configuration.*changed"):
        suite.run_revision_suite(changed, plan_only=True, resume=True)
    with pytest.raises(FileExistsError):
        suite.run_revision_suite(revision_settings, plan_only=True)


def test_failure_stops_dependencies_and_records_the_failed_job(revision_settings, monkeypatch):
    settings = replace(revision_settings, include_interpolation=False)
    called = []

    def fail(plan, job, callbacks):
        called.append(job.key)
        raise RuntimeError("intentional failure")

    monkeypatch.setattr(suite, "_execute_job", fail)
    with pytest.raises(RuntimeError, match="intentional failure"):
        suite.run_revision_suite(settings)
    assert called == ["d1_reference/seed_42"]
    path = settings.output_root / "full" / settings.run_name / "state.json"
    state = json.loads(path.read_text())
    assert state["status"] == "interrupted"
    assert "intentional failure" in state["jobs"][called[0]]["error"]


def test_refit_sources_need_an_explicit_training_only_contract(revision_settings, tmp_path):
    from src.experiments.config import InitializationConfig
    from src.experiments.initialization import pretraining_audit

    plan = suite.prepare_suite(revision_settings)
    data = plan.data["d1"]
    folder = tmp_path / "refit_source"
    fold = folder / "fold_0"
    fold.mkdir(parents=True)
    checkpoint = fold / "best.weights.h5"
    checkpoint.write_bytes(b"audit does not load the weights")
    ids = data.subjects.subject_ids
    (fold / "split.json").write_text(json.dumps({"train": ids, "validation": []}))
    data.subjects.manifest(ids).to_csv(folder / "subjects.csv", index=False)
    training = replace(data.config.training, monitor="loss")
    config = replace(data.config, training=training)
    (folder / "experiment.json").write_text(json.dumps(config.to_dict()))
    settings = InitializationConfig(mode="weights", checkpoint=checkpoint)
    with pytest.raises(ValueError, match="checkpoint-selection"):
        pretraining_audit(data, settings, [])
    (folder / "refit.json").write_text(json.dumps({
        "purpose": "Deployment refit; no independent evaluation", "epochs": training.epochs,
    }))
    audit = pretraining_audit(data, settings, [])
    assert str(folder / "refit.json") in audit["sha256"]
    with pytest.raises(ValueError, match="Pretraining leakage"):
        pretraining_audit(data, settings, [ids[0]])
    bad = replace(config, training=replace(training, monitor="val_loss"))
    (folder / "experiment.json").write_text(json.dumps(bad.to_dict()))
    with pytest.raises(ValueError, match="training-only deployment"):
        pretraining_audit(data, settings, [])


@pytest.fixture
def eager_tensorflow():
    """Avoid repeated graph compilation in the many-model CPU smoke test only."""
    import tensorflow as tf

    previous = tf.config.functions_run_eagerly()
    tf.config.run_functions_eagerly(True)
    yield
    tf.config.run_functions_eagerly(previous)


def test_partial_d4_annotations_do_not_invent_an_anatomy_average():
    from src.evaluation.paper_tables import PaperTables
    from src.experiments.revision_summary import _main_rows

    rows = pd.DataFrame([{
        "Variant": "release", "Seed": 42, "DatasetRole": "external", "Dataset": "D4",
        "Condition": "TBI", "Head": "regions", "Region": "Lesion", "Animals": 16,
        "Folds": 1, "Dice": "0.900", "HD95 (mm)": "0.300",
    }])
    result = _main_rows(PaperTables(segmentation=rows), "D4")
    assert result["AnatomyDice"].tolist() == [""]
    assert result["CompleteAnatomyAnimals"].tolist() == [0]
    assert result["LesionDice"].tolist() == ["0.900"]


def test_interrupted_cv_can_report_committed_folds_before_summary_csv_exists(revision_settings):
    folder = suite.run_revision_suite(revision_settings, plan_only=True)
    contract = json.loads((folder / "suite.json").read_text())
    subject = contract["splits"]["d1"][0]["test"][0]
    fold = folder / "experiments/d1_reference/seed_42/fold_0"
    fold.mkdir(parents=True)
    metrics = pd.DataFrame([{
        "Subject": subject, "Scan": subject.split("/")[-1], "Dataset": "D1",
        "Condition": "TBI", "Head": "regions", "Region": "Lesion", "Label": 1,
        "Metric": "Dice", "Value": 0.9, "Unit": "fraction", "Status": "ok", "Seed": 42,
    }])
    metrics.to_json(fold / "metrics.json", orient="table")
    (fold / "completed.json").write_text(json.dumps({
        "evaluations": [{"path": "metrics.json", "role": "holdout"}], "patch_metrics": None,
    }))
    paper = export_suite_summary(folder, bootstrap_samples=100)
    coverage = pd.read_csv(paper.parent / "audit/coverage.csv")
    source = coverage.loc[coverage["Job"].eq("d1_reference")].iloc[0]
    assert source["ObservedFolds"] == 1 and source["ExpectedFolds"] == 2
    table = pd.read_csv(paper / "main_results.csv")
    assert table["Folds"].tolist() == [1]
    assert table["Animals"].tolist() == [1]


def test_complete_draft_fits_every_stage_and_resumes_without_fitting(revision_settings, monkeypatch, eager_tensorflow):
    """Actual tiny CPU fits, backbone transfer, deployment refits and D4 inference."""
    folder = suite.run_revision_suite(revision_settings, mode="draft")
    state = json.loads((folder / "state.json").read_text())
    assert state["status"] == "complete" and state["external_complete"]
    assert len(state["jobs"]) == 16
    for item in state["jobs"].values():
        run = Path(item["directory"])
        for file in run.glob("fold_*/training_state.json"):
            saved = json.loads(file.read_text())
            assert saved["completed_epochs"] == 1
    draft = folder / "draft_data"
    assert (draft / "draft.json").is_file()
    metadata = json.loads((draft / "D2/SUBJ_0000/metadata.json").read_text())
    assert metadata["timepoints"] == ["7days", "30days"]
    paper = Path(state["paper_summary"])
    workbook = pd.ExcelFile(paper / "paper_tables.xlsx")
    assert workbook.sheet_names == ["main_results", "ablations", "per_region", "validation"]
    assert len(list(paper.glob("*.csv"))) == 4
    main = pd.read_csv(paper / "main_results.csv")
    assert main.DRAFT.all()
    assert {"d1", "mouse_full", "rats", "D4"}.issubset(set(main.Task))
    for index in range(2):
        path = folder / f"experiments/mouse_full_reference/seed_42/fold_{index}/initialization.json"
        assert path.is_file()
    external = folder / "external/D4/pretraining_audit.json"
    assert json.loads(external.read_text())["sha256"]
    release = folder / "experiments/release_mouse_full/seed_42"
    split = json.loads((release / "splits.json").read_text())
    assert not split["validation"] and not split["test"]
    assert len(split["train"]) == 12

    import tensorflow as tf

    def forbidden(*args, **kwargs):
        pytest.fail("Completed fits must be reused")

    monkeypatch.setattr(tf.keras.Model, "fit", forbidden)
    resumed = suite.run_revision_suite(revision_settings, mode="draft", resume=True)
    assert resumed == folder
    another = export_suite_summary(folder, bootstrap_samples=100)
    assert another != paper and (another / "paper_tables.xlsx").is_file()
