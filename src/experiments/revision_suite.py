"""Sequential reviewer experiments using the repository's existing fit runners."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np
import pandas as pd

from .config import DataConfig, DatasetSpec, ExperimentConfig, InitializationConfig
from .data import PreparedData, SubjectSplit, prepare_data
from .revision_config import RevisionSuiteConfig, experiment_configuration
from .revision_draft import balanced_sample, validate_draft_sources, write_draft_dataset
from .revision_splits import (
    SuiteSplits, make_suite_splits, read_selection, select_animals, split_membership,
)
from ..evaluation.k_fold import FoldSplit


@dataclass(frozen=True)
class SuiteJob:
    """One variant/seed, potentially containing all outer folds."""

    name: str
    task: Literal["d1", "mouse_full", "rats"]
    group: str
    seed: int
    variant: str = "reference"
    kind: Literal["cross_validation", "fixed_split", "refit"] = "cross_validation"
    source: str | None = None
    order: int = 1

    @property
    def key(self) -> str:
        """Return the stable dependency and output-directory key."""
        return f"{self.name}/seed_{self.seed}"


@dataclass
class SuitePlan:
    """Resolved data, shared folds and ordered dependencies before any fit."""

    settings: RevisionSuiteConfig
    mode: Literal["full", "draft"]
    folder: Path
    data: dict[str, PreparedData]
    external: PreparedData | None
    splits: SuiteSplits
    jobs: list[SuiteJob]

    def run_dir(self, job: SuiteJob) -> Path:
        """Locate exactly this job; never search for a latest checkpoint."""
        return self.folder / "experiments" / job.key


def _json_value(value: Any) -> Any:
    """Normalize Paths and tuples to stable JSON values."""
    return json.loads(json.dumps(value, default=str))


def _write_json(path: Path, value: Any) -> None:
    """Commit a suite state atomically, without importing TensorFlow for planning."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(_json_value(value), indent=2), encoding="utf-8")
    temporary.replace(path)


def _selected_data(
    settings: RevisionSuiteConfig, mode: Literal["full", "draft"], folder: Path,
) -> tuple[dict[str, PreparedData], PreparedData | None]:
    """Validate real inventory first; automatic draft sampling never overrides files."""
    d1_config = experiment_configuration(settings, "d1", output_dir=folder, mode=mode)
    full_config = experiment_configuration(settings, "mouse_full", output_dir=folder, mode=mode)
    d1 = prepare_data(d1_config)
    mouse_full = prepare_data(full_config)
    d1_ids = settings.d1_ids or tuple(d1.subjects.subject_ids)
    sham_ids = settings.sham_ids or tuple(sorted(set(mouse_full.subjects.subject_ids) - set(d1.subjects.subject_ids)))
    if set(d1_ids).intersection(sham_ids):
        raise ValueError("D1 and SHAM selections contain the same biological animal")
    d1 = select_animals(d1, d1_ids)
    mouse_full = select_animals(mouse_full, (*d1_ids, *sham_ids))
    source_selection = settings.folds_file or settings.anchor_split_file or settings.reference_splits
    if mode == "draft" and source_selection is None:
        balance = select_animals(mouse_full, d1_ids)
        selected_d1 = balanced_sample(balance, settings.draft.d1_animals, settings.split_seed)
        balance = select_animals(mouse_full, sham_ids)
        selected_sham = balanced_sample(balance, settings.draft.sham_animals, settings.split_seed)
        d1 = select_animals(d1, selected_d1)
        mouse_full = select_animals(mouse_full, selected_d1 + selected_sham)

    data: dict[str, PreparedData] = {"d1": d1, "mouse_full": mouse_full}
    if settings.include_rats:
        rat_config = experiment_configuration(settings, "rats", output_dir=folder, mode=mode)
        rats = prepare_data(rat_config)
        rat_ids = settings.rat_ids or tuple(rats.subjects.subject_ids)
        rats = select_animals(rats, rat_ids)
        if mode == "draft" and settings.rat_folds_file is None:
            selected_rats = balanced_sample(rats, settings.draft.rat_animals, settings.split_seed)
            rats = select_animals(rats, selected_rats)
        if len(rats.subjects.subject_ids) < 3:
            raise ValueError("Rat CV needs at least three animals for train/validation/test")
        if set(rats.subjects.subject_ids).intersection(mouse_full.subjects.subject_ids):
            raise ValueError("Rat and mouse cohorts share an explicitly linked animal")
        data["rats"] = rats

    external: PreparedData | None = None
    if settings.include_external:
        overrides = {key: value for key, value in settings.animal_ids.items() if key.startswith("D4/")}
        external_data = DataConfig(
            root=settings.data_root, datasets=(DatasetSpec(name="D4"),), animal_ids=overrides,
        )
        config = replace(full_config, data=external_data)
        external = prepare_data(config, require_training_loss=False)
        external_ids = settings.external_ids or tuple(external.subjects.subject_ids)
        external = select_animals(external, external_ids)
        if mode == "draft":
            ids = balanced_sample(external, settings.draft.external_animals, settings.split_seed)
            external = select_animals(external, ids)
        development = set(mouse_full.subjects.subject_ids)
        development.update(data.get("rats", d1).subjects.subject_ids)
        if development.intersection(external.subjects.subject_ids):
            raise ValueError("External D4 animals overlap development animal identities")
        development_paths: set[str] = set()
        for selected in data.values():
            paths = selected.subjects.manifest(selected.subjects.subject_ids)["ImagePath"]
            development_paths.update(str(Path(path).resolve()).casefold() for path in paths)
        external_paths = {
            str(Path(path).resolve()).casefold()
            for path in external.subjects.manifest(external.subjects.subject_ids)["ImagePath"]
        }
        if development_paths.intersection(external_paths):
            raise ValueError("External images are also registered as development images")
    return data, external


def suite_jobs(settings: RevisionSuiteConfig) -> list[SuiteJob]:
    """Define the economical study: one reference, shared controls, no duplicate fits."""
    jobs: list[SuiteJob] = []
    if settings.include_interpolation:
        for seed in settings.seeds:
            for order in settings.interpolation_orders:
                name = f"interpolation_order_{order}"
                jobs.append(SuiteJob(name, "d1", "interpolation", seed, name, "fixed_split", order=order))
    for seed in settings.seeds:
        jobs.append(SuiteJob("d1_reference", "d1", "d1", seed))
    for seed in settings.seeds:
        jobs.append(SuiteJob("mouse_full_reference", "mouse_full", "mouse_full", seed))
        for variant in settings.supervision_controls:
            name = f"mouse_full_{variant}"
            group: str = "mouse_full_ablation"
            kind: Literal["fixed_split", "cross_validation"] = settings.ablation_mode
            if kind == "cross_validation":
                group = "mouse_full"
            jobs.append(SuiteJob(name, "mouse_full", group, seed, variant, kind))
    for seed in settings.seeds:
        for variant in settings.architecture_ablations:
            group = "d1_ablation"
            if settings.ablation_mode == "cross_validation":
                group = "d1"
            jobs.append(SuiteJob(f"d1_{variant}", "d1", group, seed, variant, settings.ablation_mode))
    if settings.include_rats:
        for seed in settings.seeds:
            source = f"d1_reference/seed_{seed}"
            jobs.append(SuiteJob("rat_finetuning", "rats", "rats", seed, source=source))
    if settings.include_release:
        seed = settings.release_seed
        jobs.append(SuiteJob("release_tbi", "d1", "release", seed, kind="refit"))
        jobs.append(SuiteJob("release_mouse_full", "mouse_full", "release", seed, kind="refit"))
    return jobs


def prepare_suite(
    settings: RevisionSuiteConfig, *, mode: Literal["full", "draft"] = "full",
) -> SuitePlan:
    """Resolve identities and validate every fold before creating result folders."""
    settings.validate()
    if mode not in ("full", "draft"):
        raise ValueError("Suite mode must be full or draft")
    folder = (settings.output_root / mode / settings.run_name).resolve()
    data, external = _selected_data(settings, mode, folder)
    k: int = settings.folds
    if mode == "draft":
        k = settings.draft.folds
    selected_folds = settings.folds_file
    if selected_folds is None and settings.anchor_split_file is None:
        selected_folds = settings.reference_splits
    if selected_folds is not None:
        supplied = read_selection(
            selected_folds, seed=settings.split_seed, validation_fraction=settings.validation_fraction,
        )
        if not isinstance(supplied, list):
            raise ValueError("folds_file requires outer CV folds")
        k = len(supplied)
    splits = make_suite_splits(
        data["d1"], data["mouse_full"], data.get("rats"), k=k, seed=settings.split_seed,
        validation_fraction=settings.validation_fraction, folds_file=selected_folds,
        anchor_split_file=settings.anchor_split_file, rat_folds_file=settings.rat_folds_file,
    )
    if not 0 <= settings.rat_source_fold < len(splits.d1):
        raise ValueError("rat_source_fold must identify a source fold")
    if settings.reference_fold >= len(splits.d1):
        raise ValueError("reference_fold must identify a fold in the shared plan")
    selected = splits.d1[settings.reference_fold]
    train_ids = tuple(selected.train_ids)
    validation_ids = tuple(selected.val_ids)
    test_ids = tuple(selected.test_ids)
    interpolation = SubjectSplit(train=train_ids, validation=validation_ids, test=test_ids)
    splits = replace(splits, interpolation=interpolation)
    return SuitePlan(settings, mode, folder, data, external, splits, suite_jobs(settings))


def _input_records(plan: SuitePlan) -> list[dict[str, Any]]:
    """Fingerprint selected source metadata, reference, split files and scan files."""
    paths: set[Path] = {plan.settings.reference.resolve()}
    datasets = list(plan.data.values())
    if plan.external is not None:
        datasets.append(plan.external)
    for data in datasets:
        for key in data.loader.get_subjects_list():
            scan = data.loader.get_subject(key)
            paths.update(Path(path).resolve() for path in (scan.image_path, scan.labels_path, scan.brain_mask_path))
            dataset, subject = scan.subject_group
            metadata = plan.settings.data_root / dataset / subject / "metadata.json"
            paths.add(metadata.resolve())
    for path in (
        plan.settings.folds_file, plan.settings.anchor_split_file,
        plan.settings.rat_folds_file, plan.settings.inter_rater_manifest,
    ):
        if path is not None:
            paths.add(path.resolve())
    if plan.settings.folds_file is None and plan.settings.anchor_split_file is None:
        if plan.settings.reference_splits is not None:
            paths.add(plan.settings.reference_splits.resolve())
    records: list[dict[str, Any]] = []
    for path in sorted(paths):
        stat = path.stat()
        row: dict[str, Any] = {"Path": str(path), "Bytes": stat.st_size, "ModifiedNs": stat.st_mtime_ns}
        if path.suffix in (".json", ".csv") or path == plan.settings.reference.resolve():
            row["SHA256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        records.append(row)
    return records


def suite_contract(plan: SuitePlan, callbacks: Sequence[Any]) -> dict[str, Any]:
    """Describe exact inputs, jobs and folds; draft and full contracts cannot mix."""
    from .training import callback_configuration

    return _json_value({
        "schema": 1, "mode": plan.mode, "DRAFT": plan.mode == "draft",
        "settings": asdict(plan.settings), "splits": plan.splits.to_dict(),
        "task_configurations": {task: data.config.to_dict() for task, data in plan.data.items()},
        "jobs": [asdict(job) for job in plan.jobs], "callbacks": callback_configuration(callbacks),
        "inputs": _input_records(plan),
    })


def _draft_data(plan: SuitePlan, *, resume: bool) -> None:
    """Bind tiny copies only after the original input/fold contract was checked."""
    folder = plan.folder / "draft_data"
    if folder.exists():
        validate_draft_sources(folder)
    else:
        datasets = list(plan.data.values())
        if plan.external is not None:
            datasets.append(plan.external)
        write_draft_dataset(
            datasets, folder, shape=plan.settings.draft.patch_size,
            spacing_mm=plan.settings.spacing_mm, reference=plan.settings.reference,
        )
    for task, original in list(plan.data.items()):
        data_config = replace(original.config.data, root=folder)
        config = replace(original.config, data=data_config)
        loaded = prepare_data(config)
        plan.data[task] = select_animals(loaded, original.subjects.subject_ids)
    if plan.external is not None:
        original = plan.external
        data_config = replace(original.config.data, root=folder)
        config = replace(original.config, data=data_config)
        loaded = prepare_data(config, require_training_loss=False)
        plan.external = select_animals(loaded, original.subjects.subject_ids)


def _refit_epochs(plan: SuitePlan, task: str) -> int:
    """Fix release duration from inner-validation selected epochs, never test scores."""
    source = "d1_reference"
    if task == "mouse_full":
        source = "mouse_full_reference"
    folder = plan.folder / "experiments" / source / f"seed_{plan.settings.release_seed}"
    timings = pd.read_csv(folder / "summary" / "fold_timings.csv")
    selected = pd.to_numeric(timings["best_epoch"], errors="raise") + 1
    if selected.isna().any() or (selected < 1).any():
        raise ValueError("Release duration requires valid inner-validation best epochs")
    return max(1, int(np.ceil(selected.median())))


def job_configuration(plan: SuitePlan, job: SuiteJob) -> ExperimentConfig:
    """Resolve the source checkpoint only when its prerequisite has completed."""
    base = plan.data[job.task].config
    folds = getattr(plan.splits, job.task)
    training = replace(
        base.training, model_name=job.name, run_id=f"seed_{job.seed}", seed=job.seed,
        k=len(folds), output_dir=str(plan.folder / "experiments"),
    )
    config = replace(base, training=training)
    if job.kind == "fixed_split":
        # CV seeds each fold with seed + index; match its selected reference fit.
        training = replace(training, seed=job.seed + plan.settings.reference_fold)
        config = replace(config, training=training)
    if job.group == "interpolation":
        preprocessing = replace(config.preprocessing, resampling_order=job.order)
        epochs = plan.settings.interpolation_epochs
        steps = plan.settings.interpolation_steps
        if plan.mode == "draft":
            epochs = plan.settings.draft.epochs
            steps = plan.settings.draft.steps_per_epoch
        training = replace(training, epochs=epochs, steps_per_epoch=steps)
        config = replace(config, preprocessing=preprocessing, training=training)

    model = config.model
    architecture_flags = {
        "no_attention": "attention", "no_residual": "residual", "no_multiscale": "multiscale",
        "no_deep_supervision": "deep_supervision", "strided_downsampling": "max_pooling",
    }
    if job.variant in architecture_flags:
        changes = {architecture_flags[job.variant]: False}
        model = replace(model, **changes)
    if job.variant in ("no_classifier", "missing_as_background_no_classifier"):
        model = replace(model, classifier=False)
    loss = config.loss
    if job.variant in ("missing_as_background", "missing_as_background_no_classifier"):
        loss = replace(loss, annotation_policy="missing_as_background")
    config = replace(config, model=model, loss=loss)

    if job.source is not None:
        if job.task != "rats":
            raise ValueError("Suite mouse experiments start from scratch; pretraining is reserved for rat fine-tuning")
        source = plan.folder / "experiments" / job.source
        source_fold = plan.settings.rat_source_fold
        initialization = InitializationConfig.from_run(source, fold=source_fold)
        config = replace(config, initialization=initialization)
    if job.kind == "refit":
        epochs = _refit_epochs(plan, job.task)
        training = replace(config.training, epochs=epochs, monitor="loss")
        config = replace(config, training=training)
    config.validate()
    return config


def _execute_job(plan: SuitePlan, job: SuiteJob, callbacks: Sequence[Any]) -> Path:
    """Delegate fitting/recovery to the common runners, preserving shared patches."""
    from .training import run_cross_validation, run_development_refit, run_training

    config = job_configuration(plan, job)
    data = replace(plan.data[job.task], config=config)
    folder = plan.run_dir(job)
    resume_run: Path | None = None
    if (folder / "experiment.json").is_file():
        resume_run = folder
    if job.kind == "fixed_split":
        selected = getattr(plan.splits, job.task)[plan.settings.reference_fold]
        train_ids = tuple(selected.train_ids)
        validation_ids = tuple(selected.val_ids)
        test_ids = tuple(selected.test_ids)
        split = SubjectSplit(train=train_ids, validation=validation_ids, test=test_ids)
        selected_callbacks = callbacks
        if job.group == "interpolation":
            selected_callbacks = []
        result = run_training(data, split, resume_run=resume_run, callbacks=selected_callbacks)
    elif job.kind == "refit":
        result = run_development_refit(data, resume_run=resume_run)
    else:
        folds = getattr(plan.splits, job.task)
        result = run_cross_validation(data, folds=folds, resume_run=resume_run, callbacks=callbacks)
    return result.run_dir


def _external_evaluation(plan: SuitePlan) -> Path:
    """Evaluate the refit on D4, recursively checking all source animal histories."""
    from .initialization import pretraining_audit
    from ..evaluation.inference import SavedModelEvaluationConfig, evaluate_saved_model

    assert plan.external is not None
    run = plan.folder / "experiments" / "release_mouse_full" / f"seed_{plan.settings.release_seed}"
    source = InitializationConfig.from_run(run)
    ids = tuple(plan.external.subjects.subject_ids)
    audit = pretraining_audit(plan.external, source, protected_ids=ids)
    folder = plan.folder / "external" / "D4"
    settings = SavedModelEvaluationConfig(
        experiment_file=run / "experiment.json", model_file=source.checkpoint,
        output_dir=folder, subject_ids=ids, variant="release_mouse_full",
        seed=plan.settings.release_seed, data=plan.external.config.data,
        reference=plan.settings.reference, checkpoint_kind="weights",
        original_label_ids=True, evaluate_patches=True,
        save_predictions=plan.settings.save_predictions,
    )
    result = evaluate_saved_model(settings)
    _write_json(result.output_dir / "pretraining_audit.json", audit)
    return result.output_dir


def run_revision_suite(
    settings: RevisionSuiteConfig, *, mode: Literal["full", "draft"] = "full",
    callbacks: Sequence[Any] = (), resume: bool = False, plan_only: bool = False,
) -> Path:
    """Plan or execute the suite sequentially, stopping and recording any failure.

    This does not launch a collaborator's nnU-Net jobs. Explicit saved result
    folders may be included in the compact summary through external_results.
    """
    plan = prepare_suite(settings, mode=mode)
    contract = suite_contract(plan, callbacks)
    contract_path = plan.folder / "suite.json"
    if plan.folder.exists():
        if not resume:
            raise FileExistsError(f"Suite exists; use resume=True: {plan.folder}")
        saved = json.loads(contract_path.read_text(encoding="utf-8"))
        if saved != contract:
            raise ValueError("Suite configuration, source files, callbacks or split memberships changed")
    else:
        plan.folder.mkdir(parents=True)
        _write_json(contract_path, contract)
        _write_json(plan.folder / "splits.json", plan.splits.to_dict())
        memberships = [
            split_membership(plan.data["d1"], plan.splits.d1, "d1"),
            split_membership(plan.data["mouse_full"], plan.splits.mouse_full, "mouse_full"),
        ]
        if "rats" in plan.data:
            memberships.append(split_membership(plan.data["rats"], plan.splits.rats, "rats"))
        pd.concat(memberships, ignore_index=True).to_csv(plan.folder / "split_membership.csv", index=False)
        rows = []
        for job in plan.jobs:
            fits = 1
            if job.kind == "cross_validation":
                fits = len(getattr(plan.splits, job.task))
            rows.append({**asdict(job), "Key": job.key, "Fits": fits})
        pd.DataFrame(rows).to_csv(plan.folder / "experiment_plan.csv", index=False)
        warning = "FULL STUDY: independent outer tests; selection uses inner validation only."
        if mode == "draft":
            warning = "DRAFT ONLY: tiny cropped fields, tiny networks and one-step fits. Never use these values in a paper."
        (plan.folder / "README.txt").write_text(warning + "\n", encoding="utf-8")
    fits: int = 0
    for job in plan.jobs:
        count: int = 1
        if job.kind == "cross_validation":
            count = len(getattr(plan.splits, job.task))
        fits += count
    print(f"Suite: {plan.folder}\nMode: {mode}; jobs: {len(plan.jobs)}; fits: {fits}", flush=True)
    if plan_only:
        return plan.folder

    from .training import configure_runtime
    from ..evaluation.training_state import run_lock
    from .revision_summary import export_suite_summary, supplementary_analyses

    configure_runtime(seed=settings.seeds[0], mixed_precision=settings.mixed_precision)
    state_file = plan.folder / "state.json"
    state: dict[str, Any] = {"status": "running", "jobs": {}}
    if state_file.is_file():
        state = json.loads(state_file.read_text(encoding="utf-8"))
    with run_lock(plan.folder):
        if mode == "draft":
            _draft_data(plan, resume=resume)
        for job in plan.jobs:
            if job.source is not None and state["jobs"].get(job.source, {}).get("status") != "complete":
                raise ValueError(f"Prerequisite not complete: {job.source}")
            print(f"Starting {job.key}", flush=True)
            state["status"] = "running"
            state["jobs"][job.key] = {"status": "running", "directory": str(plan.run_dir(job))}
            _write_json(state_file, state)
            try:
                folder = _execute_job(plan, job, callbacks)
            except BaseException as error:
                state["status"] = "interrupted"
                state["jobs"][job.key]["status"] = "interrupted"
                state["jobs"][job.key]["error"] = f"{type(error).__name__}: {error}"
                _write_json(state_file, state)
                raise
            state["jobs"][job.key] = {"status": "complete", "directory": str(folder)}
            _write_json(state_file, state)

        if settings.include_external:
            external_folder = plan.folder / "external" / "D4"
            if not state.get("external_complete"):
                if external_folder.exists():
                    from datetime import datetime, timezone

                    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
                    archived = external_folder.with_name("D4_interrupted_" + stamp)
                    external_folder.rename(archived)
                    print(f"Preserved incomplete D4 evaluation: {archived}", flush=True)
                try:
                    _external_evaluation(plan)
                except BaseException as error:
                    state["status"] = "interrupted"
                    state["external_error"] = f"{type(error).__name__}: {error}"
                    _write_json(state_file, state)
                    raise
                state["external_complete"] = True
                state.pop("external_error", None)
                _write_json(state_file, state)
        try:
            supplementary_analyses(plan)
            bootstrap_samples = settings.bootstrap_samples
            if mode == "draft":
                bootstrap_samples = 100
            summary = export_suite_summary(plan.folder, bootstrap_samples=bootstrap_samples)
        except BaseException as error:
            state["status"] = "reporting_interrupted"
            state["reporting_error"] = f"{type(error).__name__}: {error}"
            _write_json(state_file, state)
            raise
        state["status"] = "complete"
        state.pop("reporting_error", None)
        state["paper_summary"] = str(summary)
        _write_json(state_file, state)
        print(f"Compact paper tables: {summary}", flush=True)
    return plan.folder
