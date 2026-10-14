"""Small matched ablation studies using the existing experiment runners."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Literal, Protocol, Sequence

import numpy as np
import pandas as pd

from .config import ExperimentConfig, InitializationConfig
from .data import PreparedData, SubjectSplit, fold_summary, make_cv_splits
from .training import callback_configuration, run_cross_validation, run_training
from ..evaluation.k_fold import FoldSplit, TrainingResult, load_fold_splits


ARCHITECTURE_PARAMETERS: tuple[str, ...] = (
    "residual", "attention", "max_pooling", "classifier", "multiscale", "deep_supervision",
)
LOSS_PARAMETERS: tuple[str, ...] = ("regions", "gradient_fix_version")


@dataclass(frozen=True)
class AblationVariant:
    """One named change, or an unchanged reference when parameter is None."""

    name: str
    parameter: Literal["residual", "attention", "max_pooling", "classifier",
                       "multiscale", "deep_supervision", "augment", "random_crop",
                       "data_multiplier", "regions", "gradient_fix_version"] | None = None
    value: bool | int | str | None = None
    initialization: InitializationConfig | None = None
    initialization_by_fold: dict[int, InitializationConfig] | None = None

    @property
    def category(self) -> str:
        """Separate reference, architecture, loss and training-strategy experiments."""
        if self.parameter is None:
            return "reference"
        if self.parameter in ARCHITECTURE_PARAMETERS:
            return "architecture"
        if self.parameter in LOSS_PARAMETERS:
            return "loss"
        return "training"

    def configuration(self, base: ExperimentConfig) -> ExperimentConfig:
        """Replace exactly one supported setting without changing the base."""
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", self.name):
            raise ValueError("Variant names must be safe folder identifiers")
        parameter = self.parameter
        if parameter is None:
            if self.value is not None:
                raise ValueError("A reference variant cannot change a value")
            config = base
            if self.initialization is not None:
                config = replace(config, initialization=self.initialization)
            if self.initialization_by_fold is not None:
                config = replace(config, initialization_by_fold=self.initialization_by_fold)
            config.validate()
            return config

        if parameter in LOSS_PARAMETERS:
            previous = getattr(base.loss, parameter)
            if parameter == "regions":
                if self.value not in ("dice_ce", "dice_ce_gradient_fix"):
                    raise ValueError("ROI loss must be dice_ce or dice_ce_gradient_fix")
            else:
                if base.loss.regions != "dice_ce_gradient_fix":
                    raise ValueError("gradient_fix_version only changes a gradient-fix objective")
                if isinstance(self.value, bool) or not isinstance(self.value, int) or self.value not in (1, 2, 3):
                    raise ValueError("Gradient-fix objective version must be one, two or three")
            loss_changes = {parameter: self.value}
            loss = replace(base.loss, **loss_changes)
            config = replace(base, loss=loss)
        elif parameter == "data_multiplier":
            if isinstance(self.value, bool) or not isinstance(self.value, int) or self.value < 1:
                raise ValueError("data_multiplier must be a positive integer")
            previous = base.training.data_multiplier
            training = replace(base.training, data_multiplier=self.value)
            config = replace(base, training=training)
        else:
            if not isinstance(self.value, bool):
                raise ValueError("Component and preprocessing switches must be booleans")
            if parameter in ARCHITECTURE_PARAMETERS:
                if base.model.architecture != "3drnet":
                    raise ValueError("Architectural ablations require a 3DR-Net reference")
                previous = getattr(base.model, parameter)
                model_changes = {parameter: self.value}
                model = replace(base.model, **model_changes)
                config = replace(base, model=model)
            elif parameter in ("augment", "random_crop"):
                previous = getattr(base.preprocessing, parameter)
                preprocessing = replace(base.preprocessing, **{parameter: self.value})
                config = replace(base, preprocessing=preprocessing)
            else:
                raise ValueError(f"Unsupported ablation parameter: {parameter}")

        if previous == self.value:
            raise ValueError(f"Variant {self.name} does not change {parameter}")
        if self.initialization is not None:
            config = replace(config, initialization=self.initialization)
        if self.initialization_by_fold is not None:
            config = replace(config, initialization_by_fold=self.initialization_by_fold)
        config.validate()
        return config


def ablation_plan(base: ExperimentConfig, variants: Sequence[AblationVariant]) -> pd.DataFrame:
    """Validate the experiment matrix and show each single-setting change."""
    base.validate()
    if not variants or len({variant.name for variant in variants}) != len(variants):
        raise ValueError("Provide a nonempty matrix with unique variant names")
    if sum(variant.parameter is None for variant in variants) != 1:
        raise ValueError("Include exactly one unchanged reference variant")

    rows: list[dict[str, Any]] = []
    for variant in variants:
        variant.configuration(base)
        previous: bool | int | str | None = None
        if variant.parameter == "data_multiplier":
            previous = base.training.data_multiplier
        elif variant.category == "architecture":
            previous = getattr(base.model, variant.parameter)
        elif variant.category == "loss":
            previous = getattr(base.loss, variant.parameter)
        elif variant.category == "training":
            previous = getattr(base.preprocessing, variant.parameter)
        rows.append({"Variant": variant.name, "Category": variant.category,
                     "Parameter": variant.parameter, "ReferenceValue": previous, "Value": variant.value})
    return pd.DataFrame(rows)


@dataclass
class AblationResult:
    """Study folder, typed training runs and descriptive evaluation tables."""

    study_dir: Path
    runs: dict[str, TrainingResult] = field(default_factory=dict)
    metrics: pd.DataFrame = field(default_factory=pd.DataFrame)
    timings: pd.DataFrame = field(default_factory=pd.DataFrame)
    patch_metrics: pd.DataFrame = field(default_factory=pd.DataFrame)
    classifier_predictions: pd.DataFrame = field(default_factory=pd.DataFrame)


class StudyVariant(Protocol):
    """Named configuration replacement used by matched study execution."""

    name: str

    def configuration(self, base: ExperimentConfig) -> ExperimentConfig:
        """Return the complete configuration for this study member."""
        ...


@dataclass(frozen=True)
class StudySelection:
    """One fixed split or outer-fold selection and its animal/scan counts."""

    split: SubjectSplit | None
    folds: list[FoldSplit] | None
    summary: pd.DataFrame


def study_selection(
    data: PreparedData, *, mode: Literal["fixed_split", "cross_validation"],
    study_dir: Path, split_file: Path | None = None, fold_file: Path | None = None,
    resume: bool = False,
) -> StudySelection:
    """Generate or load matched animals, checking explicit files during resumption."""
    saved = Path(study_dir) / "splits.json"
    if mode == "fixed_split":
        if fold_file is not None:
            raise ValueError("Use split_file for fixed-split studies")
        if resume:
            split = SubjectSplit.load(saved, data.subjects)
            if split_file is not None:
                requested = SubjectSplit.load(split_file, data.subjects)
                if requested != split:
                    raise ValueError("Requested split differs from the resumed study")
        elif split_file is not None:
            split = SubjectSplit.load(split_file, data.subjects)
        else:
            split = data.make_split()
        return StudySelection(split=split, folds=None, summary=data.split_summary(split))

    if mode != "cross_validation" or split_file is not None:
        raise ValueError("Use cross_validation and fold_file for grouped CV studies")
    config = data.config
    animals = data.subjects.subject_ids
    if resume:
        folds = load_fold_splits(saved, animals, config.training.k)
        if fold_file is not None:
            requested = load_fold_splits(fold_file, animals, config.training.k)
            if requested != folds:
                raise ValueError("Requested folds differ from the resumed study")
    elif fold_file is not None:
        folds = load_fold_splits(fold_file, animals, config.training.k)
    else:
        folds = make_cv_splits(
            animals, k=config.training.k, seed=config.split_seed,
            validation_fraction=config.cv_validation_fraction,
        )
    if any(not fold.test_ids for fold in folds):
        raise ValueError("Matched CV requires independent outer holdouts")
    return StudySelection(split=None, folds=folds, summary=fold_summary(folds, data.subjects))


def run_ablation(
    data: PreparedData, variants: Sequence[AblationVariant], *, study_dir: Path,
    mode: Literal["fixed_split", "cross_validation"] = "fixed_split",
    split: SubjectSplit | None = None, folds: list[FoldSplit] | None = None,
    seeds: Sequence[int] = (42,), resume: bool = False,
    callbacks: Sequence[Any] | None = None,
) -> AblationResult:
    """Run every variant/seed on identical animals through shared fitting code.

    The study owns stable run folders. Explicit resume validates the complete
    study contract, then delegates recovery and completed-run reuse to the core.
    Maximum epochs and callbacks stay matched; early stopping may change actual
    duration. Stochastic augmentation is not guaranteed to match bitwise.
    """
    plan = ablation_plan(data.config, variants)
    return run_study(
        data=data, variants=variants, plan=plan, study_dir=study_dir, mode=mode,
        split=split, folds=folds, seeds=seeds, resume=resume, callbacks=callbacks,
    )


def run_study(
    data: PreparedData, variants: Sequence[StudyVariant], *, plan: pd.DataFrame,
    study_dir: Path, mode: Literal["fixed_split", "cross_validation"] = "fixed_split",
    split: SubjectSplit | None = None, folds: list[FoldSplit] | None = None,
    seeds: Sequence[int] = (42,), resume: bool = False,
    callbacks: Sequence[Any] | None = None,
) -> AblationResult:
    """Execute a validated ablation or model comparison with one recovery contract.

    Callers validate their matrix before entering this shared executor. Variants
    are dataclasses, allowing the exact settings to be saved in the study contract.
    """
    base = data.config
    seeds = list(seeds)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Provide distinct training seeds")
    if any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in seeds):
        raise ValueError("Training seeds must be nonnegative integers")
    if base.training.run_id is not None:
        raise ValueError("The study manages run IDs; leave training.run_id unset")

    if mode == "fixed_split":
        if split is None or folds is not None:
            raise ValueError("Fixed-split studies require one split and no folds")
        split.validate(data.subjects)
        selection = {"train": split.train, "validation": split.validation, "test": split.test}
    elif mode == "cross_validation":
        if split is not None or folds is None or len(folds) != base.training.k:
            raise ValueError("CV studies require the configured folds and no fixed split")
        if [fold.index for fold in folds] != list(range(base.training.k)):
            raise ValueError("Fold indexes must cover the configured folds in order")
        animals = set(data.subjects.subject_ids)
        holdout = []
        for fold in folds:
            fold.validate(animals)
            if not fold.test_ids:
                raise ValueError("Study CV requires independent outer holdouts")
            holdout.extend(fold.test_ids)
        if sorted(holdout) != sorted(animals):
            raise ValueError("Each animal must occur in exactly one outer holdout")
        selection = [fold.to_dict() for fold in folds]
    else:
        raise ValueError("Unknown study mode")

    from .initialization import pretraining_audit

    for variant in variants:
        variant_config = variant.configuration(base)
        variant_data = replace(data, config=variant_config)
        if variant_config.loss.regions != "dice_ce_gradient_fix":
            keys = data.subjects.scan_keys(data.subjects.subject_ids)
            for key in keys:
                scan = data.loader.get_subject(key)
                if scan.unavailable_labels:
                    raise ValueError(f"{variant.name}: partial annotations require dice_ce_gradient_fix")
        if split is not None:
            protected_ids = [*split.validation, *split.test]
            pretraining_audit(variant_data, variant_data.config.initialization, protected_ids)
        else:
            for fold in folds:
                initializations = variant_data.config.initialization_by_fold
                initialization = initializations.get(fold.index, variant_data.config.initialization)
                protected_ids = [*fold.val_ids, *fold.test_ids]
                pretraining_audit(variant_data, initialization, protected_ids)

    folder = Path(study_dir).resolve()
    contract = {"base": base.to_dict(), "variants": [asdict(variant) for variant in variants],
                "seeds": seeds, "mode": mode, "splits": selection, "data": data.contract()}
    if callbacks is not None:
        contract["callbacks"] = callback_configuration(callbacks)
    contract = json.loads(json.dumps(contract, default=str))
    if resume:
        saved = json.loads((folder / "study.json").read_text(encoding="utf-8"))
        if "fixed_split" not in saved["base"]:
            # Generation defaults cannot replace the exact memberships stored
            # in an older study; the split comparison below remains mandatory.
            saved["base"]["fixed_split"] = contract["base"]["fixed_split"]
        saved["base"] = ExperimentConfig.from_dict(saved["base"]).to_dict()
        saved = json.loads(json.dumps(saved, default=str))
        saved_loss = saved["base"]["loss"]
        if saved_loss["regions"] == "dice_ce_gradient_fix":
            if saved_loss["gradient_fix_version"] != base.loss.gradient_fix_version:
                raise ValueError("Partial-label loss objective changed; start a new ablation/comparison study")
        if saved != contract:
            raise ValueError("Resume study configuration, data, seeds or splits differ")
    else:
        folder.mkdir(parents=True, exist_ok=False)

    from ..evaluation.training_state import run_lock, write_json

    result = AblationResult(study_dir=folder)
    metric_tables: list[pd.DataFrame] = []
    patch_tables: list[pd.DataFrame] = []
    classifier_tables: list[pd.DataFrame] = []
    timing_rows: list[dict[str, Any]] = []
    run_rows: list[dict[str, Any]] = []
    with run_lock(folder):
        data.save_evaluation_patches(folder / "evaluation_patches.json", resume=resume)
        if not resume:
            write_json(folder / "study.json", contract)
            plan.to_csv(folder / "plan.csv", index=False)
            if split is not None:
                split.save(folder / "splits.json")
            else:
                write_json(folder / "splits.json", selection)
        for variant in variants:
            variant_config = variant.configuration(base)
            for seed in seeds:
                run_id = f"seed_{seed}"
                training = replace(
                    variant_config.training, model_name=variant.name, seed=seed,
                    output_dir=str(folder / "runs"), run_id=run_id,
                )
                config = replace(variant_config, training=training)
                prepared = replace(data, config=config)
                run_dir = folder / "runs" / variant.name / run_id
                resume_run: Path | None = None
                if resume and run_dir.exists():
                    resume_run = run_dir
                run_options: dict[str, Any] = {"resume_run": resume_run}
                if callbacks is not None:
                    run_options["callbacks"] = callbacks
                if mode == "fixed_split":
                    run = run_training(prepared, split, **run_options)
                else:
                    run = run_cross_validation(prepared, folds=folds, **run_options)

                result.runs[f"{variant.name}/{run_id}"] = run
                if not run.patch_metrics.empty:
                    patch_table = run.patch_metrics.assign(Variant=variant.name, Seed=seed)
                    patch_tables.append(patch_table)
                if not run.classifier_predictions.empty:
                    classifier_table = run.classifier_predictions.assign(Variant=variant.name, Seed=seed)
                    classifier_tables.append(classifier_table)
                run_rows.append({"Variant": variant.name, "Seed": seed, "RunDir": str(run.run_dir),
                                 "ConfigurationSHA256": config.fingerprint()})
                for evaluator, tables in run.evaluations.items():
                    for table in tables:
                        annotated = table.assign(Variant=variant.name, Seed=seed, Evaluator=evaluator)
                        metric_tables.append(annotated)
                for timing in run.timings:
                    timing_rows.append({"Variant": variant.name, "Seed": seed, **asdict(timing)})
                write_json(folder / "runs.json", run_rows)

        if metric_tables:
            result.metrics = pd.concat(metric_tables, ignore_index=True)
        if patch_tables:
            result.patch_metrics = pd.concat(patch_tables, ignore_index=True)
        if classifier_tables:
            result.classifier_predictions = pd.concat(classifier_tables, ignore_index=True)
        from .evaluation import export_evaluation_tables

        export_evaluation_tables(
            result.patch_metrics, result.metrics, folder,
            classifier_predictions=result.classifier_predictions,
        )
        result.timings = pd.DataFrame(timing_rows)
        result.timings.to_csv(folder / "timings.csv", index=False)
    return result


def animal_metric_values(
    result: AblationResult, *, metric: str = "Dice", role: str = "test",
) -> pd.DataFrame:
    """Return descriptive scan means per animal, variant, seed and fold.

    Missing/undefined values are excluded, never replaced with zero. Means retain
    seed/fold identity; repeated models and longitudinal scans are not counted as
    additional biological animals. This helper performs no significance tests.
    """
    table = result.metrics
    if table.empty:
        return pd.DataFrame()
    selected = table.loc[(table["Metric"] == metric) & (table["DatasetRole"] == role)].copy()
    selected = selected.loc[np.isfinite(selected["Value"])]
    keys = ["Variant", "Seed", "Fold", "DatasetRole", "Subject", "Dataset", "Region", "Metric", "Unit"]
    grouped = selected.groupby(keys, dropna=False)
    summary = grouped.agg(Value=("Value", "mean"), Scans=("Scan", "nunique"))
    return summary.reset_index()
