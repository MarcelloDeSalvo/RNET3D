"""Paper-ready summaries of saved test/holdout results, without new inference."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any
import hashlib
import json

import numpy as np
import pandas as pd

from .classification import classification_report, healthy_lesion_report
from .reporting import _normalized_metrics, attach_scan_metadata, region_set_scores
from .revision import CostSource, RevisionConfig, cost_report, paired_model_comparison, volume_pairs
from .statistics import AgreementConfig, agreement_table


@dataclass(frozen=True)
class PaperTableConfig:
    """Explicit evaluation roles and uncertainty settings for manuscript tables.

    Main cells show equal-weight fold means +/- fold SD. Animal intervals use
    animal means after averaging repeated models per scan and scans per animal;
    they describe a separate estimator, conditional on the supplied models.
    Seeds, datasets, conditions and evaluation roles remain separate.
    """

    roles: tuple[str, ...] = ("holdout", "test")
    metrics: tuple[str, ...] = (
        "Dice", "HD95", "ASSD", "AVE", "RVE", "Sensitivity", "Precision", "IoU",
        "ReferenceVolume", "PredictionVolume",
    )
    complete_region_sets_only: bool = True
    confidence: float = 0.95
    bootstrap_samples: int = 2000
    seed: int = 42
    decimals: int = 3
    reference_variant: str | None = None

    def __post_init__(self) -> None:
        """Reject ambiguous roles or invalid uncertainty/format settings."""
        allowed: set[str] = {"test", "holdout", "external", "validation"}
        if not self.roles or not set(self.roles).issubset(allowed) or len(set(self.roles)) != len(self.roles):
            raise ValueError("Select explicit unique test/holdout/external/validation roles")
        if not self.metrics or len(set(self.metrics)) != len(self.metrics):
            raise ValueError("Select unique metric names")
        if not 0 < self.confidence < 1 or self.bootstrap_samples < 100:
            raise ValueError("Use confidence between zero and one and at least 100 bootstrap samples")
        if type(self.seed) is not int or self.seed < 0 or type(self.decimals) is not int or not 0 <= self.decimals <= 8:
            raise ValueError("Use a nonnegative integer seed and 0–8 decimal places")


@dataclass
class PaperTables:
    """Compact manuscript dataframes plus full-precision numerical audit tables."""

    segmentation: pd.DataFrame = field(default_factory=pd.DataFrame)
    region_sets: pd.DataFrame = field(default_factory=pd.DataFrame)
    animal_intervals: pd.DataFrame = field(default_factory=pd.DataFrame)
    statistics: pd.DataFrame = field(default_factory=pd.DataFrame)
    fold_metrics: pd.DataFrame = field(default_factory=pd.DataFrame)
    patches: pd.DataFrame = field(default_factory=pd.DataFrame)
    classification: pd.DataFrame = field(default_factory=pd.DataFrame)
    confusion: pd.DataFrame = field(default_factory=pd.DataFrame)
    agreement: pd.DataFrame = field(default_factory=pd.DataFrame)
    costs: pd.DataFrame = field(default_factory=pd.DataFrame)
    comparisons: pd.DataFrame = field(default_factory=pd.DataFrame)
    healthy_lesion: pd.DataFrame = field(default_factory=pd.DataFrame)
    fold_coverage: pd.DataFrame = field(default_factory=pd.DataFrame)
    sources: pd.DataFrame = field(default_factory=pd.DataFrame)
    settings: PaperTableConfig = field(default_factory=PaperTableConfig)

    def tables(self) -> dict[str, pd.DataFrame]:
        """Return stable sheet/CSV names; settings are stored separately as JSON."""
        return {item.name: getattr(self, item.name) for item in fields(self) if item.name != "settings"}

    def export(self, output_dir: Path) -> Path:
        """Write a fresh CSV/Excel bundle, verifying inputs have not changed."""
        for source in self.sources.to_dict(orient="records"):
            digest: str = hashlib.sha256(Path(source["Path"]).read_bytes()).hexdigest()
            if digest != source["SHA256"]:
                raise ValueError("Paper-table input changed after analysis; rebuild the tables before exporting")
        folder: Path = Path(output_dir)
        folder.mkdir(parents=True, exist_ok=False)
        with pd.ExcelWriter(folder / "paper_tables.xlsx") as writer:
            for name, table in self.tables().items():
                table.to_csv(folder / f"paper_{name}.csv", index=False)
                table.to_excel(writer, sheet_name=name, index=False)
        metadata: dict[str, Any] = {
            "settings": asdict(self.settings),
            "main_cells": "equal-weight mean of fold animal-means +/- sample SD of fold means",
            "animal_intervals": "separate animal mean estimator; repeats per scan, then scans per animal",
            "single_fold": "mean only; fold SD is undefined, not zero",
            "missing_values": "unavailable and undefined values excluded with explicit denominator counts",
            "classifier_intervals": "exact animal binomial intervals within a model or disjoint pooled holdouts",
            "sources": self.sources.to_dict(orient="records"),
        }
        (folder / "paper_analysis.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        return folder


GROUPS: list[str] = ["Variant", "Seed", "DatasetRole", "Dataset", "Condition", "Head", "Region", "Metric", "Unit"]


def _interval(values: np.ndarray, settings: PaperTableConfig) -> tuple[float, float]:
    """Bootstrap unique animal means; no scan/fold observation is independent."""
    if len(values) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(settings.seed)
    means: np.ndarray = np.empty(settings.bootstrap_samples)
    for index in range(settings.bootstrap_samples):
        draw: np.ndarray = rng.integers(0, len(values), len(values))
        means[index] = values[draw].mean()
    alpha: float = (1 - settings.confidence) / 2
    lower, upper = np.quantile(means, [alpha, 1 - alpha])

    return float(lower), float(upper)


def _summarize(table: pd.DataFrame, settings: PaperTableConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Average scans within animals, then animals within each fold before folds."""
    fold_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for key, group in table.groupby(GROUPS, dropna=False):
        identifiers: dict[str, Any] = dict(zip(GROUPS, key))
        fold_values: list[float] = []
        for fold, samples in group.groupby("Fold", dropna=False):
            finite: pd.DataFrame = samples.loc[np.isfinite(samples["Value"])]
            animals: pd.Series = finite.groupby("Subject")["Value"].mean()
            value: float = float("nan")
            if not animals.empty:
                value = float(animals.mean())
                fold_values.append(value)
            annotated: pd.DataFrame = samples.loc[samples["Status"].ne("unavailable")]
            fold_rows.append({
                **identifiers, "Fold": fold, "Mean": value, "Animals": len(animals),
                "ObservedAnimals": samples["Subject"].nunique(), "AnnotatedAnimals": annotated["Subject"].nunique(),
                "ObservedScans": len(samples[["Subject", "Scan"]].drop_duplicates()),
                "UnavailableScans": int(samples["Status"].eq("unavailable").sum()),
                "UndefinedScans": int((~np.isfinite(samples["Value"]) & samples["Status"].ne("unavailable")).sum()),
            })
        finite = group.loc[np.isfinite(group["Value"])]
        scans: pd.DataFrame = finite.groupby(["Subject", "Scan"], dropna=False)["Value"].mean().reset_index()
        animals = scans.groupby("Subject")["Value"].mean()
        values: np.ndarray = animals.to_numpy(dtype=float)
        lower, upper = _interval(values, settings)
        fold_mean: float = float("nan")
        fold_sd: float = float("nan")
        animal_mean: float = float("nan")
        animal_sd: float = float("nan")
        if fold_values:
            fold_mean = float(np.mean(fold_values))
        if len(fold_values) > 1:
            fold_sd = float(np.std(fold_values, ddof=1))
        if len(values):
            animal_mean = float(values.mean())
        if len(values) > 1:
            animal_sd = float(values.std(ddof=1))
        repeat_counts: pd.Series = group[["Subject", "Fold"]].drop_duplicates().groupby("Subject").size()
        summary_rows.append({
            **identifiers, "Mean": fold_mean, "FoldSD": fold_sd,
            "Folds": group["Fold"].nunique(), "FiniteFolds": len(fold_values),
            "Animals": len(values), "ObservedAnimals": group["Subject"].nunique(),
            "ObservedScans": len(group[["Subject", "Scan"]].drop_duplicates()),
            "AnimalMean": animal_mean, "AnimalSD": animal_sd,
            "AnimalCILower": lower, "AnimalCIUpper": upper,
            "RepeatedAnimalsAcrossFolds": bool(repeat_counts.gt(1).any()),
        })

    return pd.DataFrame(summary_rows), pd.DataFrame(fold_rows)


def _compact(statistics: pd.DataFrame, settings: PaperTableConfig, *, animal_intervals: bool = False) -> pd.DataFrame:
    """Pivot already summarized values; single-fold results do not acquire zero SD."""
    if statistics.empty:
        return pd.DataFrame()
    ids: list[str] = GROUPS[:-2]
    rows: list[dict[str, Any]] = []
    for key, group in statistics.groupby(ids, dropna=False):
        row: dict[str, Any] = dict(zip(ids, key))
        row.update({"Folds": int(group["Folds"].max()), "ObservedAnimals": int(group["ObservedAnimals"].max())})
        counts: dict[str, int] = {}
        for item in group.to_dict(orient="records"):
            name: str = item["Metric"]
            if item["Unit"] not in ("fraction", "unitless"):
                name += f" ({item['Unit']})"
            counts[name] = int(item["Animals"])
            value: float = item["Mean"]
            spread: float = item["FoldSD"]
            text: str = "—"
            if animal_intervals:
                value = item["AnimalMean"]
            if np.isfinite(value):
                text = f"{value:.{settings.decimals}f}"
                if animal_intervals and np.isfinite(item["AnimalCILower"]):
                    text += f" [{item['AnimalCILower']:.{settings.decimals}f}, {item['AnimalCIUpper']:.{settings.decimals}f}]"
                elif not animal_intervals and np.isfinite(spread):
                    text += f" ± {spread:.{settings.decimals}f}"
            row[name] = text
        row["Animals"] = max(counts.values(), default=0)
        if "Dice" in counts:
            row["Animals"] = counts["Dice"]
            row["DiceFolds"] = int(group.loc[group["Metric"].eq("Dice"), "FiniteFolds"].iloc[0])
        boundary_counts: list[int] = []
        for metric in ("HD95", "ASSD"):
            for name, count in counts.items():
                if name.startswith(metric + " ("):
                    boundary_counts.append(count)
        if boundary_counts:
            row["BoundaryAnimals"] = min(boundary_counts)
        rows.append(row)
    result: pd.DataFrame = pd.DataFrame(rows)
    metric_names: list[str] = []
    for metric in settings.metrics:
        metric_names.extend(column for column in result if column == metric or column.startswith(metric + " ("))
    other: list[str] = [column for column in result if column not in metric_names]

    return result[[*other, *metric_names]]


def _patch_summary(table: pd.DataFrame, settings: PaperTableConfig) -> pd.DataFrame:
    """Average native cohort-pooled patch metrics across folds, without animal CIs."""
    if table.empty:
        return pd.DataFrame()
    selected: pd.DataFrame = table.loc[table["DatasetRole"].isin(settings.roles)].copy()
    if selected.empty:
        return pd.DataFrame()
    if not selected["Aggregation"].eq("cohort_pooled").all() or not selected["EvaluationStage"].eq("patch").all():
        raise ValueError("Patch paper tables require native cohort_pooled patch measurements")
    if "Variant" not in selected:
        selected["Variant"] = selected.get("ModelVariant", "unspecified")
    keys: list[str] = ["Variant", "Seed", "DatasetRole", "Dataset", "Head", "Region", "Metric", "Unit"]
    if selected.duplicated([*keys, "Fold"]).any():
        raise ValueError("Duplicate native patch measures within a fold")
    selected["Value"] = pd.to_numeric(selected["Value"], errors="raise")
    selected.loc[~np.isfinite(selected["Value"]), "Value"] = np.nan
    return selected.groupby(keys, dropna=False).agg(
        Mean=("Value", "mean"), FoldSD=("Value", "std"), FiniteFolds=("Value", "count"),
        Folds=("Fold", "nunique"),
    ).reset_index()


def _classification_tables(table: pd.DataFrame, settings: PaperTableConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Retain exact animal intervals and add descriptive equal-fold rate means."""
    if table.empty:
        return pd.DataFrame(), pd.DataFrame()
    selected: pd.DataFrame = table.loc[table["DatasetRole"].isin(settings.roles)].copy()
    if selected.empty:
        return pd.DataFrame(), pd.DataFrame()
    if "Variant" not in selected:
        selected["Variant"] = selected.get("ModelVariant", "unspecified")
    report = classification_report(selected, confidence=settings.confidence)
    summaries: list[pd.DataFrame] = [report.summary]
    keys: list[str] = ["Variant", "Seed", "DatasetRole"]
    rows: list[dict[str, Any]] = []
    for key, group in report.summary.loc[report.summary["Scope"].eq("fold")].groupby(keys):
        if len(group) <= 1:
            continue
        animals = report.animals
        keep: pd.Series = pd.Series(True, index=animals.index)
        for column, value in zip(keys, key):
            keep &= animals[column].eq(value)
        row: dict[str, Any] = {
            **dict(zip(keys, key)), "Scope": "equal_fold_mean", "Fold": -2,
            "Folds": len(group), "Animals": animals.loc[keep, "Subject"].nunique(),
        }
        for metric in ("accuracy", "sensitivity", "specificity", "precision"):
            row[metric] = float(group[metric].mean())
            row[metric + "_fold_sd"] = float(group[metric].std(ddof=1))
            row[metric + "_folds"] = int(group[metric].count())
        rows.append(row)
    if rows:
        summaries.append(pd.DataFrame(rows))
    return pd.concat(summaries, ignore_index=True), report.confusion


def _agreement_summary(table: pd.DataFrame, settings: PaperTableConfig) -> pd.DataFrame:
    """Apply existing agreement statistics to independent animal mean volumes."""
    groups: list[str] = ["Variant", "Seed", "DatasetRole", "Dataset", "Condition", "Head"]
    rows: list[pd.DataFrame] = []
    config = AgreementConfig(
        unit="mm3", confidence=settings.confidence,
        bootstrap_samples=settings.bootstrap_samples, seed=settings.seed,
    )
    for key, group in table.groupby(groups, dropna=False):
        pairs = volume_pairs(group)
        if pairs.empty:
            continue
        for region, animals in pairs.groupby("Region"):
            row: dict[str, Any] = {**dict(zip(groups, key)), "Region": region, "n": len(animals), "Status": "too_few_animals"}
            if len(animals) >= 3:
                row.update(agreement_table(animals, config).iloc[0].to_dict())
                row["Status"] = "ok"
            rows.append(pd.DataFrame([row]))
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def build_paper_tables(
    full_volume_metrics: pd.DataFrame, *, patch_metrics: pd.DataFrame | None = None,
    classifier_predictions: pd.DataFrame | None = None, settings: PaperTableConfig = PaperTableConfig(),
    expected_folds: int | None = None,
) -> PaperTables:
    """Summarize existing measurements, keeping roles/seeds/cohorts distinct.

    Longitudinal scans receive equal weight within their animal. Fold means
    give each evaluable animal equal weight, then folds receive equal weight.
    Missing/undefined scores remain missing with per-metric denominators.
    Complete-annotation macro rows are the default; individual ROI rows retain
    all valid partial annotations. No losses or model predictions are changed.
    """
    required: set[str] = {"Subject", "Scan", "Dataset", "Region", "Metric", "Value", "Unit", "Status", "Fold", "Seed", "DatasetRole"}
    if full_volume_metrics.empty or not required.issubset(full_volume_metrics):
        raise ValueError("Select saved full-volume metrics with explicit scan/fold/seed/role identifiers")
    table: pd.DataFrame = _normalized_metrics(full_volume_metrics)
    table = table.loc[table["DatasetRole"].isin(settings.roles)].copy()
    if table.empty:
        raise ValueError(f"No full-volume rows match paper roles {settings.roles}")
    table["Value"] = pd.to_numeric(table["Value"], errors="raise")
    identifiers: list[str] = sorted(required - {"Value"})
    if table[identifiers].isna().any().any():
        raise ValueError("Paper metrics require nonmissing scan/fold/seed/role identifiers")
    if table.loc[table["Status"].eq("unavailable"), "Value"].notna().any():
        raise ValueError("Unavailable annotations cannot have numeric paper-table scores")
    identities: list[str] = [*GROUPS[:-1], "Fold", "Subject", "Scan"]
    if table.duplicated(identities).any():
        raise ValueError("Duplicate full-volume scan metrics; select each saved result once")
    holdout: pd.DataFrame = table.loc[table["DatasetRole"].eq("holdout")]
    memberships: pd.DataFrame = holdout[["Variant", "Seed", "Subject", "Fold"]].drop_duplicates()
    if memberships.duplicated(["Variant", "Seed", "Subject"]).any():
        raise ValueError("An animal occurs in multiple outer holdouts for the same variant/seed")
    measures: pd.DataFrame = table.loc[table["Metric"].isin(settings.metrics)].copy()
    if measures.empty:
        raise ValueError("None of the configured paper metrics are present")
    statistics, folds = _summarize(measures, settings)
    macros: pd.DataFrame = region_set_scores(table)
    if settings.complete_region_sets_only and not macros.empty:
        macros = macros.loc[macros["ExpectedRegions"].gt(0) & macros["AnnotatedRegions"].eq(macros["ExpectedRegions"])].copy()
    macro_statistics: pd.DataFrame = pd.DataFrame()
    macro_folds: pd.DataFrame = pd.DataFrame()
    if not macros.empty:
        macros["Head"] = "region_set"
        macros = macros.rename(columns={"RegionSet": "Region"})
        macros["Status"] = "undefined"
        macros.loc[np.isfinite(macros["Value"]), "Status"] = "ok"
        macros = macros.loc[macros["Metric"].isin(settings.metrics)].copy()
        macro_statistics, macro_folds = _summarize(macros, settings)
    all_statistics: pd.DataFrame = pd.concat([statistics, macro_statistics], ignore_index=True)
    all_folds: pd.DataFrame = pd.concat([folds, macro_folds], ignore_index=True)
    patches: pd.DataFrame = pd.DataFrame()
    if patch_metrics is not None:
        patches = _patch_summary(patch_metrics, settings)
    classifier: pd.DataFrame = pd.DataFrame()
    confusion: pd.DataFrame = pd.DataFrame()
    if classifier_predictions is not None:
        classifier, confusion = _classification_tables(classifier_predictions, settings)
    coverage_keys: list[str] = ["Variant", "Seed", "DatasetRole"]
    coverage = table.groupby(coverage_keys, dropna=False).agg(
        ObservedFolds=("Fold", "nunique"), ObservedAnimals=("Subject", "nunique"),
    ).reset_index()
    coverage["ExpectedFolds"] = expected_folds
    if expected_folds is not None:
        coverage["Complete"] = coverage["ObservedFolds"].eq(expected_folds)
    _, healthy = healthy_lesion_report(table, confidence=settings.confidence)
    comparisons: pd.DataFrame = pd.DataFrame()
    if settings.reference_variant is not None and table["Variant"].nunique() > 1:
        if settings.reference_variant not in set(table["Variant"]):
            raise ValueError("Paper reference variant is not present in the selected results")
        compare_settings = RevisionConfig(
            roles=settings.roles, metrics=settings.metrics, confidence=settings.confidence,
            bootstrap_samples=settings.bootstrap_samples, seed=settings.seed,
        )
        comparison_rows: list[pd.DataFrame] = []
        scores: pd.DataFrame = measures.loc[~measures["Metric"].isin(["ReferenceVolume", "PredictionVolume"])]
        for key, group in scores.groupby(["Seed", "Condition", "Head"], dropna=False):
            candidates: list[str] = sorted(set(group["Variant"]) - {settings.reference_variant})
            if settings.reference_variant not in set(group["Variant"]):
                continue
            for candidate in candidates:
                compared = paired_model_comparison(group, settings.reference_variant, candidate, compare_settings)
                if not compared.empty:
                    compared = compared.assign(Seed=key[0], Condition=key[1], Head=key[2])
                    comparison_rows.append(compared)
        if comparison_rows:
            comparisons = pd.concat(comparison_rows, ignore_index=True)
            valid: pd.Series = comparisons["WilcoxonP"].dropna().sort_values()
            factors: np.ndarray = np.arange(len(valid), 0, -1)
            adjusted: np.ndarray = np.minimum(1.0, np.maximum.accumulate(valid.to_numpy() * factors))
            comparisons["HolmP"] = np.nan
            comparisons.loc[valid.index, "HolmP"] = adjusted

    return PaperTables(
        segmentation=_compact(statistics, settings), region_sets=_compact(macro_statistics, settings),
        animal_intervals=_compact(all_statistics, settings, animal_intervals=True), statistics=all_statistics,
        fold_metrics=all_folds, patches=patches, classification=classifier, confusion=confusion,
        agreement=_agreement_summary(table, settings), fold_coverage=coverage, settings=settings,
        comparisons=comparisons, healthy_lesion=healthy,
    )


def paper_tables_from_folder(folder: Path, *, settings: PaperTableConfig = PaperTableConfig()) -> PaperTables:
    """Read one explicit run, study or checkpoint-evaluation folder; never select latest.

    A run's summary/ is preferred to per-evaluator files. Existing outputs are
    read only; export is an explicit separate operation in a fresh folder.
    """
    root: Path = Path(folder).resolve()
    locations: list[Path] = [root / "summary", root, root / "regions"]
    metric_file: Path | None = None
    for location in locations:
        candidate: Path = location / "full_volume_metrics.csv"
        if candidate.is_file() and candidate.stat().st_size > 10:
            metric_file = candidate
            break
    if metric_file is None:
        raise ValueError("No saved full-volume summary found. Finish evaluation or select a checkpoint-evaluation folder")
    selected_files: list[Path] = [metric_file]
    metrics: pd.DataFrame = pd.read_csv(metric_file)
    manifest_file: Path = root / "subjects.csv"
    if manifest_file.is_file() and manifest_file.stat().st_size > 10:
        manifest: pd.DataFrame = pd.read_csv(manifest_file)
        metrics = attach_scan_metadata(metrics, manifest)
        selected_files.append(manifest_file)
    extra: dict[str, pd.DataFrame] = {}
    for name in ("patch_metrics", "classifier_predictions"):
        path: Path = metric_file.parent / f"{name}.csv"
        table: pd.DataFrame = pd.DataFrame()
        if path.is_file() and path.stat().st_size > 10:
            table = pd.read_csv(path)
            selected_files.append(path)
        extra[name] = table
    expected_folds: int | None = None
    split_file: Path = root / "splits.json"
    if split_file.is_file():
        splits = json.loads(split_file.read_text(encoding="utf-8"))
        expected_folds = 1
        if isinstance(splits, list):
            expected_folds = len(splits)
        selected_files.append(split_file)
    evaluation_file: Path = root / "evaluation.json"
    experiment_file: Path = root / "experiment.json"
    metadata: dict[str, Any] = {}
    metadata_file: Path | None = None
    if evaluation_file.exists():
        metadata_file = evaluation_file
        expected_folds = 1
    elif experiment_file.exists():
        metadata_file = experiment_file
    if metadata_file is not None:
        metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
        selected_files.append(metadata_file)
    result = build_paper_tables(
        metrics, patch_metrics=extra["patch_metrics"], classifier_predictions=extra["classifier_predictions"],
        settings=settings, expected_folds=expected_folds,
    )
    cost_sources: list[CostSource] = []
    for name in ("timings.csv", "summary/fold_timings.csv", "inference_timings.csv"):
        path = root / name
        if not path.is_file() or path.stat().st_size <= 10:
            continue
        variant: str | None = None
        seed: int | None = None
        fold: int | None = None
        runtime_file: Path | None = None
        if evaluation_file.exists():
            variant = metadata.get("variant")
            seed = metadata.get("seed")
            fold = metadata.get("fold")
            runtime_file = evaluation_file
        elif metadata:
            variant = metadata.get("model", {}).get("architecture")
            seed = metadata.get("training", {}).get("seed")
            if (root / "metadata.json").is_file():
                runtime_file = root / "metadata.json"
        role: str | None = None
        workload: str | None = None
        if evaluation_file.exists():
            role = metadata.get("role")
            workload = "+".join(sorted(metrics["Dataset"].unique()))
        cost_sources.append(CostSource(
            path=path, name=variant, seed=seed, fold=fold, metadata_file=runtime_file,
            role=role, dataset=workload,
        ))
    if cost_sources:
        costs = cost_report(cost_sources)
        result.costs = costs.summary
        selected_files.extend(Path(path) for path in costs.sources["Path"])
    records: list[dict[str, Any]] = []
    for path in sorted(set(selected_files)):
        content: bytes = path.read_bytes()
        records.append({"Path": str(path.resolve()), "SHA256": hashlib.sha256(content).hexdigest()})
    result.sources = pd.DataFrame(records)

    return result
