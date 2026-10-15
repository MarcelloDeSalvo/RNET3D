"""Animal-level revision tables from explicit saved full-volume metric sources.

Scans are averaged within animals, after averaging repeated model predictions
for each scan. These summaries describe animal mean performance/volume; they
do not model longitudinal trajectories or estimate training-seed uncertainty.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence
import hashlib
import json

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from .statistics import AgreementConfig, agreement_table
from .classification import healthy_lesion_report
from .reporting import EvaluationReport, attach_scan_metadata, evaluation_report


@dataclass(frozen=True)
class MetricSource:
    """One saved long-form CSV, with explicit metadata for missing columns.

    Keep name=None to retain Variant values in a study CSV. External baseline
    tables must use canonical animal IDs and the same region names/units.
    """

    path: Path
    name: str | None = None
    role: str | None = None
    seed: int | None = None
    fold: int | None = None
    manifest: Path | None = None


@dataclass(frozen=True)
class CostSource:
    """One saved training/inference timing CSV with explicit run provenance.

    Study CSVs already contain Variant/Seed. Single-run files need name/seed;
    inference files also need fold. metadata_file optionally supplies the run's
    metadata.json or a checkpoint evaluation's evaluation.json. Missing runtime
    provenance stays unknown. role/dataset distinguish inference workloads.
    """

    path: Path
    name: str | None = None
    seed: int | None = None
    fold: int | None = None
    metadata_file: Path | None = None
    role: str | None = None
    dataset: str | None = None


@dataclass
class CostReport:
    """Recorded cost measurements, descriptive summaries and input hashes."""

    measurements: pd.DataFrame = field(default_factory=pd.DataFrame)
    summary: pd.DataFrame = field(default_factory=pd.DataFrame)
    sources: pd.DataFrame = field(default_factory=pd.DataFrame)


@dataclass(frozen=True)
class RevisionConfig:
    """Prespecified cohort roles and animal bootstrap settings."""

    roles: tuple[str, ...] = ("test", "holdout", "external")
    metrics: tuple[str, ...] = ("Dice", "HD95", "ASSD", "AVE")
    bootstrap_samples: int = 2000
    confidence: float = 0.95
    seed: int = 42
    failure_dice_threshold: float = 0.5

    def __post_init__(self) -> None:
        """Reject undefined resampling settings before reading result files."""
        if not self.roles or not self.metrics:
            raise ValueError("Select explicit roles and metrics")
        if self.bootstrap_samples < 100 or not 0 < self.confidence < 1:
            raise ValueError("Use at least 100 bootstrap samples and a confidence between zero and one")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("Bootstrap seed must be a nonnegative integer")
        if not np.isfinite(self.failure_dice_threshold) or not 0 <= self.failure_dice_threshold <= 1:
            raise ValueError("failure_dice_threshold must be finite and between zero and one")


@dataclass
class RevisionResult:
    """Auditable raw rows and summaries; unavailable values remain missing."""

    raw: pd.DataFrame
    animals: pd.DataFrame
    summary: pd.DataFrame
    statuses: pd.DataFrame
    volumes: pd.DataFrame
    agreement: pd.DataFrame
    comparisons: pd.DataFrame = field(default_factory=pd.DataFrame)
    sham_lesion_animals: pd.DataFrame = field(default_factory=pd.DataFrame)
    sham_lesion_summary: pd.DataFrame = field(default_factory=pd.DataFrame)
    coverage: pd.DataFrame = field(default_factory=pd.DataFrame)
    failure_cases: pd.DataFrame = field(default_factory=pd.DataFrame)
    patch_metrics: pd.DataFrame = field(default_factory=pd.DataFrame)
    costs: CostReport = field(default_factory=CostReport)
    sources: pd.DataFrame = field(default_factory=pd.DataFrame)
    analysis_config: RevisionConfig | None = None
    reports: EvaluationReport = field(default_factory=lambda: evaluation_report(pd.DataFrame()))


def _source_record(path: Path, kind: str) -> dict[str, str | int]:
    """Identify an exact reporting input without modifying the source file."""
    resolved: Path = Path(path).resolve()
    content: bytes = resolved.read_bytes()
    digest: str = hashlib.sha256(content).hexdigest()

    return {"Kind": kind, "Path": str(resolved), "SHA256": digest, "Bytes": len(content)}


def metric_coverage(table: pd.DataFrame) -> pd.DataFrame:
    """Count observed animals/scans and finite measurements, including missing GT.

    Counts describe supplied files. Entirely absent scans require a separate
    expected manifest and cannot be inferred from metric rows alone.
    """
    keys: list[str] = ["Variant", "DatasetRole", "Dataset", "Region", "Metric", "Unit"]
    rows: list[dict[str, Any]] = []
    for key, group in table.groupby(keys, dropna=False):
        finite: pd.Series = np.isfinite(group["Value"])
        unavailable: pd.Series = group["Status"].eq("unavailable")
        scans: pd.DataFrame = group[["Subject", "Scan"]].drop_duplicates()
        finite_scans: pd.DataFrame = group.loc[finite, ["Subject", "Scan"]].drop_duplicates()

        rows.append({
            **dict(zip(keys, key)), "ObservedAnimals": group["Subject"].nunique(),
            "ObservedScans": len(scans), "FiniteAnimals": group.loc[finite, "Subject"].nunique(),
            "FiniteScans": len(finite_scans), "RawRows": len(group),
            "UnavailableRows": int(unavailable.sum()),
            "UndefinedRows": int((~finite & ~unavailable).sum()),
        })

    return pd.DataFrame(rows)


def failure_case_table(table: pd.DataFrame, *, dice_threshold: float = 0.5) -> pd.DataFrame:
    """List scan measurements for review without diagnosing or excluding failures.

    Retain seed/fold/source identity. Both-empty conventions and unavailable
    annotations are not failures; empty predictions and positive predictions
    on empty references deserve inspection. The threshold only selects Dice
    review candidates and never changes metrics or model selection.
    """
    if not np.isfinite(dice_threshold) or not 0 <= dice_threshold <= 1:
        raise ValueError("dice_threshold must be finite and between zero and one")

    cases: pd.DataFrame = table.copy()
    finite: pd.Series = np.isfinite(cases["Value"])
    cases["ReviewReason"] = ""
    low_dice: pd.Series = cases["Metric"].eq("Dice") & finite & cases["Value"].lt(dice_threshold)
    cases.loc[low_dice, "ReviewReason"] = "low_dice"
    undefined: pd.Series = ~finite & cases["Status"].eq("ok")
    cases.loc[undefined, "ReviewReason"] = "undefined_measurement"
    for status in ("reference_empty", "prediction_empty"):
        selected: pd.Series = cases["Status"].eq(status)
        cases.loc[selected, "ReviewReason"] = status

    cases = cases.loc[cases["ReviewReason"].ne("")].copy()
    cases["ReviewDiceThreshold"] = dice_threshold
    return cases.reset_index(drop=True)


def load_patch_sources(sources: Sequence[MetricSource]) -> pd.DataFrame:
    """Collect native pooled patch results separately from animal inference.

    These are already compiled cohort measurements. Preserve their formulas
    and counts; do not average them as if they were animal observations.
    """
    tables: list[pd.DataFrame] = []
    required: set[str] = {"DatasetRole", "Metric", "Value", "EvaluationStage", "Aggregation"}
    for source in sources:
        table: pd.DataFrame = pd.read_csv(source.path)
        missing: set[str] = required - set(table.columns)
        if missing or table.empty:
            raise ValueError(f"{source.path}: missing native patch columns or rows: {sorted(missing)}")
        if not table["EvaluationStage"].eq("patch").all() or not table["Aggregation"].eq("cohort_pooled").all():
            raise ValueError("Select native cohort_pooled patch tables, separately from scan metrics")

        metadata: dict[str, str | int | None] = {
            "Variant": source.name, "Seed": source.seed, "Fold": source.fold,
        }
        for column, value in metadata.items():
            if column == "Variant" and value is not None:
                table[column] = value
            elif column not in table:
                if value is None:
                    raise ValueError(f"{source.path}: provide explicit {column}")
                table[column] = value
        table["Value"] = pd.to_numeric(table["Value"], errors="raise")
        table["SourceFile"] = str(Path(source.path).resolve())
        tables.append(table)

    if not tables:
        return pd.DataFrame()
    combined: pd.DataFrame = pd.concat(tables, ignore_index=True)
    keys: list[str] = ["Variant", "Seed", "Fold", "DatasetRole", "Metric"]
    for column in ("Head", "Region", "Label"):
        if column in combined:
            keys.append(column)
    if combined[keys[:5]].isna().any().any() or combined.duplicated(keys).any():
        raise ValueError("Missing patch identifiers or duplicate pooled patch measurements")

    return combined


def cost_report(sources: Sequence[CostSource]) -> CostReport:
    """Summarize saved wall-clock costs without fitting or benchmarking again.

    Training rows remain per fold; inference rows remain per scan/model. Device
    names are recorded physical-device identifiers, not inferred GPU models.
    Missing device/precision provenance prevents controlled hardware claims.
    """
    rows: list[dict[str, Any]] = []
    provenance: list[dict[str, str | int]] = []
    training_columns: tuple[str, ...] = (
        "training_seconds", "evaluation_seconds", "epochs_completed", "parameter_count", "best_epoch",
    )
    inference_columns: tuple[str, ...] = (
        "PreprocessingSeconds", "InferenceSeconds", "PostprocessingSeconds",
    )

    for source in sources:
        table: pd.DataFrame = pd.read_csv(source.path)
        kind: str = "training"
        columns: tuple[str, ...] = training_columns
        if "training_seconds" not in table:
            kind = "inference"
            columns = inference_columns
        if table.empty or not set(columns).issubset(table.columns):
            raise ValueError(f"{source.path}: expected a saved fold or inference timing table")
        if kind == "inference" and not {"Subject", "Scan"}.issubset(table.columns):
            raise ValueError("Inference timings require Subject and Scan")
        if kind == "inference" and table[["Subject", "Scan"]].isna().any().any():
            raise ValueError("Inference Subject and Scan identifiers must not be missing")

        metadata: dict[str, Any] = {}
        if source.metadata_file is not None:
            metadata_path: Path = Path(source.metadata_file)
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            provenance.append(_source_record(metadata_path, "cost_metadata"))
        runtime: dict[str, Any] = metadata.get("runtime", metadata)
        devices: list[str] = runtime.get("devices", runtime.get("gpu_devices", []))
        device: str = "unknown"
        if devices:
            device = ", ".join(devices)
        precision: str = runtime.get("precision", "unknown")
        provenance_status: str = "recorded"
        if device == "unknown" or precision == "unknown":
            provenance_status = "missing_hardware_metadata"

        constants: dict[str, str | int | None] = {
            "Variant": source.name, "Seed": source.seed, "Fold": source.fold,
        }
        if "fold" in table and "Fold" not in table:
            table = table.rename(columns={"fold": "Fold"})
        for column, value in constants.items():
            if column == "Variant" and value is not None:
                table[column] = value
            elif column not in table:
                if value is None:
                    raise ValueError(f"{source.path}: provide explicit {column}")
                table[column] = value
        if table[list(constants)].isna().any().any():
            raise ValueError("Timing identifiers must not be missing")
        for column in ("Seed", "Fold"):
            numbers: pd.Series = pd.to_numeric(table[column], errors="raise")
            valid: pd.Series = np.isfinite(numbers) & numbers.ge(0) & numbers.eq(np.floor(numbers))
            if not valid.all():
                raise ValueError("Timing Seed and Fold must be nonnegative integers")
            table[column] = numbers.astype(int)
        if "DatasetRole" not in table:
            table["DatasetRole"] = "unknown"
            if kind == "training":
                table["DatasetRole"] = "training"
            elif source.role is not None:
                table["DatasetRole"] = source.role
        if "Dataset" not in table:
            table["Dataset"] = "unknown"
            if source.dataset is not None:
                table["Dataset"] = source.dataset

        provenance.append(_source_record(source.path, "costs"))
        for record in table.to_dict(orient="records"):
            for metric in columns:
                value: float = float(record[metric])
                if not np.isfinite(value) or value < 0:
                    raise ValueError(f"{source.path}: {metric} must be finite and nonnegative")
                unit: str = "s"
                if kind == "training" and metric not in ("training_seconds", "evaluation_seconds"):
                    unit = "count"
                    if not value.is_integer():
                        raise ValueError(f"{metric} must be an integer count")
                status: str = "recorded"
                if metric == "parameter_count" and value == 0:
                    status = "not_recorded"
                    value = float("nan")
                rows.append({
                    **{column: record[column] for column in constants},
                    "Kind": kind, "Subject": record.get("Subject", ""), "Scan": record.get("Scan", ""),
                    "DatasetRole": record["DatasetRole"], "Dataset": record["Dataset"],
                    "Device": device, "Precision": precision, "ProvenanceStatus": provenance_status,
                    "Metric": metric, "Value": value, "Unit": unit, "Status": status,
                    "SourceFile": str(Path(source.path).resolve()),
                })

    if not rows:
        return CostReport()
    measurements: pd.DataFrame = pd.DataFrame(rows)
    keys: list[str] = ["Variant", "Seed", "Fold", "Kind", "DatasetRole", "Dataset", "Subject", "Scan", "Metric"]
    if measurements.duplicated(keys).any():
        raise ValueError("Duplicate cost measurements: select each saved timing table once")
    groups: list[str] = [
        "Variant", "Kind", "DatasetRole", "Dataset", "Device", "Precision", "ProvenanceStatus", "Metric", "Unit",
    ]
    summary: pd.DataFrame = measurements.groupby(groups, dropna=False).agg(
        Recorded=("Value", "count"), Rows=("Value", "size"), Mean=("Value", "mean"),
        Median=("Value", "median"), Minimum=("Value", "min"), Maximum=("Value", "max"),
    ).reset_index()
    source_table: pd.DataFrame = pd.DataFrame(provenance).drop_duplicates().reset_index(drop=True)

    return CostReport(measurements=measurements, summary=summary, sources=source_table)


def load_metric_sources(sources: Sequence[MetricSource]) -> pd.DataFrame:
    """Read explicit files, preserving statuses and rejecting duplicated measures."""
    tables = []
    required = {"Subject", "Scan", "Dataset", "Region", "Metric", "Value", "Unit", "Status"}
    for source in sources:
        table = pd.read_csv(source.path)
        if "EvaluationStage" in table:
            table = table.loc[table["EvaluationStage"] == "full_volume"].copy()
        if "Evaluator" in table:
            selected_evaluators: pd.Series = table["Evaluator"].isin(["regions", "brain_mask"])
            table = table.loc[selected_evaluators].copy()
        missing = required - set(table.columns)
        if missing:
            raise ValueError(f"{source.path}: missing columns {sorted(missing)}")

        metadata = {"Variant": source.name, "DatasetRole": source.role,
                    "Seed": source.seed, "Fold": source.fold}
        for column, value in metadata.items():
            if column == "Variant" and value is not None:
                table[column] = value
            elif column not in table:
                if value is None:
                    raise ValueError(f"{source.path}: provide explicit {column}")
                table[column] = value
        identifiers = sorted(required - {"Value"}) + list(metadata)
        if table.empty or table[identifiers].isna().any().any():
            raise ValueError(f"{source.path}: missing identifiers or no ROI measurements")
        table["Value"] = pd.to_numeric(table["Value"], errors="raise")
        unavailable = table["Status"] == "unavailable"
        if table.loc[unavailable, "Value"].notna().any():
            raise ValueError("Unavailable annotations must have missing metric values")
        table["SourceFile"] = str(Path(source.path).resolve())
        if source.manifest is not None:
            manifest = pd.read_csv(source.manifest)
            table = attach_scan_metadata(table, manifest)
        tables.append(table)

    if not tables:
        raise ValueError("Select at least one metric source")
    combined = pd.concat(tables, ignore_index=True)
    keys = ["Variant", "Seed", "Fold", "DatasetRole", "Subject", "Scan", "Dataset", "Region", "Metric", "Unit"]
    if combined.duplicated(keys).any():
        raise ValueError("Duplicate scan metrics: select each saved result once")
    return combined


def animal_results(table: pd.DataFrame) -> pd.DataFrame:
    """Average finite model repetitions per scan, then scans per biological animal.

    Keep dataset, region, role and units separate. A finite both-empty Dice is
    retained; undefined distances and unavailable labels are never set to zero.
    """
    selected = table.loc[np.isfinite(table["Value"])].copy()
    keys = ["Variant", "DatasetRole", "Dataset", "Region", "Metric", "Unit", "Subject"]
    scan_keys = [*keys, "Scan"]
    scans = selected.groupby(scan_keys, dropna=False)["Value"].mean().reset_index()
    animals = scans.groupby(keys, dropna=False).agg(Value=("Value", "mean"), Scans=("Scan", "nunique"))
    return animals.reset_index()


def _mean_interval(values: np.ndarray, config: RevisionConfig) -> tuple[float, float]:
    """Resample independent animal values for a percentile interval of their mean."""
    if len(values) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(config.seed)
    estimates = np.empty(config.bootstrap_samples)
    for index in range(config.bootstrap_samples):
        sample = rng.integers(0, len(values), len(values))
        estimates[index] = values[sample].mean()
    alpha = (1 - config.confidence) / 2
    lower, upper = np.quantile(estimates, [alpha, 1 - alpha])
    return float(lower), float(upper)


def summarize_animals(animals: pd.DataFrame, config: RevisionConfig) -> pd.DataFrame:
    """Report animal counts, descriptive means and animal bootstrap intervals."""
    keys = ["Variant", "DatasetRole", "Dataset", "Region", "Metric", "Unit"]
    rows = []
    for key, group in animals.groupby(keys, dropna=False):
        values = group["Value"].to_numpy(dtype=float)
        lower, upper = _mean_interval(values, config)
        sd = float("nan")
        if len(values) > 1:
            sd = float(values.std(ddof=1))
        rows.append({**dict(zip(keys, key)), "Animals": len(group), "Mean": values.mean(),
                     "SD": sd, "CILower": lower, "CIUpper": upper})
    return pd.DataFrame(rows)


def volume_pairs(table: pd.DataFrame) -> pd.DataFrame:
    """Pair physical volumes before averaging repetitions and longitudinal scans.

    The estimand is each animal's mean scan volume. Missing annotations are
    excluded; genuine zero-volume references remain valid measurements.
    """
    selected = table.loc[table["Metric"].isin(["ReferenceVolume", "PredictionVolume"])].copy()
    selected = selected.loc[np.isfinite(selected["Value"]) & (selected["Status"] != "unavailable")]
    if selected.empty:
        return pd.DataFrame()
    if set(selected["Unit"]) != {"mm3"}:
        raise ValueError("Volume agreement requires physical mm3 measurements")
    keys = ["Variant", "Seed", "Fold", "DatasetRole", "Dataset", "Region", "Subject", "Scan"]
    paired = selected.pivot(index=keys, columns="Metric", values="Value").reset_index()
    volume_columns = ["ReferenceVolume", "PredictionVolume"]
    if not set(volume_columns).issubset(paired):
        return pd.DataFrame()
    paired = paired.dropna(subset=volume_columns)
    shared_reference = ["DatasetRole", "Dataset", "Region", "Subject", "Scan"]
    for _, group in paired.groupby(shared_reference, dropna=False):
        reference = group["ReferenceVolume"].to_numpy()
        if not np.allclose(reference, reference[0], rtol=1e-6, atol=1e-8):
            raise ValueError("Compared models use different reference volumes for the same scan")

    animal_keys = ["Variant", "DatasetRole", "Dataset", "Region", "Subject"]
    scans = paired.groupby([*animal_keys, "Scan"], dropna=False)[volume_columns].mean().reset_index()
    animals = scans.groupby(animal_keys, dropna=False).agg(
        Manual=("ReferenceVolume", "mean"), Automated=("PredictionVolume", "mean"),
        Scans=("Scan", "nunique"),
    )
    return animals.reset_index()


def paired_model_comparison(
    table: pd.DataFrame, reference: str, candidate: str, config: RevisionConfig,
) -> pd.DataFrame:
    """Compare models on identical scans, then pair animals and adjust p-values.

    Report candidate-reference differences without assigning a better direction.
    Wilcoxon tests are exploratory; Holm correction covers all returned groups.
    Intervals condition on the supplied trained models and sampled animals.
    """
    if reference == candidate or not {reference, candidate}.issubset(set(table["Variant"])):
        raise ValueError("Select two distinct model variants present in the results")
    selected = table.loc[table["Variant"].isin([reference, candidate])].copy()
    selected = selected.loc[np.isfinite(selected["Value"])]
    keys = ["DatasetRole", "Dataset", "Region", "Metric", "Unit", "Subject", "Scan"]
    scans = selected.groupby([*keys, "Variant"], dropna=False)["Value"].mean().reset_index()
    paired = scans.pivot(index=keys, columns="Variant", values="Value").reset_index()
    if not {reference, candidate}.issubset(paired):
        return pd.DataFrame()
    paired = paired.dropna(subset=[reference, candidate])
    groups = ["DatasetRole", "Dataset", "Region", "Metric", "Unit"]
    aggregation = {reference: (reference, "mean"), candidate: (candidate, "mean"),
                   "PairedScans": ("Scan", "nunique")}
    animals = paired.groupby([*groups, "Subject"], dropna=False).agg(**aggregation).reset_index()
    rows = []
    for key, group in animals.groupby(groups, dropna=False):
        difference = (group[candidate] - group[reference]).to_numpy(dtype=float)
        lower, upper = _mean_interval(difference, config)
        pvalue = float("nan")
        status = "too_few_animals"
        if len(group) >= 3:
            status = "ok"
            pvalue = 1.0
            if np.any(difference != 0):
                pvalue = float(wilcoxon(difference, method="auto").pvalue)
        rows.append({**dict(zip(groups, key)), "Reference": reference, "Candidate": candidate,
                     "PairedAnimals": len(group), "MeanDifference": difference.mean(),
                     "PairedScans": int(group["PairedScans"].sum()),
                     "CILower": lower, "CIUpper": upper, "WilcoxonP": pvalue, "Status": status})
    result = pd.DataFrame(rows)
    if not result.empty:
        result["HolmP"] = np.nan
        valid = result["WilcoxonP"].dropna().sort_values()
        factors = np.arange(len(valid), 0, -1)
        adjusted = np.minimum(1.0, np.maximum.accumulate(valid.to_numpy() * factors))
        result.loc[valid.index, "HolmP"] = adjusted
    return result


def revision_report(
    sources: Sequence[MetricSource], config: RevisionConfig = RevisionConfig(), *,
    comparison: tuple[str, str] | None = None,
    patch_sources: Sequence[MetricSource] = (), cost_sources: Sequence[CostSource] = (),
) -> RevisionResult:
    """Collect saved evaluation evidence without rerunning training or inference.

    Full-volume uncertainty counts animals. Optional native patch rows and cost
    tables are exported separately and never enter the animal bootstrap.
    """
    raw = load_metric_sources(sources)
    selected = raw.loc[raw["DatasetRole"].isin(config.roles)].copy()
    if selected.empty:
        raise ValueError("No measurements match the selected dataset roles")
    measures = selected.loc[selected["Metric"].isin(config.metrics)].copy()
    animals = animal_results(measures)
    summary = summarize_animals(animals, config)
    status_keys = ["Variant", "DatasetRole", "Dataset", "Region", "Metric", "Unit", "Status"]
    statuses = selected.groupby(status_keys, dropna=False).size().reset_index(name="RawRows")
    volumes = volume_pairs(selected)
    agreements = []
    keys = ["Variant", "DatasetRole", "Dataset", "Region"]
    if not volumes.empty:
        settings = AgreementConfig(
            unit="mm3", bootstrap_samples=config.bootstrap_samples,
            confidence=config.confidence, seed=config.seed,
        )
        for key, group in volumes.groupby(keys, dropna=False):
            if len(group) < 3:
                agreements.append({**dict(zip(keys, key)), "n": len(group), "Status": "too_few_animals"})
                continue
            report = agreement_table(group, settings).iloc[0].to_dict()
            agreements.append({**dict(zip(keys, key)), **report, "Status": "ok"})
    comparisons = pd.DataFrame()
    if comparison is not None:
        reference, candidate = comparison
        comparisons = paired_model_comparison(measures, reference, candidate, config)
    healthy_animals, healthy_summary = healthy_lesion_report(selected, confidence=config.confidence)
    coverage: pd.DataFrame = metric_coverage(selected)
    cases: pd.DataFrame = failure_case_table(selected, dice_threshold=config.failure_dice_threshold)
    patches: pd.DataFrame = load_patch_sources(patch_sources)
    costs: CostReport = cost_report(cost_sources)
    source_records: list[dict[str, str | int]] = []
    for source in sources:
        source_records.append(_source_record(source.path, "full_volume_metrics"))
        if source.manifest is not None:
            source_records.append(_source_record(source.manifest, "scan_metadata"))
    for source in patch_sources:
        source_records.append(_source_record(source.path, "patch_metrics"))
    provenance: list[pd.DataFrame] = [pd.DataFrame(source_records)]
    if not costs.sources.empty:
        provenance.append(costs.sources)
    source_files: pd.DataFrame = pd.concat(provenance, ignore_index=True).drop_duplicates()
    manifest_tables: list[pd.DataFrame] = []
    for source in sources:
        if source.manifest is not None:
            manifest_tables.append(pd.read_csv(source.manifest))
    inventory = pd.DataFrame()
    if manifest_tables:
        inventory = pd.concat(manifest_tables, ignore_index=True).drop_duplicates()
    reports = evaluation_report(selected, manifest=inventory)

    return RevisionResult(
        raw=raw, animals=animals, summary=summary, statuses=statuses,
        volumes=volumes, agreement=pd.DataFrame(agreements), comparisons=comparisons,
        sham_lesion_animals=healthy_animals, sham_lesion_summary=healthy_summary,
        coverage=coverage, failure_cases=cases, patch_metrics=patches,
        costs=costs, sources=source_files, analysis_config=config,
        reports=reports,
    )


def export_revision(result: RevisionResult, output_dir: Path, config: RevisionConfig) -> Path:
    """Save CSVs, an Excel workbook and input hashes in a fresh output folder.

    Raw values retain numeric precision. Undefined/unavailable measurements
    remain blank, and patch or timing rows never become animal observations.
    """
    from dataclasses import asdict

    folder = Path(output_dir)
    if result.analysis_config is not None and result.analysis_config != config:
        raise ValueError("Reporting configuration changed since analysis; rerun analysis before exporting")
    for source in result.sources.to_dict(orient="records"):
        current: dict[str, str | int] = _source_record(Path(source["Path"]), source["Kind"])
        if current["SHA256"] != source["SHA256"]:
            raise ValueError(f"Reporting input changed since analysis: {source['Path']}; rerun analysis")
    folder.mkdir(parents=True, exist_ok=False)
    tables = {
        "raw_metrics": result.raw, "animal_metrics": result.animals, "metric_summary": result.summary,
        "metric_statuses": result.statuses, "animal_volume_pairs": result.volumes,
        "volume_agreement": result.agreement, "paired_comparisons": result.comparisons,
        "sham_lesion_animals": result.sham_lesion_animals,
        "sham_lesion_summary": result.sham_lesion_summary,
        "metric_coverage": result.coverage, "failure_review": result.failure_cases,
        "native_patch_metrics": result.patch_metrics, "cost_measurements": result.costs.measurements,
        "cost_summary": result.costs.summary, "source_files": result.sources,
        **result.reports.tables(),
    }
    for name, table in tables.items():
        table.to_csv(folder / f"{name}.csv", index=False)
    with pd.ExcelWriter(folder / "revision_tables.xlsx") as writer:
        for name, table in tables.items():
            table.to_excel(writer, sheet_name=name, index=False)
    payload = {"analysis": asdict(config), "unit_of_analysis": "biological animal",
               "aggregation": "model repetitions per scan, then longitudinal scans per animal",
               "sources": sorted(result.raw["SourceFile"].unique().tolist()),
               "source_files": result.sources.to_dict(orient="records"),
               "coverage_scope": "observed rows only; entirely missing scans are not inferred",
               "failure_review": "descriptive candidates; no automatic exclusions or diagnosis",
               "cost_scope": "saved wall-clock records; no new benchmark or memory measurement"}
    (folder / "analysis.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    notes: str = (
        "# Reporting evidence\n\n"
        "Full-volume estimates count biological animals after the recorded aggregation.\n"
        "Native patch results are pooled cohort measurements and are exported separately.\n"
        "Inspect metric coverage, empty-mask statuses and failure_review before selecting examples.\n"
        "Coverage describes supplied rows; scans absent from every input cannot be counted.\n"
        "Failure-review thresholds do not change scores, model selection or exclusions.\n"
        "Cost summaries describe supplied timings; missing hardware metadata stays unknown.\n"
        "Peak memory, manual review findings and missing experiments are not inferred.\n"
        "Volume agreement is computed only from supplied paired reference/prediction measurements.\n"
        "This export does not establish that the independent D4 manual-volume analysis is complete.\n"
        "No new training, inference or interpolation sensitivity study was performed.\n"
    )
    (folder / "reporting_notes.md").write_text(notes, encoding="utf-8")
    return folder
