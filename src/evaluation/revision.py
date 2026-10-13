"""Animal-level revision tables from explicit saved full-volume metric sources.

Scans are averaged within animals, after averaging repeated model predictions
for each scan. These summaries describe animal mean performance/volume; they
do not model longitudinal trajectories or estimate training-seed uncertainty.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence
import json

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from .statistics import AgreementConfig, agreement_table


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


@dataclass(frozen=True)
class RevisionConfig:
    """Prespecified cohort roles and animal bootstrap settings."""

    roles: tuple[str, ...] = ("test", "holdout", "external")
    metrics: tuple[str, ...] = ("Dice", "HD95", "ASSD", "AVE")
    bootstrap_samples: int = 2000
    confidence: float = 0.95
    seed: int = 42

    def __post_init__(self) -> None:
        """Reject undefined resampling settings before reading result files."""
        if not self.roles or not self.metrics:
            raise ValueError("Select explicit roles and metrics")
        if self.bootstrap_samples < 100 or not 0 < self.confidence < 1:
            raise ValueError("Use at least 100 bootstrap samples and a confidence between zero and one")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("Bootstrap seed must be a nonnegative integer")


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


def load_metric_sources(sources: Sequence[MetricSource]) -> pd.DataFrame:
    """Read explicit files, preserving statuses and rejecting duplicated measures."""
    tables = []
    required = {"Subject", "Scan", "Dataset", "Region", "Metric", "Value", "Unit", "Status"}
    for source in sources:
        table = pd.read_csv(source.path)
        if "Evaluator" in table:
            table = table.loc[table["Evaluator"] == "regions"].copy()
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
) -> RevisionResult:
    """Produce status audits, animal uncertainty, volume agreement and comparisons."""
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
    return RevisionResult(
        raw=raw, animals=animals, summary=summary, statuses=statuses,
        volumes=volumes, agreement=pd.DataFrame(agreements), comparisons=comparisons,
    )


def export_revision(result: RevisionResult, output_dir: Path, config: RevisionConfig) -> Path:
    """Export auditable manuscript tables into a fresh folder without overwriting."""
    from dataclasses import asdict

    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=False)
    tables = {
        "raw_metrics": result.raw, "animal_metrics": result.animals, "metric_summary": result.summary,
        "metric_statuses": result.statuses, "animal_volume_pairs": result.volumes,
        "volume_agreement": result.agreement, "paired_comparisons": result.comparisons,
    }
    for name, table in tables.items():
        table.to_csv(folder / f"{name}.csv", index=False)
    payload = {"analysis": asdict(config), "unit_of_analysis": "biological animal",
               "aggregation": "model repetitions per scan, then longitudinal scans per animal",
               "sources": sorted(result.raw["SourceFile"].unique().tolist())}
    (folder / "analysis.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return folder
