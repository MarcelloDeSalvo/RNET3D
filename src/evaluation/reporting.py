"""Descriptive anatomy, cohort, participant and structure-size evaluation tables."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Mapping, Sequence
import json

import numpy as np
import pandas as pd


SCAN_METADATA: tuple[str, ...] = (
    "OriginalID", "Condition", "Modality", "Timepoint", "Subgroup", "Strain", "Species", "Protocol", "FusedBoundary",
)
MODEL_COLUMNS: tuple[str, ...] = ("Variant", "Seed", "Fold", "DatasetRole")


def attach_scan_metadata(table: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    """Join explicit metadata without duplicating rows or overwriting known facts.

    FusedBoundary is an optional expert annotation, never inferred from model
    predictions. Manifests must identify scans uniquely within each dataset.
    """
    result: pd.DataFrame = table.copy()
    if table.empty or manifest.empty:
        return result
    keys: list[str] = ["Subject", "Scan"]
    if "Dataset" in table and "Dataset" in manifest:
        keys.append("Dataset")
    if not set(keys).issubset(table) or not set(keys).issubset(manifest):
        raise ValueError("Scan metadata requires Subject and Scan identifiers")
    columns: list[str] = [column for column in SCAN_METADATA if column in manifest]
    metadata: pd.DataFrame = manifest[[*keys, *columns]].drop_duplicates()
    if metadata[keys].isna().any().any() or metadata.duplicated(keys).any():
        raise ValueError("Scan metadata must contain exactly one consistent record per scan")
    result = result.merge(metadata, on=keys, how="left", suffixes=("", "_metadata"), validate="many_to_one")
    for column in columns:
        alternate: str = column + "_metadata"
        if alternate not in result:
            continue
        known: pd.Series = result[column].notna() & ~result[column].astype(str).isin(["", "unknown", "unspecified"])
        supplied: pd.Series = result[alternate].notna() & ~result[alternate].astype(str).isin(["", "unknown", "unspecified"])
        conflict: pd.Series = known & supplied & result[column].astype(str).ne(result[alternate].astype(str))
        if conflict.any():
            raise ValueError(f"Conflicting scan metadata for {column}")
        result.loc[~known, column] = result.loc[~known, alternate]
        result = result.drop(columns=alternate)

    return result


def _normalized_metrics(table: pd.DataFrame) -> pd.DataFrame:
    """Normalize reporting metadata only; preserve all measurement values."""
    result: pd.DataFrame = table.copy()
    defaults: dict[str, Any] = {
        "Variant": "unspecified", "Seed": -1, "Fold": -1, "DatasetRole": "unspecified",
        "Condition": "unknown", "Head": "regions",
    }
    if "Variant" not in result and "ModelVariant" in result:
        result["Variant"] = result["ModelVariant"]
    for column, value in defaults.items():
        if column not in result:
            result[column] = value
    for column in SCAN_METADATA:
        if column in result:
            result[column] = result[column].fillna("unknown").replace("", "unknown")
    names: pd.Series = result["Region"].str.strip().str.casefold()
    result.loc[names.eq("brain"), "Head"] = "brain_mask"
    background: pd.Series = names.eq("background")
    if "Label" in result:
        background |= result["Head"].eq("regions") & result["Label"].eq(0)
    result = result.loc[~background].copy()

    return result


def participant_flow(manifest: pd.DataFrame) -> pd.DataFrame:
    """Count animals and scans separately within explicit cohort/split strata.

    Timepoint/sequence rows are not additive animal counts: an animal may occur
    in several strata. Dataset totals are provided as distinct Scope rows.
    """
    required: set[str] = {"Subject", "Scan", "Dataset"}
    if manifest.empty or not required.issubset(manifest):
        return pd.DataFrame()
    base: list[str] = [column for column in MODEL_COLUMNS if column in manifest]
    strata: list[str] = [column for column in ("Condition", "Modality", "Timepoint", "Subgroup") if column in manifest]
    keys: list[str] = [*base, "Dataset"]
    columns: list[str] = list(dict.fromkeys([*keys, *strata, "Subject", "Scan"]))
    selected: pd.DataFrame = manifest[columns].copy()
    for column in strata:
        selected[column] = selected[column].fillna("unknown").replace("", "unknown")
    identity: list[str] = [*keys, "Subject", "Scan"]
    selected = selected.drop_duplicates()
    if selected.duplicated(identity).any():
        raise ValueError("Participant metadata conflicts for the same scan and split")
    tables: list[pd.DataFrame] = []
    for scope, groups in (("dataset_total", keys), ("stratum", [*keys, *strata])):
        grouped = selected.groupby(groups, dropna=False)
        animals = grouped["Subject"].nunique()
        scans = grouped.size()
        table: pd.DataFrame = pd.DataFrame({"Animals": animals, "Scans": scans}).reset_index()
        table["Scope"] = scope
        tables.append(table)

    return pd.concat(tables, ignore_index=True)


def split_scan_manifest(manifest: pd.DataFrame, splits: Mapping | Sequence[Mapping]) -> pd.DataFrame:
    """Expand saved fixed/CV memberships to all scans, preserving fold roles."""
    entries: list[Mapping] = []
    cross_validation: bool = not isinstance(splits, Mapping)
    if cross_validation:
        entries = list(splits)
    else:
        entries = [splits]
    tables: list[pd.DataFrame] = []
    for fold, entry in enumerate(entries):
        for saved_role, subjects in entry.items():
            if saved_role not in ("train", "validation", "test", "external"):
                continue
            role: str = saved_role
            if cross_validation and saved_role == "test":
                role = "holdout"
            unknown: set[str] = set(subjects) - set(manifest["Subject"])
            if unknown:
                raise ValueError(f"Split contains subjects absent from the manifest: {sorted(unknown)}")
            selected: pd.DataFrame = manifest.loc[manifest["Subject"].isin(subjects)].copy()
            selected["DatasetRole"] = role
            selected["Fold"] = fold
            tables.append(selected)

    if not tables:
        return pd.DataFrame()
    return pd.concat(tables, ignore_index=True)


def _animal_summary(table: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    """Average model repetitions per scan, then scans per animal in each stratum."""
    valid: pd.DataFrame = table.loc[np.isfinite(table["Value"])].copy()
    if valid.empty:
        return pd.DataFrame()
    scan_keys: list[str] = [*groups, "Subject", "Scan"]
    scans: pd.DataFrame = valid.groupby(scan_keys, dropna=False)["Value"].mean().reset_index()
    animals: pd.DataFrame = scans.groupby([*groups, "Subject"], dropna=False).agg(
        Value=("Value", "mean"), Scans=("Scan", "nunique"),
    ).reset_index()
    summary: pd.DataFrame = animals.groupby(groups, dropna=False).agg(
        Animals=("Subject", "nunique"), Scans=("Scans", "sum"), Mean=("Value", "mean"),
        SD=("Value", "std"), Minimum=("Value", "min"), Maximum=("Value", "max"),
    ).reset_index()

    return summary


def region_set_scores(table: pd.DataFrame) -> pd.DataFrame:
    """Compute explicit scan macro scores with annotated/finite region counts.

    Available-region macros describe their stated subset, not unannotated TBI
    anatomy. CompleteRegionSet means every expected region has a finite score.
    Background is excluded; brain is included only in explicitly named sets.
    """
    metrics: tuple[str, ...] = ("Dice", "IoU", "HD95", "ASSD", "AVE", "RVE", "Sensitivity", "Precision")
    selected: pd.DataFrame = table.loc[table["Metric"].isin(metrics)].copy()
    keys: list[str] = [*MODEL_COLUMNS, "Dataset", "Condition", "Subject", "Scan", "Metric", "Unit"]
    rows: list[dict[str, Any]] = []
    region_sets: dict[str, tuple[bool, bool]] = {
        "all_roi": (True, False), "anatomy_only": (False, False),
        "all_roi_and_brain": (True, True), "anatomy_and_brain": (False, True),
    }
    for key, group in selected.groupby(keys, dropna=False):
        for name, (include_lesion, include_brain) in region_sets.items():
            keep: pd.Series = pd.Series(True, index=group.index)
            if not include_lesion:
                lesion: pd.Series = group["Region"].str.strip().str.casefold().eq("lesion")
                if "Label" in group:
                    lesion |= group["Head"].eq("regions") & group["Label"].eq(1)
                keep &= ~lesion
            if not include_brain:
                keep &= group["Head"].ne("brain_mask")
            regions: pd.DataFrame = group.loc[keep]
            finite: pd.DataFrame = regions.loc[np.isfinite(regions["Value"])]
            expected: int = len(regions)
            annotated: int = int(regions["Status"].ne("unavailable").sum())
            value: float = float("nan")
            if not finite.empty:
                value = float(finite["Value"].mean())
            rows.append({
                **dict(zip(keys, key)), "RegionSet": name, "Value": value,
                "ExpectedRegions": expected, "AnnotatedRegions": annotated, "FiniteRegions": len(finite),
                "CompleteRegionSet": bool(expected > 0 and len(finite) == expected),
                "ContributingRegions": json.dumps(sorted(finite["Region"].tolist())),
                "Aggregation": "scan_macro_available_regions",
            })

    return pd.DataFrame(rows)


def structure_size_table(table: pd.DataFrame) -> pd.DataFrame:
    """Pair scan scores with reference/prediction volumes without choosing cutoffs."""
    keys: list[str] = [*MODEL_COLUMNS, "Dataset", "Condition", "Subject", "Scan", "Head", "Region"]
    volumes: pd.DataFrame = table.loc[table["Metric"].isin(["ReferenceVolume", "PredictionVolume"])]
    if volumes.empty:
        return pd.DataFrame()
    if set(volumes["Unit"]) != {"mm3"}:
        raise ValueError("Structure-size reports require physical mm3 volumes")
    paired: pd.DataFrame = volumes.pivot(index=keys, columns="Metric", values="Value").reset_index()
    for name in ("ReferenceVolume", "PredictionVolume"):
        if name not in paired:
            paired[name] = np.nan
    measures: tuple[str, ...] = ("Dice", "HD95", "ASSD", "Sensitivity", "Precision")
    scores: pd.DataFrame = table.loc[table["Metric"].isin(measures)].copy()
    result: pd.DataFrame = scores.merge(paired, on=keys, how="left", validate="many_to_one")
    reference: pd.Series = result["ReferenceVolume"]
    prediction: pd.Series = result["PredictionVolume"]
    result["MissedStructure"] = pd.Series(pd.NA, index=result.index, dtype="boolean")
    known: pd.Series = np.isfinite(reference) & np.isfinite(prediction)
    result.loc[known, "MissedStructure"] = reference[known].gt(0) & prediction[known].eq(0)

    return result


@dataclass
class EvaluationReport:
    """Additional descriptive tables; original metric formulas remain unchanged."""

    region_set_scans: pd.DataFrame
    region_set_summary: pd.DataFrame
    region_coverage: pd.DataFrame
    subgroup_summary: pd.DataFrame
    participant_flow: pd.DataFrame
    evaluation_flow: pd.DataFrame
    split_flow: pd.DataFrame
    structure_size_metrics: pd.DataFrame

    def tables(self) -> dict[str, pd.DataFrame]:
        """Expose stable CSV/workbook names for all evaluation entry points."""
        return {item.name: getattr(self, item.name) for item in fields(self)}


def evaluation_report(
    metrics: pd.DataFrame, *, manifest: pd.DataFrame | None = None,
    split_manifest: pd.DataFrame | None = None,
) -> EvaluationReport:
    """Build the same descriptive additions for runs, studies and saved evaluations."""
    inventory: pd.DataFrame = pd.DataFrame()
    if manifest is not None:
        inventory = manifest
    memberships: pd.DataFrame = pd.DataFrame()
    if split_manifest is not None:
        memberships = split_manifest
    required: set[str] = {"Subject", "Scan", "Dataset", "Region", "Metric", "Value", "Unit", "Status"}
    empty: dict[str, pd.DataFrame] = {item.name: pd.DataFrame() for item in fields(EvaluationReport)}
    empty["participant_flow"] = participant_flow(inventory)
    empty["split_flow"] = participant_flow(memberships)
    if metrics.empty or not required.issubset(metrics):
        return EvaluationReport(**empty)
    enriched: pd.DataFrame = attach_scan_metadata(metrics, inventory)
    selected: pd.DataFrame = _normalized_metrics(enriched)
    ids: list[str] = [*MODEL_COLUMNS, "Dataset", "Subject", "Scan", "Head", "Region", "Metric"]
    if selected.duplicated(ids).any():
        raise ValueError("Duplicate full-volume measurements in descriptive reports")
    sets: pd.DataFrame = region_set_scores(selected)
    base: list[str] = ["Variant", "DatasetRole", "Dataset", "Condition"]
    set_summary: pd.DataFrame = pd.DataFrame()
    if not sets.empty:
        groups: list[str] = [*base, "RegionSet", "Metric", "Unit"]
        set_summary = _animal_summary(sets, groups)
    coverage_keys: list[str] = [*base, "Head", "Region", "Metric", "Unit", "Status"]
    coverage_rows: pd.DataFrame = selected.copy()
    coverage_rows["FiniteMeasurement"] = np.isfinite(coverage_rows["Value"])
    coverage: pd.DataFrame = coverage_rows.groupby(coverage_keys, dropna=False).agg(
        Animals=("Subject", "nunique"), Measurements=("Value", "size"), Finite=("FiniteMeasurement", "sum"),
    ).reset_index()
    summaries: list[pd.DataFrame] = []
    for column in ("Condition", "Modality", "Timepoint", "Subgroup", "Strain", "FusedBoundary"):
        if column not in selected:
            continue
        subgroup: pd.DataFrame = selected.copy()
        subgroup["GroupValue"] = subgroup[column].astype(str)
        groups = ["Variant", "DatasetRole", "Dataset", "GroupValue", "Head", "Region", "Metric", "Unit"]
        summary: pd.DataFrame = _animal_summary(subgroup, groups)
        if not summary.empty:
            summary.insert(3, "GroupBy", column)
            summaries.append(summary)
    subgroup_summary: pd.DataFrame = pd.DataFrame()
    if summaries:
        subgroup_summary = pd.concat(summaries, ignore_index=True)
    return EvaluationReport(
        region_set_scans=sets, region_set_summary=set_summary, region_coverage=coverage,
        subgroup_summary=subgroup_summary, participant_flow=empty["participant_flow"],
        evaluation_flow=participant_flow(selected), split_flow=empty["split_flow"],
        structure_size_metrics=structure_size_table(selected),
    )


def structure_size_figure(report: EvaluationReport, *, region: str, metric: str = "Dice") -> Any:
    """Explore scan/model scores against GT volume with explicit repeated models."""
    import plotly.express as px

    table: pd.DataFrame = report.structure_size_metrics
    if table.empty:
        raise ValueError("No paired full-volume scores/volumes are available")
    selected: pd.DataFrame = table.loc[table["Region"].eq(region) & table["Metric"].eq(metric)].copy()
    selected = selected.loc[np.isfinite(selected["Value"]) & np.isfinite(selected["ReferenceVolume"])]
    hover: list[str] = ["Subject", "Scan", "Seed", "Fold", "Condition", "Status", "MissedStructure"]
    figure = px.scatter(
        selected, x="ReferenceVolume", y="Value", color="Dataset", symbol="Variant",
        hover_data=hover, title=f"{region}: {metric} versus reference volume (mm3)",
        labels={"ReferenceVolume": "Reference volume (mm3)", "Value": metric},
    )

    return figure
