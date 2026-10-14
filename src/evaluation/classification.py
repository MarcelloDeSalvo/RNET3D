"""Animal-level classifier reports and independent healthy-lesion diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import pandas as pd

from .statistics import calculate_classification


MODEL_KEYS: list[str] = ["Variant", "Seed", "DatasetRole", "Fold"]


@dataclass(frozen=True)
class ClassificationReport:
    """Scan decisions, one vote per animal, rate intervals and confusion counts."""

    scans: pd.DataFrame
    animals: pd.DataFrame
    summary: pd.DataFrame
    confusion: pd.DataFrame


def _animal_groups(animals: pd.DataFrame) -> Iterator[tuple[dict[str, Any], pd.DataFrame]]:
    """Yield model-specific groups and disjoint out-of-fold holdout pools.

    Repeated test/external animals across folds remain separate model estimates.
    Holdout pooling requires each animal to belong to exactly one fold.
    """
    for key, group in animals.groupby(MODEL_KEYS, dropna=False):
        metadata = dict(zip(MODEL_KEYS, key))
        yield {**metadata, "Scope": "fold"}, group
    holdout = animals.loc[animals["DatasetRole"] == "holdout"]
    pool_keys = MODEL_KEYS[:-1]
    for key, group in holdout.groupby(pool_keys, dropna=False):
        if group["Fold"].nunique() <= 1:
            continue
        if group["Subject"].duplicated().any():
            raise ValueError("Pooled classifier holdouts must contain each animal in exactly one fold")
        metadata = dict(zip(pool_keys, key))
        yield {**metadata, "Fold": -1, "Scope": "pooled_holdout"}, group


def classification_report(scans: pd.DataFrame, *, confidence: float = 0.95) -> ClassificationReport:
    """Summarize preserved scan decisions using majority votes across animal scans.

    Ties stay negative (SHAM=0); TBI is positive. Each variant/seed/role/fold
    supplies one decision per animal. Binomial intervals condition on the selected
    models, not independent patches/scans/seeds. No probability calibration or
    classifier gating of segmentation is introduced.
    """
    if not 0 < confidence < 1:
        raise ValueError("Classification confidence must lie between zero and one")
    if scans.empty:
        return ClassificationReport(scans.copy(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
    required = {*MODEL_KEYS, "Subject", "Scan", "Dataset", "Reference", "Prediction"}
    if not required.issubset(scans.columns) or scans[list(required)].isna().any().any():
        raise ValueError("Classifier predictions require explicit model/cohort/scan identities and binary decisions")
    for column in ("Variant", "DatasetRole", "Subject", "Scan", "Dataset"):
        if scans[column].astype(str).str.strip().eq("").any():
            raise ValueError("Classifier identities and cohort labels must not be empty")
    for column in ("Reference", "Prediction"):
        if not np.isin(scans[column], [0, 1]).all():
            raise ValueError("Classifier reference and prediction must be binary zero/one decisions")
    scan_keys = [*MODEL_KEYS, "Subject", "Dataset", "Scan"]
    if scans.duplicated(scan_keys).any():
        raise ValueError("Duplicate classifier scan predictions; select each source once")
    animal_keys = [*MODEL_KEYS, "Subject"]
    groups = scans.groupby(animal_keys, dropna=False)
    if groups["Reference"].nunique().gt(1).any():
        raise ValueError("An animal has conflicting SHAM/TBI references; select a consistent condition cohort")
    animals = groups.agg(
        Reference=("Reference", "first"), PositiveScanFraction=("Prediction", "mean"),
        ScanCount=("Scan", "size"),
    ).reset_index()
    animals["Prediction"] = (animals["PositiveScanFraction"] > 0.5).astype(int)
    summaries: list[dict[str, Any]] = []
    confusion_rows: list[dict[str, Any]] = []
    for metadata, group in _animal_groups(animals):
        scores = calculate_classification(group, confidence=confidence)
        scan_count = int(group["ScanCount"].sum())
        row = {**metadata, "Animals": scores.n, "Scans": scan_count,
               "AnimalAggregation": "majority_scan_vote_ties_SHAM", "Confidence": confidence}
        for metric in ("accuracy", "sensitivity", "specificity", "precision"):
            interval = scores.ci[metric]
            lower, upper = np.nan, np.nan
            if interval is not None:
                lower, upper = interval
            row[metric] = getattr(scores, metric)
            row[f"{metric}_ci_lower"] = lower
            row[f"{metric}_ci_upper"] = upper
        summaries.append(row)
        for reference in (0, 1):
            for prediction in (0, 1):
                count = scores.confusion_matrix[reference][prediction]
                confusion_rows.append({**metadata, "Reference": reference, "Prediction": prediction,
                                       "Count": count})
    return ClassificationReport(scans.copy(), animals, pd.DataFrame(summaries), pd.DataFrame(confusion_rows))


def load_classifier_predictions(paths: Sequence[Path]) -> pd.DataFrame:
    """Read complete saved decision tables; duplicate/identity checks occur in reporting."""
    tables: list[pd.DataFrame] = []
    for path in paths:
        table = pd.read_csv(path)
        source_file = str(Path(path).resolve())
        table = table.assign(SourceFile=source_file)
        tables.append(table)
    if not tables:
        return pd.DataFrame()
    return pd.concat(tables, ignore_index=True)


def healthy_lesion_report(
    metrics: pd.DataFrame, *, confidence: float = 0.95,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Report SHAM lesion false-positive volumes independently of classifier votes.

    Mean/max volumes are calculated per animal. Any nonzero predicted lesion in
    an available healthy scan is an animal false positive. Missing condition or
    unavailable lesion GT never establishes healthy/negative supervision.
    """
    required = {*MODEL_KEYS, "Subject", "Scan", "Dataset", "Condition", "Region", "Metric", "Value", "Unit", "Status"}
    if "Variant" not in metrics and "ModelVariant" in metrics:
        metrics = metrics.assign(Variant=metrics["ModelVariant"])
    if metrics.empty or not required.issubset(metrics.columns):
        return pd.DataFrame(), pd.DataFrame()
    selected = metrics.loc[
        (metrics["Condition"] == "SHAM") & (metrics["Region"] == "Lesion")
        & (metrics["Metric"] == "PredictionVolume") & (metrics["Status"] != "unavailable")
    ].copy()
    if selected.empty:
        return pd.DataFrame(), pd.DataFrame()
    if not selected["Unit"].eq("mm3").all():
        raise ValueError("Healthy lesion volume reporting requires cubic millimetres")
    if not selected["Status"].isin(["reference_empty", "both_empty"]).all():
        raise ValueError("SHAM lesion diagnostics require an available, empty lesion reference")
    if not np.isfinite(selected["Value"]).all() or (selected["Value"] < 0).any():
        raise ValueError("Healthy lesion volumes must be finite and nonnegative")
    scan_keys = [*MODEL_KEYS, "Subject", "Dataset", "Scan"]
    if selected.duplicated(scan_keys).any():
        raise ValueError("Duplicate healthy lesion scan volumes")
    animal_keys = [*MODEL_KEYS, "Subject"]
    animals = selected.groupby(animal_keys, dropna=False).agg(
        MeanLesionVolumeMM3=("Value", "mean"), MaxLesionVolumeMM3=("Value", "max"),
        ScanCount=("Scan", "size"),
    ).reset_index()
    animals["Reference"] = 0
    animals["Prediction"] = (animals["MaxLesionVolumeMM3"] > 0).astype(int)
    rows: list[dict[str, Any]] = []
    for metadata, group in _animal_groups(animals):
        scores = calculate_classification(group, confidence=confidence)
        interval = scores.ci["specificity"]
        lower, upper = np.nan, np.nan
        if interval is not None:
            lower = 1 - interval[1]
            upper = 1 - interval[0]
        scan_count = int(group["ScanCount"].sum())
        false_positive_animals = int(group["Prediction"].sum())
        false_positive_rate = 1 - scores.specificity
        mean_volume = float(group["MeanLesionVolumeMM3"].mean())
        maximum_volume = float(group["MaxLesionVolumeMM3"].max())
        row = {
            **metadata, "Animals": scores.n, "Scans": scan_count,
            "FalsePositiveAnimals": false_positive_animals,
            "FalsePositiveAnimalRate": false_positive_rate,
            "RateCILower": lower, "RateCIUpper": upper,
            "MeanAnimalLesionVolumeMM3": mean_volume,
            "MaxLesionVolumeMM3": maximum_volume,
            "Confidence": confidence, "AnimalAggregation": "any_positive_scan",
        }
        rows.append(row)
    return animals, pd.DataFrame(rows)


def export_classification_tables(
    scans: pd.DataFrame, metrics: pd.DataFrame, folder: Path, *, confidence: float = 0.95,
) -> dict[str, pd.DataFrame]:
    """Save auditable classifier and healthy-lesion tables without segmentation gating."""
    report = classification_report(scans, confidence=confidence)
    healthy_animals, healthy_summary = healthy_lesion_report(metrics, confidence=confidence)
    tables = {
        "classifier_predictions": report.scans, "classifier_animals": report.animals,
        "classifier_summary": report.summary, "classifier_confusion": report.confusion,
        "sham_lesion_animals": healthy_animals, "sham_lesion_summary": healthy_summary,
    }
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        if not table.empty:
            table.to_csv(folder / f"{name}.csv", index=False)
    return {name: table for name, table in tables.items() if not table.empty}
