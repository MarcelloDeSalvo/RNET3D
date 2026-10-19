"""Read saved run histories and prepare animal-level plotting data without fitting."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
import json
import re

import numpy as np
import pandas as pd

from .reporting import attach_scan_metadata


@dataclass
class RunPlotData:
    """Saved loss observations, full-volume scan measurements and input paths."""

    history: pd.DataFrame
    metrics: pd.DataFrame
    sources: tuple[Path, ...]


def _fold_folders(run_dir: Path) -> list[Path]:
    """List direct fold directories in numeric order, excluding other runs."""
    folders: list[Path] = []
    for path in run_dir.glob("fold_*"):
        if path.is_dir() and re.fullmatch(r"fold_\d+", path.name):
            folders.append(path)
    return sorted(folders, key=lambda path: int(path.name[5:]))


def _load_history(folder: Path) -> tuple[pd.DataFrame, Path | None]:
    """Prefer committed recovery history; otherwise read saved CSV epochs."""
    state_file: Path = folder / "training_state.json"
    source: Path | None = None
    table: pd.DataFrame = pd.DataFrame()
    if state_file.is_file():
        state = json.loads(state_file.read_text(encoding="utf-8"))
        table = pd.DataFrame(state["history"])
        source = state_file
    else:
        for filename in ("training.csv", "history.csv"):
            path = folder / filename
            if path.is_file():
                table = pd.read_csv(path)
                source = path
                break

    if table.empty:
        return pd.DataFrame(), source

    if "epoch" in table:
        table = table.drop_duplicates("epoch", keep="last").sort_values("epoch")
        table["Epoch"] = pd.to_numeric(table["epoch"], errors="raise") + 1
    else:
        table["Epoch"] = np.arange(1, len(table) + 1)

    keys: list[str] = []
    for column in table:
        if column in ("loss", "val_loss") or column.endswith("_loss"):
            keys.append(column)
    if not keys:
        return pd.DataFrame(), source

    history = table.melt(id_vars="Epoch", value_vars=keys, var_name="Loss", value_name="Value")
    history["Value"] = pd.to_numeric(history["Value"], errors="raise")
    history["Fold"] = int(folder.name[5:])
    history["SourceFile"] = str(source)
    return history, source


def load_run_plot_data(run_dir: Path, *, metrics_file: Path | None = None) -> RunPlotData:
    """Read one run, using one aggregate metric file or its per-fold fallback.

    An explicit metrics_file supports stopped-checkpoint evaluations. No latest
    run/evaluation is selected. Missing evaluations leave metrics empty, allowing
    loss inspection while training is unfinished. Saved metadata supplies condition
    and modality; absent facts remain 'unknown'. Hover metadata retains original
    IDs and strain, using saved c52/cd1 protocol codes when strain is unspecified.
    """
    run_dir = Path(run_dir).resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run folder does not exist: {run_dir}")

    folders = _fold_folders(run_dir)
    histories: list[pd.DataFrame] = []
    sources: list[Path] = []
    for folder in folders:
        history, source = _load_history(folder)
        if not history.empty:
            histories.append(history)
        if source is not None:
            sources.append(source)

    paths: list[Path] = []
    if metrics_file is not None:
        path = Path(metrics_file).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Metric file does not exist: {path}")
        paths.append(path)
    else:
        for filename in ("summary/full_volume_metrics.csv", "regions/full_volume_metrics.csv", "full_volume_metrics.csv"):
            path = run_dir / filename
            if path.is_file():
                paths.append(path)
                break
        if not paths:
            for folder in folders:
                paths.extend(sorted(folder.glob("evaluations/*/regions/metrics.json")))

    tables: list[pd.DataFrame] = []
    for path in paths:
        if path.suffix == ".json":
            table = pd.read_json(path, orient="table")
            table["Fold"] = int(path.parents[3].name[5:])
            table["DatasetRole"] = path.parents[1].name
        else:
            table = pd.read_csv(path)
        table["SourceFile"] = str(path)
        tables.append(table)
        sources.append(path)

    history = pd.DataFrame()
    metrics = pd.DataFrame()
    if histories:
        history = pd.concat(histories, ignore_index=True)
    if not tables:
        source_files = tuple(sources)
        return RunPlotData(history=history, metrics=metrics, sources=source_files)

    metrics = pd.concat(tables, ignore_index=True)
    required: set[str] = {"Subject", "Scan", "Dataset", "Region", "Metric", "Value", "Unit", "Status", "DatasetRole", "Fold"}
    missing = required - set(metrics)
    if missing:
        raise ValueError(f"Full-volume metrics are missing columns: {sorted(missing)}")
    if "EvaluationStage" in metrics:
        metrics = metrics.loc[metrics["EvaluationStage"].eq("full_volume")].copy()

    keys = ["Subject", "Scan", "Dataset", "Region", "Metric", "Unit", "DatasetRole", "Fold"]
    if metrics[keys].isna().any().any() or metrics.duplicated(keys).any():
        raise ValueError("Missing metric identifiers or duplicate scan/fold measurements")
    metrics["Value"] = pd.to_numeric(metrics["Value"], errors="raise")
    unavailable = metrics["Status"].eq("unavailable")
    if metrics.loc[unavailable, "Value"].notna().any():
        raise ValueError("Unavailable annotations must have missing metric values")

    manifest_file = run_dir / "subjects.csv"
    if manifest_file.is_file():
        manifest = pd.read_csv(manifest_file)
        metrics = attach_scan_metadata(metrics, manifest)
        sources.append(manifest_file)

    contract_file = run_dir / "data_contract.json"
    if contract_file.is_file():
        records = json.loads(contract_file.read_text(encoding="utf-8"))
        metadata: list[dict[str, str]] = []
        for record in records:
            scan_metadata = record.get("metadata", {})
            metadata.append({
                "Subject": record["animal"], "Scan": record["scan"][1],
                "Condition": scan_metadata.get("condition", "unknown"),
                "Modality": scan_metadata.get("modality", "unknown"),
                "OriginalID": scan_metadata.get("original_id", "unknown"),
                "Strain": scan_metadata.get("strain", "unknown"),
                "Protocol": scan_metadata.get("protocol", "unknown"),
            })
        metrics = attach_scan_metadata(metrics, pd.DataFrame(metadata))
        sources.append(contract_file)

    for column in ("Condition", "Modality", "OriginalID", "Strain", "Protocol"):
        if column not in metrics:
            metrics[column] = "unknown"
        metrics[column] = metrics[column].fillna("unknown").replace("", "unknown")
    missing_strain = metrics["Strain"].isin(("unknown", "unspecified"))
    strain_protocol = metrics["Protocol"].str.casefold().isin(("c52", "cd1"))
    use_protocol = missing_strain & strain_protocol
    metrics.loc[use_protocol, "Strain"] = metrics.loc[use_protocol, "Protocol"]
    if "Head" not in metrics:
        metrics["Head"] = "regions"
        brain = metrics["Region"].str.casefold().eq("brain")
        metrics.loc[brain, "Head"] = "brain_mask"

    source_files = tuple(sources)
    return RunPlotData(history=history, metrics=metrics, sources=source_files)


def _hover_metadata(values: pd.Series) -> str:
    """List supplied metadata values without duplicating an animal's plot point."""
    known = values.dropna().astype(str)
    known = known.loc[~known.isin(("", "unknown", "unspecified"))]
    if known.empty:
        return "unknown"

    return ", ".join(sorted(known.unique()))


def animal_metric_points(
    metrics: pd.DataFrame, *, metric: str, roles: Sequence[str] = ("test", "holdout"),
) -> pd.DataFrame:
    """Average repeated predictions per scan, then scans per animal and stratum.

    Keep condition, modality, cohort, role, head, region and unit separate. Each
    point represents one animal within that stratum, even with longitudinal
    scans or repeated fixed-test predictions across folds. Missing/unavailable
    measurements are excluded; genuine zero and both-empty scores are retained.
    Original IDs and strains describe the resulting points without changing
    grouping or weights; multiple supplied values are listed together.
    """
    if metrics.empty:
        raise ValueError("No saved full-volume metrics; evaluate the run first or supply METRICS_FILE")
    selected = metrics.loc[metrics["Metric"].eq(metric) & metrics["DatasetRole"].isin(roles)].copy()
    valid = np.isfinite(selected["Value"]) & selected["Status"].ne("unavailable")
    selected = selected.loc[valid].copy()
    if selected.empty:
        available = sorted(metrics["DatasetRole"].unique())
        raise ValueError(f"No finite {metric} measurements for roles {tuple(roles)}; available roles: {available}")

    for column in ("OriginalID", "Strain"):
        if column not in selected:
            selected[column] = "unknown"

    keys: list[str] = ["Dataset", "DatasetRole", "Condition", "Modality", "Head", "Region", "Metric", "Unit", "Subject"]
    scan_keys = [*keys, "Scan"]
    scans = selected.groupby(scan_keys, dropna=False)["Value"].mean().reset_index()
    animals = scans.groupby(keys, dropna=False).agg(
        Value=("Value", "mean"), Scans=("Scan", "nunique"),
    ).reset_index()
    fold_counts = selected.groupby(keys, dropna=False)["Fold"].nunique().reset_index(name="Folds")
    metadata = selected.groupby(keys, dropna=False).agg(
        OriginalID=("OriginalID", _hover_metadata), Strain=("Strain", _hover_metadata),
    ).reset_index()
    animals = animals.merge(fold_counts, on=keys, validate="one_to_one")
    return animals.merge(metadata, on=keys, validate="one_to_one")
