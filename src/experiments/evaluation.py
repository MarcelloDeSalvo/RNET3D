"""Best-checkpoint native patch measurements and common evaluation exports."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .data import PreparedData


def evaluate_patches(
    model: Any, data: PreparedData, subject_ids: Sequence[str],
    options: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Evaluate every saved deterministic patch using the compiled training graph.

    Native Keras losses and metrics retain their training definitions and cohort
    aggregation. They are not animal-level full-volume scores. Available-class
    soft Dice is additionally exposed from the existing partial-label accumulators.
    No optimizer updates, augmentation, truncation or postprocessing occur.
    """
    from ..evaluation.metrics import PartialLabelDice

    options = dict(options or {})
    ids = list(subject_ids)
    keys = data.subjects.scan_keys(ids)
    if data.config.loss.regions != "dice_ce_gradient_fix":
        partial = any(data.loader.get_subject(key).unavailable_labels for key in keys)
        if partial:
            raise ValueError("Native patch evaluation of partial annotations requires the saved gradient-fixed objective")
    generator = data.make_generator(ids, training=False)
    measurements = model.evaluate(generator, verbose=0, return_dict=True, workers=0)
    datasets = sorted({data.loader.get_subject(key).subject_group[0] for key in keys})
    dataset_name = "+".join(datasets)
    animal_count = len(ids)
    scan_count = len(keys)
    patch_count = len(generator.ids)
    common = {
        "Subject": "", "Scan": "", "Dataset": dataset_name,
        "EvaluationStage": "patch", "Aggregation": "cohort_pooled",
        "AnimalCount": animal_count, "ScanCount": scan_count, "PatchCount": patch_count,
        "Unit": "unitless",
    }
    rows: list[dict[str, Any]] = []
    for name, value in measurements.items():
        head = "total"
        region = "All heads"
        if name.startswith("regions"):
            head, region = "regions", "Regions"
        elif name.startswith("brain_mask"):
            head, region = "brain_mask", "Brain"
        elif name.startswith("classifier"):
            head, region = "classifier", "Condition"
        metric_value = float(value)
        row = {**common, "Head": head, "Region": region, "Metric": name,
               "Value": metric_value, "Status": "ok"}
        rows.append(row)

    for metric in model.metrics:
        if not isinstance(metric, PartialLabelDice):
            continue
        union = metric.prediction_mass.numpy() + metric.truth_mass.numpy()
        scores = (2 * metric.intersection.numpy() + metric.smooth) / (union + metric.smooth)
        available = metric.annotated_voxels.numpy() > 0
        first_class = 0
        if metric.exclude_background:
            first_class = 1
        for channel in range(first_class, metric.num_classes):
            status = "unavailable"
            value = np.nan
            if available[channel]:
                status = "ok"
                value = float(scores[channel])
            rows.append({**common, "Head": "regions", "Region": data.config.labels.names[channel],
                         "Label": channel, "Metric": "SoftDice", "Value": value, "Status": status})

    table = pd.DataFrame(rows)
    if "evaluation_dir" in options:
        folder = Path(options["evaluation_dir"])
        folder.mkdir(parents=True, exist_ok=True)
        data.subjects.manifest(ids).to_csv(folder / "subjects.csv", index=False)
    return table


def export_evaluation_tables(
    patch_metrics: pd.DataFrame, full_volume_metrics: pd.DataFrame, folder: Path,
    *, classifier_predictions: pd.DataFrame | None = None,
    manifest: pd.DataFrame | None = None, split_manifest: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Save separate native/full-volume tables and one workbook with both stages."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    tables = {"patch_metrics": patch_metrics, "full_volume_metrics": full_volume_metrics}
    for name, table in tables.items():
        table.to_csv(folder / f"{name}.csv", index=False)
    nonempty = [table for table in tables.values() if not table.empty]
    combined = pd.DataFrame()
    if nonempty:
        combined = pd.concat(nonempty, ignore_index=True)
        combined.to_csv(folder / "evaluation_metrics.csv", index=False)
        from ..evaluation.classification import export_classification_tables

        classifiers = pd.DataFrame()
        if classifier_predictions is not None:
            classifiers = classifier_predictions
        extra_tables = export_classification_tables(classifiers, full_volume_metrics, folder)
        from ..evaluation.reporting import evaluation_report

        report = evaluation_report(full_volume_metrics, manifest=manifest, split_manifest=split_manifest)
        report_tables = report.tables()
        for name, table in report_tables.items():
            table.to_csv(folder / f"{name}.csv", index=False)
        with pd.ExcelWriter(folder / "evaluation_metrics.xlsx") as writer:
            workbook_tables = {**tables, **extra_tables, **report_tables}
            for name, table in workbook_tables.items():
                table.to_excel(writer, sheet_name=name, index=False)
    return combined
