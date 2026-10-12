"""Typed inference outputs and evaluation tables, independent of plotting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
import pandas as pd


@dataclass(frozen=True)
class PredictionResult:
    """Full-volume ROI labels and optional brain/classification outputs."""

    roi: NDArray[Any]
    brain_mask: NDArray[Any] | None = None
    classifier: bool | None = None

    def as_dict(self) -> dict[str, NDArray[Any] | bool]:
        """Expose the historical dictionary keys, retaining false classifier results."""
        outputs: dict[str, NDArray[Any] | bool] = {}
        if self.brain_mask is not None:
            outputs['brain_mask'] = self.brain_mask
        if self.classifier is not None:
            outputs['classifier'] = self.classifier
        outputs['roi'] = self.roi
        return outputs


@dataclass(frozen=True)
class EvaluationResult:
    """Raw scan metrics and the legacy per-metric report with class means.

    metrics has a Subject index and Metric/Stride/Modality/class columns.
    summary has Subject as a column plus Mean over all class columns, including
    background. Frames are ordinary mutable pandas objects, not immutable data.
    No timepoint, subject, stride or fold aggregation is performed.
    """

    metrics: pd.DataFrame
    summary: pd.DataFrame

    @staticmethod
    def with_class_means(table: pd.DataFrame) -> pd.DataFrame:
        """Add a Mean column to a copy, using only explicitly named class columns."""
        class_columns: list[str] = [column for column in table if column.startswith('Class: ')]
        if not class_columns:
            raise ValueError('Evaluation tables require at least one class column')
        result: pd.DataFrame = table.copy(deep=True)
        result['Mean'] = result[class_columns].mean(axis=1)
        return result

    @classmethod
    def from_metrics(cls, metrics: pd.DataFrame) -> EvaluationResult:
        """Copy validated metric rows and calculate class means without plotting."""
        required: set[str] = {'Metric', 'Stride', 'Modality'}
        if metrics.empty or metrics.index.name != 'Subject' or not required.issubset(metrics.columns):
            raise ValueError('Expected nonempty metrics with a Subject index and Metric/Stride/Modality columns')
        raw: pd.DataFrame = metrics.copy(deep=True)
        tables: list[pd.DataFrame] = []
        for metric in raw['Metric'].unique():
            subset: pd.DataFrame = raw.loc[raw['Metric'] == metric].sort_index()
            subset = cls.with_class_means(subset)
            tables.append(subset.reset_index())
        summary: pd.DataFrame = pd.concat(tables)
        return cls(raw, summary)
