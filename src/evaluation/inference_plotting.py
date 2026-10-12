"""Plotting and disk reports for computed inference evaluation results."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import pandas as pd
import seaborn as sns

from .inference_results import EvaluationResult


class EvaluationReportWriter:
    """Save tables and draw reports without owning models or computing metrics."""

    def __init__(self, output_dir: str | Path | None = None) -> None:
        """Store an optional output folder; create it only when writing files."""
        self.output_dir: Path | None = None
        if output_dir is not None:
            self.output_dir = Path(output_dir)

    def save_table(self, table: pd.DataFrame, filename: str) -> None:
        """Write an Excel table when a destination is configured."""
        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            destination: Path = self.output_dir / filename
            table.to_excel(destination)

    def write(self, result: EvaluationResult, *, show: bool = True) -> None:
        """Export existing raw/summary values, per-class plots and the overall plot."""
        if self.output_dir is None:
            raise ValueError('An output directory is required to write a report')
        self.save_table(result.metrics, 'evaluation_results.xlsx')
        self.save_table(result.metrics, 'all_metrics_data.xlsx')
        self.save_table(result.summary, 'final_metrics.xlsx')

        rows: pd.DataFrame = result.metrics.reset_index()
        class_columns: list[str] = [column for column in rows if column.startswith('Class: ')]
        selection: list[str] = ['Subject', *class_columns]
        for (stride, modality, metric), group in rows.groupby(['Stride', 'Modality', 'Metric'], sort=False):
            per_class: pd.DataFrame = group.loc[:, selection].copy()
            per_class.index = pd.Index([metric] * len(per_class), name='Metric')
            suffix: str = f'_stride_{stride}_modality_{modality}'
            self.plot_subjects_metric_per_class(per_class, metric, suffix)

        output_file: Path = self.output_dir / 'metrics_per_stride_and_modality.png'
        self.plot_summary(result.summary, output_file, show=show)

    def plot_subjects_metric_per_class(self, subjects_df: pd.DataFrame, metric: str, file_name: str) -> None:
        """Draw per-class scores from a copy and optionally export their table."""
        data: pd.DataFrame = subjects_df.melt(id_vars='Subject', var_name='Class', value_name=metric)
        figure, axes = plt.subplots(1, 2, figsize=(15, 7))
        try:
            sns.pointplot(y=metric, x='Subject', data=data, ax=axes[0], palette='deep', hue='Class')
            sns.boxplot(y=metric, hue='Class', data=data, ax=axes[1], palette='deep', legend=False)
            for axis in axes:
                axis.tick_params(axis='x', rotation=90)
                axis.set_ylabel(metric)
            figure.suptitle(metric + '_' + file_name)
            figure.tight_layout()
            if self.output_dir is not None:
                folder: Path = self.output_dir / 'metrics per class'
                folder.mkdir(parents=True, exist_ok=True)
                stem: str = metric + '_' + file_name
                figure.savefig(folder / f'{stem}.png')
                subjects_df.to_excel(folder / f'{stem}.xlsx')
        finally:
            plt.close(figure)

    def plot_summary(self, summary: pd.DataFrame, output_file: str | Path, *, show: bool = True) -> None:
        """Plot an already-computed summary, keeping all supplied values intact."""
        groups = list(summary.groupby('Metric', sort=False))
        if not groups:
            raise ValueError('Cannot plot an empty evaluation summary')
        rows: int = (len(groups) + 2) // 3
        figure, axes = plt.subplots(rows, 3, figsize=(15, 5 * rows), squeeze=False)
        try:
            for (metric, table), axis in zip(groups, axes.flat):
                self.plot_results(table, metric, axis)
            for axis in list(axes.flat)[len(groups):]:
                axis.set_visible(False)
            figure.tight_layout()
            figure.savefig(output_file)
            if show:
                plt.show()
        finally:
            plt.close(figure)

    def plot_results(self, data: pd.DataFrame, metric: str, ax: Axes) -> None:
        """Draw computed Mean scores across strides and collection groups."""
        data = data.copy()
        if 'Subject' not in data.columns and data.index.name == 'Subject':
            data = data.reset_index()
        if 'Mean' not in data.columns:
            raise ValueError('Plotting requires a computed Mean column')
        sns.boxplot(x='Stride', y='Mean', hue='Modality', data=data, ax=ax, palette='deep')
        sns.stripplot(x='Stride', y='Mean', hue='Modality', data=data, dodge=True,
                      jitter=True, palette='dark:black', ax=ax, alpha=0.5)
        handles, labels = ax.get_legend_handles_labels()
        unique_count: int = len(set(labels))
        ax.legend(handles[:unique_count], labels[:unique_count], fontsize='small')
        ax.set_title(f'{metric} per Stride')
        ax.set_xlabel('Stride')
        ax.set_ylabel(metric)
