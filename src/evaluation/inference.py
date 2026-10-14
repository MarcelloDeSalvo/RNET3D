"""Sliding-window predictions and typed full-volume evaluation reports.

Inputs must already share the intended preprocessing/orientation. Model heads
must return probabilities. ROI logits from gradient-fixed training require an
explicit probability-producing inference view. Patch order and repeated edge
patches retain the historical weighting; classifier output is a majority vote.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from itertools import product
from pathlib import Path
import time
import json
import hashlib
from typing import TYPE_CHECKING, Any, Literal, TypeAlias
from urllib.parse import quote, unquote

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import seaborn as sns
import nibabel as nib
import numpy as np
from numpy.typing import NDArray
import pandas as pd
import tensorflow as tf

from .metrics import (
    MetricFunction, dice_coefficient, precision_coefficient, specificity_coefficient,
    sensitivity_coefficient, iou_coefficient, volume_similarity_coefficient,
)

from ..utils.logger import PrintLogger

if TYPE_CHECKING:
    from ..experiments.config import DataConfig, PostprocessingConfig
    from ..experiments.data import PreparedData
    from ..utils.loader import Rodent, RodentDatasets, SubjectKey
    from ..preprocessing.preprocessor import Preprocessor
    from .postprocessing import OutputLabels

Array: TypeAlias = NDArray[Any]
Shape3D: TypeAlias = tuple[int, int, int]
ImageInput: TypeAlias = nib.spatialimages.SpatialImage | Array
PostprocessCallback: TypeAlias = Callable[[Array], Array]


def _spatial_slices(start: Sequence[int], end: Sequence[int]) -> tuple[slice, ...]:
    """Build a shared spatial selection, leaving trailing channels untouched."""
    return tuple(slice(first, last) for first, last in zip(start, end))


@dataclass(frozen=True)
class PredictionResult:
    """Full-volume ROI labels and optional brain/classification outputs."""

    roi: NDArray[Any]
    brain_mask: NDArray[Any] | None = None
    classifier: bool | None = None
    classifier_positive_fraction: float | None = None
    classifier_patch_count: int | None = None

@dataclass(frozen=True)
class EvaluationResult:
    """Raw scan metrics and the per-metric summary with class means.

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


class RandomCroppingPrediction:
    """Average overlapping patches from a single-channel 3D input.

    Despite its historical name, the inference grid is deterministic. Patch probabilities are
    accumulated in float64. ROI decisions use argmax or >= threshold;
    classifier patch votes use >0.5 and ties in the final vote are false.
    """

    def __init__(self, model: Any, patch_size: Sequence[int], num_classes: int,
                 stride: int = 16, threshold: float = 0.5) -> None:
        """Validate patch dimensions and settings before allocating full volumes."""
        self.patch_size: Shape3D = tuple(patch_size)
        if len(self.patch_size) != 3 or any(not isinstance(n, (int, np.integer)) or n <= 0 for n in self.patch_size):
            raise ValueError('patch_size must contain three positive integers')
        if not isinstance(num_classes, (int, np.integer)) or num_classes < 1:
            raise ValueError('num_classes must be a positive integer')
        self.num_classes: int = num_classes
        self.model: Any = model
        self.set_stride(stride)
        self.set_threshold(threshold)

    def fix_dimension(self, depth: int, height: int, width: int) -> Shape3D:
        """Ensure the padded volume can contain one full patch."""
        dimensions: Shape3D = (depth, height, width)
        return tuple(max(size, patch) for size, patch in zip(dimensions, self.patch_size))

    def padding(self, img_data: Array, fixed_depth: int, fixed_height: int, fixed_width: int) -> Array:
        """Zero-pad at the upper edge of each axis in a single operation."""
        dimensions: Shape3D = (fixed_depth, fixed_height, fixed_width)
        padding = [(0, max(0, size - original)) for size, original in zip(dimensions, img_data.shape)]
        return np.pad(img_data, padding, mode='constant')

    def preprocess_input(self, input_img: ImageInput) -> tuple[Array, Shape3D, Shape3D]:
        """Load NIfTI voxel data or accept a 3D array, then pad to patch size."""
        if isinstance(input_img, nib.spatialimages.SpatialImage):
            volume: Array = input_img.get_fdata()
        else:
            volume = np.asarray(input_img)
        if volume.ndim != 3 or any(size == 0 for size in volume.shape):
            raise ValueError('Expected a nonempty single-channel 3D input volume')
        original_shape: Shape3D = volume.shape
        padded_shape: Shape3D = self.fix_dimension(*original_shape)
        volume = self.padding(volume, *padded_shape)
        return volume, original_shape, padded_shape

    def initialize_prediction_arrays(self, padded_shape: Shape3D) -> tuple[Array, Array]:
        """Allocate float64 ROI sums and per-class counts as in the original API."""
        shape = (*padded_shape, self.num_classes)
        return np.zeros(shape), np.zeros(shape)

    def initialize_brain_arrays(self, padded_shape: Shape3D) -> tuple[Array, Array]:
        """Allocate single-channel float64 sums and counts for the brain head."""
        shape = (*padded_shape, 1)
        return np.zeros(shape), np.zeros(shape)

    def initialize_classifier_arrays(self) -> tuple[Array, Array]:
        """Allocate the positive-vote and total-patch counters."""
        return np.zeros(1), np.zeros(1)

    def update_prediction_arrays(self, predicted_patches: Array, count_map: Array,
                                 predicted_patch: Array, start: Shape3D, end: Shape3D) -> None:
        """Accumulate all channels together instead of looping over classes."""
        selection = _spatial_slices(start, end)
        predicted_patches[selection] += predicted_patch
        count_map[selection] += 1

    def update_brain_arrays(self, predicted_brain_patches: Array, brain_count_map: Array,
                            brain_patch: Array, start: Shape3D, end: Shape3D) -> None:
        """Use the same accumulation rule for the single-channel brain head."""
        self.update_prediction_arrays(predicted_brain_patches, brain_count_map, brain_patch, start, end)

    def update_classifier_arrays(self, predicted_classifier_patches: Array,
                                 count_map: Array, predicted_patch: Array) -> None:
        """Count a positive patch vote at >0.5, preserving the original policy."""
        probability: float = float(np.asarray(predicted_patch).item())
        if probability > 0.5:
            predicted_classifier_patches += 1
        count_map += 1

    def calculate_steps(self, padded_shape: Shape3D) -> Shape3D:
        """Retain the legacy step counts, including duplicate clamped edge patches."""
        steps: list[int] = []
        for size, patch in zip(padded_shape, self.patch_size):
            count: int = max(1, (size - patch) // self.stride + 1)
            if count * self.stride < size:
                count += 1
            steps.append(count)
        return tuple(steps)

    def calculate_patch_bounds(self, d: int, h: int, w: int,
                               padded_shape: Shape3D) -> tuple[Shape3D, Shape3D]:
        """Clamp the last patch to each upper image boundary."""
        positions: Shape3D = (d, h, w)
        start = tuple(min(position, size - patch) for position, size, patch in zip(positions, padded_shape, self.patch_size))
        end = tuple(position + patch for position, patch in zip(start, self.patch_size))
        return start, end

    def extract_patch(self, input_volume: Array, start: Shape3D, end: Shape3D) -> Array:
        """Select one spatial patch without copying its voxel data."""
        selection = _spatial_slices(start, end)
        return input_volume[selection]

    def perform_inference(self, patch: Array, with_brain_mask: bool,
                          with_classifier_output: bool) -> tuple[Array, Array | None, Array | None]:
        """Accept a single ROI head, positional heads or named output dictionaries."""
        fixed_patch: Array = patch[np.newaxis, ..., np.newaxis]
        predictions = self.model.predict(fixed_patch, verbose=False)
        if isinstance(predictions, Mapping):
            heads = [predictions['regions'], predictions.get('brain_mask'), predictions.get('classifier')]
        elif isinstance(predictions, (list, tuple)):
            heads = list(predictions)
        else:
            heads = [predictions]

        roi: Array = np.asarray(heads[0])[0]
        expected_roi = (*self.patch_size, self.num_classes)
        if roi.shape != expected_roi:
            raise ValueError(f'ROI output shape {roi.shape} does not match {expected_roi}')
        brain: Array | None = None
        classifier: Array | None = None
        if with_brain_mask:
            if len(heads) < 2 or heads[1] is None:
                raise ValueError('Requested brain_mask output is missing')
            brain = np.asarray(heads[1])[0]
            if brain.shape != (*self.patch_size, 1):
                raise ValueError('Brain output must have the patch shape and one channel')
        if with_classifier_output:
            if len(heads) < 3 or heads[2] is None:
                raise ValueError('Requested classifier output is missing')
            classifier = np.asarray(heads[2])[0]
            if classifier.size != 1:
                raise ValueError('Classifier must return one probability per patch')
            if not np.isfinite(classifier).all() or np.any(classifier < 0) or np.any(classifier > 1):
                raise ValueError('Classifier probabilities must be finite and lie in [0, 1]')
        return roi, brain, classifier

    def _final_mask(self, sums: Array, counts: Array, original_shape: Shape3D, binary: bool) -> Array:
        """Normalize in place and crop a thresholded/argmax mask to the input shape."""
        counts[counts == 0] = 1
        sums /= counts
        if binary:
            mask: Array = (sums[..., 0] >= self.threshold).astype(np.float64)
        else:
            mask = np.argmax(sums, axis=-1).astype(np.float64)
        origin: Shape3D = (0, 0, 0)
        selection = _spatial_slices(origin, original_shape)
        return mask[selection]

    def calculate_final_prediction(self, predicted_patches: Array, count_map: Array,
                                   original_shape: Shape3D) -> Array:
        """Return ROI labels with the original threshold/argmax policy."""
        binary: bool = self.num_classes == 1
        return self._final_mask(predicted_patches, count_map, original_shape, binary)

    def calculate_final_brain_mask(self, predicted_brain_patches: Array, brain_count_map: Array,
                                   original_shape: Shape3D) -> Array:
        """Return a binary brain mask from averaged probabilities."""
        return self._final_mask(predicted_brain_patches, brain_count_map, original_shape, binary=True)

    def calculate_final_classifier_output(self, predicted_classifier_patches: Array, count_map: Array) -> bool:
        """Return the majority patch vote; ties remain negative."""
        if count_map.item() <= 0:
            raise ValueError('Cannot classify a volume without patch votes')
        positive_fraction: float = float((predicted_classifier_patches / count_map).item())
        return positive_fraction > 0.5

    def set_threshold(self, threshold: float) -> None:
        """Set a finite probability threshold in [0, 1]."""
        if not np.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError('threshold must be between zero and one')
        self.threshold: float = threshold

    def set_stride(self, stride: int) -> None:
        """Reject invalid strides and strides that would leave uncovered voxels."""
        if isinstance(stride, bool) or not isinstance(stride, (int, np.integer)) or not 1 <= stride <= min(self.patch_size):
            raise ValueError('stride must be a positive integer no larger than the smallest patch dimension')
        self.stride: int = stride

    def predict(self, input_img: ImageInput, with_brain_mask: bool = False,
                with_classifier: bool = False) -> PredictionResult:
        """Accumulate patches in the original order and return selected output masks."""
        volume, original_shape, padded_shape = self.preprocess_input(input_img)
        sums, counts = self.initialize_prediction_arrays(padded_shape)
        if with_brain_mask:
            brain_sums, brain_counts = self.initialize_brain_arrays(padded_shape)
        if with_classifier:
            positive_votes, total_votes = self.initialize_classifier_arrays()

        steps: Shape3D = self.calculate_steps(padded_shape)
        positions = [range(0, count * self.stride, self.stride) for count in steps]
        for d, h, w in product(*positions):
            start, end = self.calculate_patch_bounds(d, h, w, padded_shape)
            patch = self.extract_patch(volume, start, end)
            roi, brain, classifier = self.perform_inference(patch, with_brain_mask, with_classifier)
            self.update_prediction_arrays(sums, counts, roi, start, end)
            if with_brain_mask:
                self.update_brain_arrays(brain_sums, brain_counts, brain, start, end)
            if with_classifier:
                self.update_classifier_arrays(positive_votes, total_votes, classifier)

        brain_mask: Array | None = None
        classifier_output: bool | None = None
        positive_fraction: float | None = None
        patch_count: int | None = None
        if with_brain_mask:
            brain_mask = self.calculate_final_brain_mask(brain_sums, brain_counts, original_shape)
        if with_classifier:
            classifier_output = self.calculate_final_classifier_output(positive_votes, total_votes)
            positive_fraction = float((positive_votes / total_votes).item())
            patch_count = int(total_votes.item())
        roi_mask: Array = self.calculate_final_prediction(sums, counts, original_shape)
        return PredictionResult(
            roi=roi_mask, brain_mask=brain_mask, classifier=classifier_output,
            classifier_positive_fraction=positive_fraction, classifier_patch_count=patch_count,
        )


class ExperimentVolumeEvaluator:
    """Bridge configured experiments to physical metrics and prediction artifacts.

    Postprocessing and export IDs are explicit inference settings. Metrics use
    those same IDs on the complete orthogonal grid; exports restore MRI geometry.
    GT comes from source annotations, preserving sides merged during training.
    """

    def __init__(
        self, model: Any, data: PreparedData, subject_ids: Sequence[str],
        options: Mapping[str, Any],
    ) -> None:
        """Store a probability-producing model and an explicit evaluation cohort."""
        self.model: Any = model
        self.data: PreparedData = data
        self.subject_ids: list[str] = list(subject_ids)
        self.options: dict[str, Any] = dict(options)
        self.classifier_predictions: pd.DataFrame = pd.DataFrame()

    def evaluate(self) -> pd.DataFrame:
        """Predict each scan once, preserving missing-annotation statuses and timings."""
        from ..experiments.data import make_preprocessor
        from .metrics import segmentation_metrics_table
        from .postprocessing import FullVolumePostprocessor

        config = self.data.config
        postprocessor = FullVolumePostprocessor(
            config.labels, config.inference.postprocessing,
            original_label_ids=config.inference.original_label_ids,
        )
        output_labels = postprocessor.labels
        pipeline = make_preprocessor(
            config, stage="inference", label_mapping=output_labels.ground_truth_mapping,
        )
        predictor = RandomCroppingPrediction(
            model=self.model, patch_size=config.preprocessing.patch_size,
            num_classes=len(config.labels.names), stride=config.inference.stride,
        )
        output_dir = Path(self.options["evaluation_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        regions = {label: name for label, name in output_labels.names.items() if label != 0}
        identity_mapping = {label: label for label in output_labels.names}
        label_metadata = {
            "original_label_ids": config.inference.original_label_ids,
            "postprocessing": asdict(config.inference.postprocessing),
            "names": output_labels.names,
            "ground_truth_mapping": output_labels.ground_truth_mapping,
            "prediction_mapping": output_labels.prediction_mapping,
        }
        label_file = output_dir / "prediction_labels.json"
        label_file.write_text(json.dumps(label_metadata, indent=2), encoding="utf-8")
        save_predictions: bool = bool(self.options.get("save_predictions", False))
        tables: list[pd.DataFrame] = []
        timings: list[dict[str, str | float]] = []
        classifier_rows: list[dict[str, Any]] = []

        # Exclude tracing/device warm-up from per-volume prediction durations.
        warmup_shape = (1, *config.preprocessing.patch_size, 1)
        warmup = np.zeros(warmup_shape, dtype=np.float32)
        self.model.predict(warmup, verbose=0)

        for subject_id in self.subject_ids:
            for key in self.data.subjects.scan_keys([subject_id]):
                scan = self.data.loader.get_subject(key)
                started = time.perf_counter()
                reference_masks = [scan.labels]
                if scan.brain_mask is not None:
                    reference_masks.append(scan.brain_mask)
                image, references = pipeline.preprocess(scan.image, reference_masks)
                reference = references[0]
                preprocessing_seconds = time.perf_counter() - started

                if image.shape != reference.shape or not np.allclose(image.affine, reference.affine):
                    raise ValueError(f"{scan.id}: preprocessed image and reference must share a physical grid")

                spacing = nib.affines.voxel_sizes(image.affine)
                directions = image.affine[:3, :3] / spacing
                orthogonal = np.allclose(directions.T @ directions, np.eye(3), atol=1e-5)
                if not orthogonal:
                    raise ValueError(f"{scan.id}: physical distance metrics require an orthogonal grid")

                started = time.perf_counter()
                if config.model.classifier:
                    prediction = predictor.predict(image, with_brain_mask=True, with_classifier=True)
                else:
                    prediction = predictor.predict(image, with_brain_mask=True)
                inference_seconds = time.perf_counter() - started
                condition = "TBI"
                if scan.get_sham():
                    condition = "SHAM"
                if config.model.classifier:
                    if prediction.classifier is None:
                        raise ValueError(f"{scan.id}: configured classifier returned no decision")
                    classifier_reference = int(not scan.get_sham())
                    classifier_prediction = int(prediction.classifier)
                    classifier_row = {
                        "Subject": subject_id, "Scan": scan.id, "Dataset": scan.subject_group[0],
                        "Reference": classifier_reference, "Prediction": classifier_prediction,
                        "PositivePatchFraction": prediction.classifier_positive_fraction,
                        "PatchCount": prediction.classifier_patch_count,
                        "Condition": condition, "PatchThreshold": 0.5,
                        "ScanDecisionRule": "majority_patch_vote_ties_SHAM",
                    }
                    classifier_rows.append(classifier_row)
                started = time.perf_counter()
                try:
                    processed = postprocessor.process(prediction.roi, prediction.brain_mask)
                except ValueError as error:
                    raise ValueError(f"{scan.id}: full-volume postprocessing failed: {error}") from error
                postprocessing_seconds = time.perf_counter() - started
                unavailable = set(scan.get_unavailable_labels() or ())
                unavailable_classes = {
                    label for source, label in output_labels.ground_truth_mapping.items() if source in unavailable
                }
                available_classes = sorted(set(identity_mapping) - unavailable_classes)
                reference_data = reference.get_fdata()
                table = segmentation_metrics_table(
                    reference=reference_data, prediction=processed.regions, spacing_mm=spacing,
                    regions=regions, subject=subject_id, scan=scan.id,
                    dataset=scan.subject_group[0], available_labels=available_classes,
                )
                table = table.assign(Head="regions", EvaluationStage="full_volume", Aggregation="scan", Condition=condition)
                tables.append(table)

                brain_reference = np.zeros(image.shape, dtype=np.uint8)
                brain_available: list[int] = []
                if scan.brain_mask is not None:
                    reference_brain = references[1]
                    if reference_brain.shape != image.shape or not np.allclose(reference_brain.affine, image.affine):
                        raise ValueError(f"{scan.id}: brain reference must share the inference grid")
                    brain_reference = reference_brain.get_fdata() != 0
                    brain_available = [1]
                brain_table = segmentation_metrics_table(
                    reference=brain_reference, prediction=processed.brain_mask, spacing_mm=spacing,
                    regions={1: "Brain"}, subject=subject_id, scan=scan.id,
                    dataset=scan.subject_group[0], available_labels=brain_available,
                )
                brain_table = brain_table.assign(Head="brain_mask", EvaluationStage="full_volume", Aggregation="scan", Condition=condition)
                tables.append(brain_table)

                if save_predictions:
                    group, scan_id = key
                    group_name = quote(group, safe="")
                    scan_name = quote(scan_id, safe="")
                    folder = output_dir / "predictions" / group_name / scan_name
                    folder.mkdir(parents=True, exist_ok=True)
                    masks = {"regions": processed.regions, "brain_mask": processed.brain_mask}
                    for name, mask in masks.items():
                        if mask is None:
                            continue
                        mask_data = np.asarray(mask, dtype=np.int16)
                        grid_prediction = nib.Nifti1Image(mask_data, image.affine)
                        restoration_mapping = identity_mapping
                        if name == "brain_mask":
                            restoration_mapping = {0: 0, 1: 1}
                        restored = pipeline.deprocess(
                            prediction=grid_prediction, original_image=scan.image,
                            mapping=restoration_mapping,
                        )
                        # Store integer data, rather than asking NIfTI to scale
                        # floating-point labels into the int16 storage range.
                        restored_data = restored.get_fdata().astype(np.int16)
                        saved_mask = nib.Nifti1Image(restored_data, restored.affine, restored.header)
                        saved_mask.set_data_dtype(np.int16)
                        saved_mask.header.set_slope_inter(1, 0)
                        nib.save(saved_mask, folder / f"{name}.nii.gz")

                timings.append({
                    "Subject": subject_id, "Scan": scan.id,
                    "PreprocessingSeconds": preprocessing_seconds, "InferenceSeconds": inference_seconds,
                    "PostprocessingSeconds": postprocessing_seconds,
                })

        if not tables:
            raise ValueError("No evaluation scans selected")

        pd.DataFrame(timings).to_csv(output_dir / "inference_timings.csv", index=False)
        self.classifier_predictions = pd.DataFrame(classifier_rows)
        if classifier_rows:
            self.classifier_predictions.to_csv(output_dir / "classifier_predictions.csv", index=False)
        return pd.concat(tables, ignore_index=True)


@dataclass(frozen=True)
class SavedModelEvaluationConfig:
    """Explicit saved-model cohort and fresh output folder for offline evaluation.

    data/reference optionally replace inputs or the orientation reference.
    Model channels and source-label semantics still come from experiment_file.
    Every selected animal includes all its registered longitudinal scans.
    original_label_ids/postprocessing override evaluation settings only; None
    inherits the saved run. Notebook cells provide explicit original-ID settings.
    """

    experiment_file: Path
    model_file: Path
    output_dir: Path
    subject_ids: tuple[str, ...]
    role: str = "external"
    variant: str = "saved_model"
    seed: int = 42
    fold: int = 0
    save_predictions: bool = True
    data: DataConfig | None = None
    reference: Path | None = None
    stride: int | None = None
    checkpoint_kind: Literal["model", "weights"] = "model"
    original_label_ids: bool | None = None
    postprocessing: PostprocessingConfig | None = None
    evaluate_patches: bool = False


@dataclass(frozen=True)
class SavedModelEvaluationResult:
    """Offline full-volume rows, timings and the selected scan manifest."""

    output_dir: Path
    metrics: pd.DataFrame
    timings: pd.DataFrame
    manifest: pd.DataFrame
    data: PreparedData
    patch_metrics: pd.DataFrame = field(default_factory=pd.DataFrame)
    classifier_predictions: pd.DataFrame = field(default_factory=pd.DataFrame)


def best_checkpoint_settings(
    run_dir: Path, *, fold: int = 0, role: str = "test", output_dir: Path | None = None,
    reference: Path | None = None,
    original_label_ids: bool | None = None, postprocessing: PostprocessingConfig | None = None,
    evaluate_patches: bool = False,
) -> SavedModelEvaluationConfig:
    """Select the saved cohort and best committed weights of a stopped/completed run.

    Fixed-split test animals come from splits.json; outer holdouts and rat
    validation come from fold_N/split.json. CV test/external cohorts come from
    run metadata. Evaluation defaults to a fresh sibling folder, never writing
    into the training run. No finished-fit flag or optimizer state is required.
    An explicit reference overrides the saved input orientation for evaluation.
    Label/postprocessing overrides do not change fitting weights or recovery.
    """
    from uuid import uuid4
    from ..experiments.config import ExperimentConfig

    if isinstance(fold, bool) or not isinstance(fold, int) or fold < 0:
        raise ValueError("Choose a nonnegative fold index")
    if role not in ("test", "holdout", "validation", "external"):
        raise ValueError("Select test, holdout, validation or external evaluation")

    run_dir = Path(run_dir).resolve()
    experiment_file = run_dir / "experiment.json"
    config = ExperimentConfig.load(experiment_file)
    fold_dir = run_dir / f"fold_{fold}"
    split = json.loads((run_dir / "splits.json").read_text(encoding="utf-8"))
    metadata: dict[str, Any] = {}
    metadata_file = run_dir / "metadata.json"
    if metadata_file.exists():
        metadata = json.loads(metadata_file.read_text(encoding="utf-8"))

    if isinstance(split, dict):
        if fold != 0 or role not in ("test", "validation"):
            raise ValueError("Fixed-split runs have fold 0 and test/validation cohorts")
        subject_ids = tuple(split[role])
    elif role in ("holdout", "validation"):
        fold_split = json.loads((fold_dir / "split.json").read_text(encoding="utf-8"))
        split_role = "validation"
        if role == "holdout":
            split_role = "test"
        subject_ids = tuple(fold_split.get(split_role, ()))
    else:
        cohort_key = f"{role}_ids"
        subject_ids = tuple(metadata.get("evaluation", {}).get(cohort_key, ()))
    if not subject_ids:
        raise ValueError(f"No saved animals for role {role!r} in fold {fold}")

    model_file = fold_dir / "best.weights.h5"
    state_file = fold_dir / "training_state.json"
    if state_file.exists():
        state = json.loads(state_file.read_text(encoding="utf-8"))
        model_file = fold_dir / state["selected_checkpoint"]
    if not model_file.is_file():
        raise FileNotFoundError("No saved best checkpoint; complete an epoch before evaluating")

    if output_dir is None:
        evaluation_id = uuid4().hex[:12]
        output_dir = run_dir.parent / "checkpoint_evaluations" / run_dir.name / f"fold_{fold}_{evaluation_id}"
    output_dir = Path(output_dir).resolve()
    if output_dir == run_dir or run_dir in output_dir.parents:
        raise ValueError("Checkpoint evaluation must use a folder outside the training run")
    variant = config.training.model_name
    return SavedModelEvaluationConfig(
        experiment_file=experiment_file, model_file=model_file, output_dir=output_dir,
        subject_ids=subject_ids, role=role, variant=variant, seed=config.training.seed,
        fold=fold, checkpoint_kind="weights", save_predictions=True, reference=reference,
        original_label_ids=original_label_ids, postprocessing=postprocessing,
        evaluate_patches=evaluate_patches,
    )


def evaluate_saved_model(settings: SavedModelEvaluationConfig) -> SavedModelEvaluationResult:
    """Load a configured export or fitting weights for full-volume evaluation.

    No fit or optimizer initialization occurs. Existing output folders are
    rejected. Label semantics must match the supplied saved configuration;
    matching tensor shapes alone cannot establish that semantic contract.
    """
    from ..experiments.config import ExperimentConfig, InitializationConfig
    from ..experiments.data import prepare_data
    from ..experiments.training import build_network, compile_model, configure_runtime, inference_model, training_view

    if settings.role not in ("test", "holdout", "validation", "external"):
        raise ValueError("Use an explicit test, holdout, validation or external role")
    if settings.checkpoint_kind not in ("model", "weights"):
        raise ValueError("Checkpoint kind must be 'model' or 'weights'")
    if not settings.variant.strip():
        raise ValueError("Provide a model variant name")
    if not settings.subject_ids:
        raise ValueError("Select explicit animal IDs before evaluation")
    for value in (settings.seed, settings.fold):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("Evaluation seed and fold must be nonnegative integers")
    if not Path(settings.model_file).is_file():
        raise FileNotFoundError(settings.model_file)
    folder = Path(settings.output_dir).resolve()
    source_run = Path(settings.experiment_file).resolve().parent
    is_training_run = (source_run / "splits.json").is_file() and (source_run / f"fold_{settings.fold}").is_dir()
    if is_training_run and (folder == source_run or source_run in folder.parents):
        raise ValueError("Evaluation outputs must be outside the source experiment folder")
    if folder.exists():
        raise FileExistsError(folder)

    saved = ExperimentConfig.load(settings.experiment_file)
    config = replace(saved, initialization=InitializationConfig(), initialization_by_fold={})
    if settings.data is not None:
        config = replace(config, data=settings.data)
    preprocessing = config.preprocessing
    if settings.reference is not None:
        preprocessing = replace(preprocessing, reference=settings.reference)
    inference = replace(config.inference, enabled=True, save_predictions=settings.save_predictions)
    if settings.original_label_ids is not None:
        inference = replace(inference, original_label_ids=settings.original_label_ids)
    if settings.postprocessing is not None:
        inference = replace(inference, postprocessing=settings.postprocessing)
    if settings.stride is not None:
        inference = replace(inference, stride=settings.stride)
    config = replace(config, preprocessing=preprocessing, inference=inference)
    config.validate()
    data = prepare_data(config, require_training_loss=False)
    manifest = data.subjects.manifest(settings.subject_ids)

    mixed_precision = False
    metadata_file = source_run / "metadata.json"
    if settings.checkpoint_kind == "weights" and metadata_file.exists():
        source_metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
        mixed_precision = source_metadata.get("precision") == "mixed_float16"
    runtime = configure_runtime(seed=settings.seed, mixed_precision=mixed_precision)
    if settings.checkpoint_kind == "weights":
        model = build_network(config)
        model = training_view(model, config)
        model.load_weights(str(settings.model_file))
    else:
        model = tf.keras.models.load_model(str(settings.model_file), compile=False)
    model = inference_model(model)
    expected_input = (None, *config.preprocessing.patch_size, 1)
    if tuple(model.input_shape) != expected_input:
        raise ValueError("Model input shape differs from the saved preprocessing configuration")
    expected_heads = ["regions", "brain_mask"]
    if config.model.classifier:
        expected_heads.append("classifier")
    if model.output_names != expected_heads:
        raise ValueError("Model output heads differ from the saved configuration")
    if model.output_shape[0][-1] != len(config.labels.names) or model.output_shape[1][-1] != 1:
        raise ValueError("Model output channels differ from the saved label contract")
    activation = getattr(model.get_layer("regions"), "activation", None)
    if getattr(activation, "__name__", None) != "softmax":
        raise ValueError("ROI inference output must use softmax probabilities")

    folder.mkdir(parents=True, exist_ok=False)
    digest = hashlib.sha256()
    with Path(settings.model_file).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    model_hash = digest.hexdigest()
    provenance = {
        "model_file": str(Path(settings.model_file).resolve()), "model_sha256": model_hash,
        "checkpoint_kind": settings.checkpoint_kind,
        "source_experiment": saved.to_dict(), "evaluation_experiment": config.to_dict(),
        "runtime": runtime, "role": settings.role, "variant": settings.variant,
        "seed": settings.seed, "fold": settings.fold, "subjects": list(settings.subject_ids),
        "data_contract": data.contract(),
    }
    (folder / "evaluation.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    manifest.to_csv(folder / "subjects.csv", index=False)
    patch_metrics = pd.DataFrame()
    if settings.evaluate_patches:
        from ..experiments.evaluation import evaluate_patches
        from ..experiments.patches import EvaluationPatches

        sampling_source = "new evaluation sampling"
        saved_patches = source_run / "evaluation_patches.json"
        if saved_patches.is_file():
            payload = json.loads(saved_patches.read_text(encoding="utf-8"))
            digest_payload = {key: value for key, value in payload.items() if key != "sha256"}
            patch_digest = hashlib.sha256(json.dumps(digest_payload, sort_keys=True).encode()).hexdigest()
            if patch_digest != payload["sha256"]:
                raise ValueError("Saved evaluation patches were modified")
            if payload["contract"] == EvaluationPatches.expected_contract(data):
                data.evaluation_patches = EvaluationPatches.load(saved_patches, data)
                sampling_source = "saved training-run patches"
        data.patch_collection().save(folder / "evaluation_patches.json")
        native_model = compile_model(model, config)
        patch_metrics = evaluate_patches(native_model, data, settings.subject_ids)
        patch_metrics = patch_metrics.assign(
            DatasetRole=settings.role, Variant=settings.variant, Seed=settings.seed, Fold=settings.fold,
        )
        provenance["patch_sampling_source"] = sampling_source
        (folder / "evaluation.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    options = {"evaluation_dir": folder, "save_predictions": settings.save_predictions}
    evaluator = ExperimentVolumeEvaluator(
        model=model, data=data, subject_ids=settings.subject_ids, options=options,
    )
    metrics = evaluator.evaluate()
    metrics = metrics.assign(
        DatasetRole=settings.role, Variant=settings.variant, Seed=settings.seed, Fold=settings.fold,
    )
    classifier_predictions = evaluator.classifier_predictions
    if not classifier_predictions.empty:
        classifier_predictions = classifier_predictions.assign(
            DatasetRole=settings.role, Variant=settings.variant, Seed=settings.seed, Fold=settings.fold,
        )
    from ..experiments.evaluation import export_evaluation_tables

    export_evaluation_tables(patch_metrics, metrics, folder, classifier_predictions=classifier_predictions)
    timings = pd.read_csv(folder / "inference_timings.csv")
    return SavedModelEvaluationResult(output_dir=folder, metrics=metrics, timings=timings,
                                      manifest=manifest, data=data, patch_metrics=patch_metrics,
                                      classifier_predictions=classifier_predictions)


def load_saved_evaluation(
    output_dir: Path, *, reference: Path | None = None,
) -> SavedModelEvaluationResult:
    """Reopen results with an optional viewing orientation, without recomputation.

    A reference replacement affects the viewer's orientation only; saved metrics
    and provenance remain the original evaluation's values.
    """
    from ..experiments.config import ExperimentConfig
    from ..experiments.data import prepare_data

    folder = Path(output_dir)
    metadata = json.loads((folder / "evaluation.json").read_text(encoding="utf-8"))
    config = ExperimentConfig.from_dict(metadata["evaluation_experiment"])
    if reference is not None:
        preprocessing = replace(config.preprocessing, reference=Path(reference))
        config = replace(config, preprocessing=preprocessing)
    data = prepare_data(config, require_training_loss=False)
    metrics = pd.read_csv(folder / "full_volume_metrics.csv")
    timings = pd.read_csv(folder / "inference_timings.csv")
    manifest = pd.read_csv(folder / "subjects.csv")
    patch_metrics = pd.DataFrame()
    patch_file = folder / "patch_metrics.csv"
    if patch_file.is_file() and patch_file.read_text(encoding="utf-8").strip():
        patch_metrics = pd.read_csv(patch_file)
        for column in ("Subject", "Scan"):
            if column in patch_metrics:
                patch_metrics[column] = patch_metrics[column].fillna("")
    classifier_predictions = pd.DataFrame()
    classifier_file = folder / "classifier_predictions.csv"
    if classifier_file.is_file():
        classifier_predictions = pd.read_csv(classifier_file)
    return SavedModelEvaluationResult(
        output_dir=folder, metrics=metrics, timings=timings, manifest=manifest, data=data,
        patch_metrics=patch_metrics,
        classifier_predictions=classifier_predictions,
    )


def prediction_slice_figure(
    image_path: Path, prediction_path: Path, *, axis: int = 2, index: int | None = None,
) -> Any:
    """Inspect a saved ROI mask on its original image grid with native Plotly."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    image = nib.load(str(image_path))
    prediction = nib.load(str(prediction_path))
    if image.shape != prediction.shape or not np.allclose(image.affine, prediction.affine):
        raise ValueError("Image and saved prediction must share the original physical grid")
    if axis not in (0, 1, 2):
        raise ValueError("Slice axis must be zero, one or two")
    if index is None:
        index = image.shape[axis] // 2
    if not 0 <= index < image.shape[axis]:
        raise ValueError("Slice index is outside the image")

    image_slice = np.take(image.get_fdata(), index, axis=axis).T
    prediction_slice = np.take(prediction.get_fdata(), index, axis=axis).T
    foreground = np.where(prediction_slice == 0, np.nan, prediction_slice)
    figure = make_subplots(rows=1, cols=2, subplot_titles=("Original image", "Saved ROI labels"))
    for column in (1, 2):
        background = go.Heatmap(z=image_slice, colorscale="Gray", showscale=False)
        figure.add_trace(background, row=1, col=column)
    overlay = go.Heatmap(z=foreground, colorscale="Turbo", opacity=0.6, colorbar={"title": "Class ID"})
    figure.add_trace(overlay, row=1, col=2)
    figure.update_yaxes(scaleanchor="x", row=1, col=1)
    figure.update_yaxes(scaleanchor="x2", row=1, col=2)
    figure.update_layout(title=f"Voxel axis {axis}, slice {index}", template="plotly_white")
    return figure


def saved_prediction_manifest(
    data: PreparedData, run_dir: Path, *, role: str = "test",
) -> pd.DataFrame:
    """List saved ROI masks for one run/role, retaining animal, scan and fold.

    Missing prediction exports produce an empty table. No inference is run.
    Encoded scan directories are resolved through the existing loader, including
    legacy scans and normalized longitudinal timepoints.
    """
    if role not in ("test", "holdout", "validation", "external"):
        raise ValueError("Select test, holdout, validation or external predictions")

    run_dir = Path(run_dir)
    pattern = f"fold_*/evaluations/{role}/regions/predictions/*/*/regions.nii.gz"
    columns = ["Subject", "Scan", "Group", "Fold", "DatasetRole", "PredictionPath"]
    rows: list[dict[str, Any]] = []
    paths = [(path, int(path.relative_to(run_dir).parts[0].removeprefix("fold_")))
             for path in sorted(run_dir.glob(pattern))]
    evaluation_file = run_dir / "evaluation.json"
    if evaluation_file.exists():
        metadata = json.loads(evaluation_file.read_text(encoding="utf-8"))
        if metadata["role"] == role:
            paths.extend((path, metadata["fold"]) for path in sorted(run_dir.glob("predictions/*/*/regions.nii.gz")))
    for path, fold in paths:
        group = unquote(path.parent.parent.name)
        scan_id = unquote(path.parent.name)
        key = (group, scan_id)
        data.loader.get_subject(key)
        subject = data.subjects.subject_for_scan(key)
        rows.append({
            "Subject": subject, "Scan": scan_id, "Group": group, "Fold": fold,
            "DatasetRole": role, "PredictionPath": str(path),
        })

    table = pd.DataFrame(rows, columns=columns)
    table = table.sort_values(["Fold", "Subject", "Scan"]).reset_index(drop=True)
    table.index.name = "ViewIndex"
    return table


@dataclass(frozen=True)
class PredictionComparison:
    """Aligned arrays in reference orientation, using output label IDs.

    Unavailable GT classes are -1 for display; predictions retain every class.
    These arrays support visual inspection only and do not replace evaluation.
    """

    image: Array
    ground_truth: Array
    prediction: Array
    spacing: tuple[float, float, float]
    orientation: tuple[str, str, str]
    unavailable_names: tuple[str, ...]
    label_names: dict[int, str]


def _saved_prediction_labels(data: PreparedData, prediction_path: Path) -> OutputLabels:
    """Read export semantics; older artifacts without metadata use model IDs."""
    from .postprocessing import OutputLabels

    label_file = Path(prediction_path).parents[3] / "prediction_labels.json"
    if label_file.is_file():
        metadata = json.loads(label_file.read_text(encoding="utf-8"))
        prediction_mapping = {int(key): value for key, value in metadata["prediction_mapping"].items()}
        ground_truth_mapping = {int(key): value for key, value in metadata["ground_truth_mapping"].items()}
        names = {int(key): value for key, value in metadata["names"].items()}
        return OutputLabels(prediction_mapping, ground_truth_mapping, names)

    return OutputLabels.resolve(data.config.labels, split_ventricles=False, original_label_ids=False)


def load_prediction_comparison(
    data: PreparedData, scan_key: SubjectKey, prediction_path: Path,
) -> PredictionComparison:
    """Load saved predictions, map GT and reorient every input to the reference.

    Export metadata supplies sparse original IDs and their source-GT mapping.
    Predictions must share the original MRI grid. A differently sampled GT is
    resampled onto that grid with nearest-neighbour interpolation for labels.
    No training preprocessing, cropping, model prediction or file write occurs.
    """
    from nibabel.processing import resample_from_to
    from ..preprocessing.preprocessor import MapLabels, Reorient

    scan = data.loader.get_subject(scan_key)
    image = scan.image
    prediction = nib.load(str(prediction_path))
    if image.ndim != 3 or image.shape != prediction.shape:
        raise ValueError("Image and saved prediction must share a 3D physical grid")
    if not np.allclose(image.affine, prediction.affine):
        raise ValueError("Image and saved prediction must share the original physical grid")

    label_config = data.config.labels
    output_labels = _saved_prediction_labels(data, prediction_path)
    mapper = MapLabels(mapping=output_labels.ground_truth_mapping, unknown=label_config.unknown)
    ground_truth = mapper.execute(scan.labels, is_mask=True)
    if image.shape != ground_truth.shape or not np.allclose(image.affine, ground_truth.affine):
        grid = (image.shape, image.affine)
        ground_truth = resample_from_to(ground_truth, grid, order=0)

    reference = nib.load(str(data.config.preprocessing.reference))
    reorient = Reorient(ref_img=reference)
    image = reorient.execute(image)
    ground_truth = reorient.execute(ground_truth, is_mask=True)
    prediction = reorient.execute(prediction, is_mask=True)

    predicted = prediction.get_fdata()
    if not np.isfinite(predicted).all() or np.any(predicted != np.rint(predicted)):
        raise ValueError("Saved ROI predictions must contain finite integer labels")
    if not np.isin(predicted, list(output_labels.names)).all():
        raise ValueError("Saved ROI labels differ from the recorded output class IDs")

    unavailable_sources = set(scan.get_unavailable_labels() or ())
    unavailable_classes = {
        label for source, label in output_labels.ground_truth_mapping.items() if source in unavailable_sources
    }
    truth = ground_truth.get_fdata().astype(np.int16)
    truth[np.isin(truth, list(unavailable_classes))] = -1
    unavailable_names = tuple(output_labels.names[label] for label in sorted(unavailable_classes))
    spacing = tuple(nib.affines.voxel_sizes(image.affine))
    orientation = nib.aff2axcodes(image.affine)
    image_data = image.get_fdata(dtype=np.float32)
    prediction_data = predicted.astype(np.int16)
    return PredictionComparison(
        image=image_data, ground_truth=truth, prediction=prediction_data,
        spacing=spacing, orientation=orientation, unavailable_names=unavailable_names,
        label_names=output_labels.names,
    )


def _comparison_slice_sources(
    comparison: PredictionComparison, *, axis: int, index: int,
    colors: Array, intensity_range: tuple[float, float], opacity: float,
    region: int | None, max_pixels: int,
) -> list[str]:
    """Encode three PNG panels to keep the native Plotly frame payload small."""
    import base64
    from io import BytesIO
    from PIL import Image

    image = np.take(comparison.image, index, axis=axis).T
    truth = np.take(comparison.ground_truth, index, axis=axis).T
    prediction = np.take(comparison.prediction, index, axis=axis).T
    step = max(1, int(np.ceil(max(image.shape) / max_pixels)))
    image = image[::step, ::step]
    truth = truth[::step, ::step]
    prediction = prediction[::step, ::step]

    low, high = intensity_range
    intensity = np.nan_to_num(image, nan=low, posinf=high, neginf=low)
    grayscale = np.clip((intensity - low) / (high - low), 0, 1) * 255
    background = np.repeat(grayscale[..., np.newaxis], 3, axis=-1)
    panels = [background]
    for labels in (truth, prediction):
        foreground = labels > 0
        if region is not None:
            foreground = labels == region
        safe_labels = np.maximum(labels, 0)
        overlay = colors[safe_labels]
        blended = (1 - opacity) * background + opacity * overlay
        panel = np.where(foreground[..., np.newaxis], blended, background)
        panels.append(panel)

    sources: list[str] = []
    for panel in panels:
        rgb = panel.astype(np.uint8)
        buffer = BytesIO()
        Image.fromarray(rgb).save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        sources.append("data:image/png;base64," + encoded)
    return sources


def prediction_comparison_figure(
    data: PreparedData, predictions: pd.DataFrame, *,
    scan_indices: Sequence[int] | None = None, region: int | None = None,
    opacity: float = 0.55, max_slices: int | None = 64, max_pixels: int = 192,
) -> Any:
    """Compare saved MRI/GT/predictions with native scan/plane menus and a slider.

    None selects up to three manifest rows. Choose explicit row indices to view
    other animals/folds. Frames sample up to max_slices per plane; None includes
    every slice. max_pixels bounds preview resolution, leaving saved masks intact.
    The same categorical colors apply to both overlays. No widgets/HTML exports
    or inference are required; call figure.show(renderer="vscode") in notebooks.
    """
    import html
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    if predictions.empty:
        raise ValueError("No saved predictions to inspect")
    if not np.isfinite(opacity) or not 0 <= opacity <= 1:
        raise ValueError("Overlay opacity must be between zero and one")
    for name, value in (("max_slices", max_slices), ("max_pixels", max_pixels)):
        if value is None and name == "max_slices":
            continue
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")

    if scan_indices is None:
        scan_indices = tuple(range(min(3, len(predictions))))
    if not scan_indices or len(set(scan_indices)) != len(scan_indices):
        raise ValueError("Choose at least one distinct prediction row index")
    for index in scan_indices:
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(predictions):
            raise ValueError("Prediction row index is outside the manifest")

    first_path = Path(predictions.iloc[scan_indices[0]]["PredictionPath"])
    names = _saved_prediction_labels(data, first_path).names
    if region is not None and (isinstance(region, bool) or not isinstance(region, int) or region == 0 or region not in names):
        raise ValueError("Select a foreground output class ID from the saved label vocabulary, or None")

    palette = plt.get_cmap("tab20")
    colors = np.zeros((max(names) + 1, 3))
    label_colors = palette(np.arange(len(names)) % 20)[:, :3] * 255
    for index, label in enumerate(names):
        colors[label] = label_colors[index]
    figure = make_subplots(rows=1, cols=3, subplot_titles=("MRI", "Ground truth", "Prediction"))
    frames: list[Any] = []
    buttons: list[dict[str, Any]] = []
    initial_frame: Any = None
    initial_slider: dict[str, Any] = {}

    for row_index in scan_indices:
        row = predictions.iloc[row_index]
        scan_key = (row["Group"], row["Scan"])
        prediction_path = Path(row["PredictionPath"])
        comparison = load_prediction_comparison(data, scan_key, prediction_path)
        if comparison.label_names != names:
            raise ValueError("Select predictions with the same saved label vocabulary")
        finite = comparison.image[np.isfinite(comparison.image)]
        if finite.size == 0:
            raise ValueError("MRI has no finite intensities to display")
        low, high = np.percentile(finite, [1, 99])
        if high <= low:
            high = low + 1
        intensity_range = (float(low), float(high))
        missing = ", ".join(comparison.unavailable_names) or "none"
        missing = html.escape(missing)
        scan_title = html.escape(f"{row['Subject']} / {row['Scan']} — fold {row['Fold']} ({row['DatasetRole']})")

        for axis in (2, 1, 0):
            orientation = comparison.orientation
            planes = {"R": "Sagittal", "L": "Sagittal", "A": "Coronal", "P": "Coronal", "S": "Axial", "I": "Axial"}
            plane = planes[orientation[axis]]
            axis_name = f"{plane} / axis {axis} ({orientation[axis]})"
            title = f"{scan_title}<br><sup>{axis_name}; unannotated GT regions: {missing}</sup>"
            count = comparison.image.shape[axis]
            sample_count = count
            if max_slices is not None:
                sample_count = min(count, max_slices)
            indices = np.linspace(0, count - 1, sample_count, dtype=int)
            steps: list[dict[str, Any]] = []
            plane_frames: list[Any] = []
            remaining_axes = [value for value in (0, 1, 2) if value != axis]
            height_axis, width_axis = remaining_axes[1], remaining_axes[0]
            display_shape = (comparison.image.shape[height_axis], comparison.image.shape[width_axis])
            pixel_step = max(1, int(np.ceil(max(display_shape) / max_pixels)))
            dx = pixel_step * comparison.spacing[width_axis]
            dy = pixel_step * comparison.spacing[height_axis]

            for index in indices:
                sources = _comparison_slice_sources(
                    comparison, axis=axis, index=int(index), colors=colors,
                    intensity_range=intensity_range, opacity=opacity, region=region,
                    max_pixels=max_pixels,
                )
                traces = [go.Image(source=source, dx=dx, dy=dy, hoverinfo="skip") for source in sources]
                frame_name = f"{row_index}:{axis}:{index}"
                frame = go.Frame(name=frame_name, data=traces, traces=[0, 1, 2])
                frames.append(frame)
                plane_frames.append(frame)
                animation = {"mode": "immediate", "frame": {"duration": 0, "redraw": True}, "transition": {"duration": 0}}
                steps.append({"method": "animate", "label": str(index), "args": [[frame_name], animation]})

            middle = len(indices) // 2
            starting_frame = plane_frames[middle]
            slider = {"active": middle, "currentvalue": {"prefix": "Slice: "}, "steps": steps}
            sources = [trace.source for trace in starting_frame.data]
            update_data = {"source": sources, "dx": [dx] * 3, "dy": [dy] * 3}
            update_layout = {"title.text": title, "sliders": [slider]}
            for suffix in ("", "2", "3"):
                update_layout[f"xaxis{suffix}.autorange"] = True
                update_layout[f"yaxis{suffix}.autorange"] = "reversed"
            label = f"[{row_index}] {row['Scan']} / fold {row['Fold']} / {axis_name}"
            buttons.append({"label": label, "method": "update", "args": [update_data, update_layout, [0, 1, 2]]})
            if initial_frame is None:
                initial_frame = starting_frame
                initial_slider = slider
                figure.update_layout(title_text=title)

    for column, trace in enumerate(initial_frame.data, start=1):
        figure.add_trace(trace, row=1, col=column)
    for channel, name in names.items():
        if channel == 0 or region is not None and channel != region:
            continue
        red, green, blue = colors[channel].astype(int)
        color = f"rgb({red},{green},{blue})"
        marker = {"color": color, "symbol": "square", "size": 10}
        legend_trace = go.Scatter(x=[None], y=[None], mode="markers", marker=marker, name=f"{channel}: {name}")
        figure.add_trace(legend_trace, row=1, col=1)

    figure.frames = frames
    menu = {"buttons": buttons, "x": 0, "y": 1.16, "xanchor": "left", "yanchor": "top"}
    figure.update_layout(
        updatemenus=[menu], sliders=[initial_slider], template="plotly_white",
        height=650, margin={"t": 155, "b": 95},
        legend={"orientation": "h", "y": -0.2, "itemclick": False, "itemdoubleclick": False},
        uirevision="prediction_comparison",
    )
    for column in (1, 2, 3):
        anchor = "x"
        if column > 1:
            anchor = f"x{column}"
        figure.update_xaxes(showticklabels=False, row=1, col=column)
        figure.update_yaxes(showticklabels=False, scaleanchor=anchor, autorange="reversed", row=1, col=column)
    return figure


class FullVolumeEvaluation:
    """Evaluate scan-level legacy overlap metrics and optionally write reports.

    Model ROI channel count remains separate from evaluation class count.
    Existing ROI postprocessors default to the historical five output classes;
    use postprocessed_num_classes for a different label contract. This report
    retains legacy metric formulas; physical surface metrics live in metrics.py.
    """

    def __init__(
        self, model: Any, test_ids: Sequence[SubjectKey], config: Mapping[str, Any],
        loader: RodentDatasets, strides: Sequence[int], preprocessor: Preprocessor,
        brain_mask_postprocess_callback: PostprocessCallback | None = None,
        roi_postprocess_callback: PostprocessCallback | None = None,
        path: str | Path | None = None, verbose: bool = False, threshold: float = 0.5,
        *, postprocessed_num_classes: int | None = None,
    ) -> None:
        """Store inference settings without opening a log until evaluate is called."""
        self.patch_size: Shape3D = tuple(config['input_shape'])
        self.data_path: str = config['dataset_path']
        self.num_classes: int = config['num_classes']
        self.strides: list[int] = list(strides)
        if not self.strides:
            raise ValueError('At least one inference stride is required')
        self.threshold: float = threshold
        self.model: Any = model
        self.test_ids: list[SubjectKey] = list(test_ids)
        self.loader: RodentDatasets = loader
        self.modalities: list[str] = list(loader.get_modalities())
        self.preprocessor: Preprocessor = preprocessor
        self.results_path: Path | None = None
        if path is not None:
            self.results_path = Path(path)
        self.verbose: bool = verbose
        self.brain_mask_postprocess_callback = brain_mask_postprocess_callback
        self.roi_postprocess_callback = roi_postprocess_callback
        self.postprocessed_num_classes: int = 5
        if postprocessed_num_classes is not None:
            if not isinstance(postprocessed_num_classes, int) or postprocessed_num_classes < 1:
                raise ValueError('postprocessed_num_classes must be a positive integer')
            self.postprocessed_num_classes = postprocessed_num_classes
        self.logger: PrintLogger = PrintLogger(verbose=verbose)
        self.with_brain_mask: bool = True
        self.evaluate_brain_mask: bool = False
        factories = [dice_coefficient, precision_coefficient, specificity_coefficient,
                     sensitivity_coefficient, iou_coefficient, volume_similarity_coefficient]
        self.metrics: list[MetricFunction] = [factory(average='micro') for factory in factories]

    def load_subject(self, subject_key: SubjectKey) -> Rodent:
        """Retrieve a legacy or normalized loader record using its scan key."""
        return self.loader.get_subject(subject_key)

    def load_data(self, subject: Rodent) -> tuple[Any, Any, Any]:
        """Return image, ROI ground truth and optional brain ground truth."""
        return subject.get_images()

    def preprocess_data(self, image: Any, gt: Any) -> tuple[Any, Any]:
        """Apply the configured spatial/intensity pipeline to image and labels."""
        return self.preprocessor.preprocess(image, gt)

    def get_unique_test_ids(self) -> tuple[list[SubjectKey], list[list[SubjectKey]]]:
        """Deduplicate scan keys in input order and group by loader collection."""
        unique: list[SubjectKey] = list(dict.fromkeys(self.test_ids))
        groups: dict[str, list[SubjectKey]] = {modality: [] for modality in self.modalities}
        for key in unique:
            if not isinstance(key, tuple) or len(key) != 2 or key[0] not in groups:
                raise ValueError(f'Unknown loader scan key: {key!r}')
            self.load_subject(key)
            groups[key[0]].append(key)
        return unique, list(groups.values())

    def compute(self, with_brain_mask: bool = True, evaluate_brain_mask: bool = False) -> EvaluationResult:
        """Compute raw metrics and class means without creating plots or files."""
        if self.logger.log is not None and self.logger.log.closed:
            self.logger = PrintLogger(verbose=self.verbose)
        if evaluate_brain_mask and not with_brain_mask:
            raise ValueError('Brain evaluation requires with_brain_mask=True')
        self.with_brain_mask = with_brain_mask
        self.evaluate_brain_mask = evaluate_brain_mask
        initial_stride: int = self.strides[0]
        predictor = RandomCroppingPrediction(self.model, self.patch_size, self.num_classes,
                                              initial_stride, self.threshold)
        unique_ids, groups = self.get_unique_test_ids()
        if not unique_ids:
            raise ValueError('At least one evaluation scan is required')

        tables: list[pd.DataFrame] = []
        for stride in self.strides:
            self.logger.write(f'Evaluation for stride {stride}')
            table = self.evaluate_dataset(stride, groups, predictor)
            tables.append(table)
        combined: pd.DataFrame = pd.concat(tables).sort_index()
        return EvaluationResult.from_metrics(combined)

    def evaluate(self, with_brain_mask: bool = True, evaluate_brain_mask: bool = False,
                        *, show: bool = True) -> EvaluationResult:
        """Compute typed results and optionally export them with the report writer."""
        log_path: str | None = None
        if self.results_path is not None:
            self.results_path.mkdir(parents=True, exist_ok=True)
            log_path = str(self.results_path / 'evaluation.log')
        self.logger = PrintLogger(log_path, verbose=self.verbose)
        try:
            result: EvaluationResult = self.compute(with_brain_mask, evaluate_brain_mask)
            if self.results_path is not None:
                writer: EvaluationReportWriter = EvaluationReportWriter(self.results_path)
                writer.write(result, show=show)
            return result
        finally:
            self.logger.close()

    def evaluate_dataset(self, stride: int, unique_modality_test_ids: Sequence[Sequence[SubjectKey]],
                         croppingInference: RandomCroppingPrediction) -> pd.DataFrame:
        """Combine per-scan tables, retaining their collection group and stride."""
        croppingInference.set_stride(stride)
        tables: list[pd.DataFrame] = []
        for modality, keys in zip(self.modalities, unique_modality_test_ids):
            if not keys:
                continue
            subject_tables: list[pd.DataFrame] = []
            for key in keys:
                subject = self.load_subject(key)
                self.logger.write(f'Evaluation for subject {subject.get_id()}')
                table = self.compute_metrics_for_subject(subject, croppingInference)
                subject_tables.append(table)
            combined = pd.concat(subject_tables)
            combined.index.name = 'Metric'
            combined.insert(0, 'Stride', stride)
            combined.insert(1, 'Modality', modality)
            combined = combined.reset_index().set_index('Subject')
            tables.append(combined)
        if not tables:
            raise ValueError('No matching scans to evaluate')
        return pd.concat(tables)

    def compute_metrics_for_subject(self, subject: Rodent,
                                    croppingInference: RandomCroppingPrediction) -> pd.DataFrame:
        """Prepare one scan without mutating the model's ROI channel count."""
        image, roi_gt, brain_gt = self.load_data(subject)
        ground_truth: Any = roi_gt
        callback = self.roi_postprocess_callback
        output_name: str = 'roi'
        class_count: int = self.num_classes
        if self.evaluate_brain_mask:
            if brain_gt is None:
                raise ValueError(f'{subject.get_id()}: brain ground truth is missing')
            ground_truth = [roi_gt, brain_gt]
            callback = self.brain_mask_postprocess_callback
            output_name = 'brain_mask'
            class_count = 1
        elif callback is not None:
            class_count = self.postprocessed_num_classes

        processed_image, processed_gt = self.preprocess_data(image, ground_truth)
        if self.evaluate_brain_mask:
            processed_gt = processed_gt[1]
        outputs: PredictionResult = croppingInference.predict(processed_image, self.with_brain_mask)
        prediction: Array = outputs.roi
        if output_name == 'brain_mask':
            if outputs.brain_mask is None:
                raise ValueError('Requested brain mask is missing from prediction results')
            prediction = outputs.brain_mask
        if callback is not None:
            prediction = callback(prediction)
        reference: Array = processed_gt.get_fdata()
        if reference.shape != prediction.shape:
            raise ValueError('Prediction and preprocessed ground truth shapes differ')

        if class_count > 1:
            for values in (reference, prediction):
                if not np.isfinite(values).all() or np.any(values != np.rint(values)) or np.any(values < 0) or np.any(values >= class_count):
                    raise ValueError('Label values do not fit the evaluation class count; check mapping/postprocessed_num_classes')
            # Indexing preserves singleton spatial axes that to_categorical can squeeze.
            class_lookup: Array = np.eye(class_count, dtype=np.float32)
            reference_indices: Array = reference.astype(np.int64)
            prediction_indices: Array = prediction.astype(np.int64)
            reference = class_lookup[reference_indices]
            prediction = class_lookup[prediction_indices]
        else:
            reference = reference[..., np.newaxis]
            prediction = (prediction[..., np.newaxis] >= self.threshold)
        reference = reference[np.newaxis].astype(np.float32)
        prediction = prediction[np.newaxis].astype(np.float32)
        subject_id: str = subject.get_id()
        return self.compute_metrics_per_class(reference, prediction, subject_id)

    def compute_metrics_per_class(self, preprocessed_ground_truth: Array,
                                  final_prediction: Array, id: str) -> pd.DataFrame:
        """Apply the original metric callbacks independently to every class."""
        if preprocessed_ground_truth.shape != final_prediction.shape:
            raise ValueError('Encoded prediction and ground truth shapes differ')
        subject_metrics: dict[str, Any] = {'Subject': id}
        for class_index in range(final_prediction.shape[-1]):
            reference: Array = preprocessed_ground_truth[..., class_index]
            prediction: Array = final_prediction[..., class_index]
            values: dict[str, float] = {}
            for metric in self.metrics:
                values[metric.__name__] = float(metric(reference, prediction).numpy())
            subject_metrics[f'Class: {class_index}'] = values
        return pd.DataFrame(subject_metrics)
