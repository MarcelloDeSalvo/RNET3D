"""Sliding-window predictions and typed full-volume evaluation reports.

Inputs must already share the intended preprocessing/orientation. Model heads
must return probabilities. ROI logits from gradient-fixed training require an
explicit probability-producing inference view. Patch order and repeated edge
patches retain the historical weighting; classifier output is a majority vote.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeAlias

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
    from ..utils.loader import Rodent, RodentDatasets, SubjectKey
    from ..preprocessing.preprocessor import Preprocessor

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
        if with_brain_mask:
            brain_mask = self.calculate_final_brain_mask(brain_sums, brain_counts, original_shape)
        if with_classifier:
            classifier_output = self.calculate_final_classifier_output(positive_votes, total_votes)
        roi_mask: Array = self.calculate_final_prediction(sums, counts, original_shape)
        return PredictionResult(roi=roi_mask, brain_mask=brain_mask, classifier=classifier_output)


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
