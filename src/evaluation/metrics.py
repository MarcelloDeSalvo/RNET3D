"""Original training metric factories and separate physical full-volume scores."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import TypeAlias

import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np
import pandas as pd
from scipy import ndimage

from .losses import ignore_unavailable_ground_truth


MetricFunction: TypeAlias = Callable[[tf.Tensor, tf.Tensor], tf.Tensor]


def dice_coefficient(
    smooth: float = 1e-5, mode: str = 'binary', average: str = 'macro',
    domain_adaptation: bool = False, exclude_background: bool = True,
    class_index: int | None = None,
) -> MetricFunction:
    """Return binary (>0.5), soft or squared Dice with selectable class reduction."""
    def dice(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """Compute the existing smoothed Dice score over every supplied element."""
        union: tf.Tensor
        if mode == 'squared':
            squared_predictions: tf.Tensor = tf.square(y_pred)
            squared_targets: tf.Tensor = tf.square(y_true)
            union = K.sum(squared_predictions) + K.sum(squared_targets)
        elif mode == 'soft':
            union = K.sum(y_pred) + K.sum(y_true)
        elif mode == 'binary':
            predicted_positive: tf.Tensor = tf.greater(y_pred, 0.5)
            target_positive: tf.Tensor = tf.greater(y_true, 0.5)
            y_pred = tf.cast(predicted_positive, dtype=tf.float32)
            y_true = tf.cast(target_positive, dtype=tf.float32)
            union = K.sum(y_pred) + K.sum(y_true)
        else:
            raise ValueError('Mode not recognized. Choose from: soft, squared, binary')

        overlap: tf.Tensor = y_true * y_pred
        intersection: tf.Tensor = K.sum(overlap)
        score: tf.Tensor = (2. * intersection + smooth) / (union + smooth)

        return score

    return compute_metric(dice, average, domain_adaptation, exclude_background, class_index)


def volume_similarity_coefficient(
    average: str = 'macro', domain_adaptation: bool = False, exclude_background: bool = True,
) -> MetricFunction:
    """Return 1 - absolute volume difference / combined volume, with epsilon."""
    def volume_similarity(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """Threshold predictions at >0.5; retain the supplied target values."""
        y_pred = K.greater(y_pred, 0.5)
        y_pred = K.cast(y_pred, dtype='float32')
        volume_pred: tf.Tensor = K.sum(y_pred)
        volume_true: tf.Tensor = K.sum(y_true)

        difference: tf.Tensor = volume_pred - volume_true
        absolute_difference: tf.Tensor = K.abs(difference)
        sum_volumes: tf.Tensor = volume_true + volume_pred
        coefficient: tf.Tensor = 1 - absolute_difference / (sum_volumes + K.epsilon())

        return coefficient

    return compute_metric(volume_similarity, average, domain_adaptation, exclude_background)


def iou_coefficient(
    average: str = 'macro', domain_adaptation: bool = False, exclude_background: bool = True,
) -> MetricFunction:
    """Return IoU using >0.5 predictions and the original smoothed denominator."""
    def iou(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """Compute intersection and union over all supplied elements."""
        y_pred = K.greater(y_pred, 0.5)
        y_pred = K.cast(y_pred, dtype='float32')

        overlap: tf.Tensor = y_true * y_pred
        combined: tf.Tensor = K.maximum(y_true, y_pred)
        intersection: tf.Tensor = K.sum(overlap)
        union: tf.Tensor = K.sum(combined)
        score: tf.Tensor = (intersection + K.epsilon()) / (union + K.epsilon())

        return score

    return compute_metric(iou, average, domain_adaptation, exclude_background)


def _sum_rounded_clipped(values: tf.Tensor) -> tf.Tensor:
    """Preserve the shared clip-to-[0,1], backend-round, then sum sequence."""
    clipped: tf.Tensor = K.clip(values, 0, 1)
    rounded: tf.Tensor = K.round(clipped)

    return K.sum(rounded)


def accuracy_coefficient(
    average: str = 'macro', domain_adaptation: bool = False, exclude_background: bool = True,
) -> MetricFunction:
    """Return accuracy using the historical rounded soft confusion terms."""
    def accuracy(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """Reduce TP/TN/FP/FN across all supplied elements."""
        positive_overlap: tf.Tensor = y_true * y_pred
        negative_overlap: tf.Tensor = (1 - y_true) * (1 - y_pred)
        extra_predictions: tf.Tensor = (1 - y_true) * y_pred
        missed_targets: tf.Tensor = y_true * (1 - y_pred)

        true_positives: tf.Tensor = _sum_rounded_clipped(positive_overlap)
        true_negatives: tf.Tensor = _sum_rounded_clipped(negative_overlap)
        false_positives: tf.Tensor = _sum_rounded_clipped(extra_predictions)
        false_negatives: tf.Tensor = _sum_rounded_clipped(missed_targets)
        denominator: tf.Tensor = true_positives + true_negatives + false_positives + false_negatives + K.epsilon()
        score: tf.Tensor = (true_positives + true_negatives) / denominator

        return score

    return compute_metric(accuracy, average, domain_adaptation, exclude_background)


def precision_coefficient(
    average: str = 'macro', domain_adaptation: bool = False, exclude_background: bool = True,
) -> MetricFunction:
    """Return TP / predicted positives with the existing epsilon policy."""
    def precision(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """Use the original rounded/clipped overlap and prediction counts."""
        overlap: tf.Tensor = y_true * y_pred
        true_positives: tf.Tensor = _sum_rounded_clipped(overlap)
        predicted_positives: tf.Tensor = _sum_rounded_clipped(y_pred)
        score: tf.Tensor = true_positives / (predicted_positives + K.epsilon())

        return score

    return compute_metric(precision, average, domain_adaptation, exclude_background)


def sensitivity_coefficient(
    average: str = 'macro', domain_adaptation: bool = False, exclude_background: bool = True,
) -> MetricFunction:
    """Return TP / target positives with the existing epsilon policy."""
    def sensitivity(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """Use the original rounded/clipped overlap and target counts."""
        overlap: tf.Tensor = y_true * y_pred
        true_positives: tf.Tensor = _sum_rounded_clipped(overlap)
        possible_positives: tf.Tensor = _sum_rounded_clipped(y_true)
        score: tf.Tensor = true_positives / (possible_positives + K.epsilon())

        return score

    return compute_metric(sensitivity, average, domain_adaptation, exclude_background)


def specificity_coefficient(
    average: str = 'macro', domain_adaptation: bool = False, exclude_background: bool = True,
) -> MetricFunction:
    """Return TN / target negatives with the existing epsilon policy."""
    def specificity(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """Use the original rounded/clipped complementary counts."""
        negative_overlap: tf.Tensor = (1 - y_true) * (1 - y_pred)
        target_negatives: tf.Tensor = 1 - y_true
        true_negatives: tf.Tensor = _sum_rounded_clipped(negative_overlap)
        possible_negatives: tf.Tensor = _sum_rounded_clipped(target_negatives)
        score: tf.Tensor = true_negatives / (possible_negatives + K.epsilon())

        return score

    return compute_metric(specificity, average, domain_adaptation, exclude_background)


def ignore_background(y_true: tf.Tensor, y_pred: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    """Remove channel zero only for multichannel 5D inputs, as before."""
    shape: tuple = K.int_shape(y_true)
    if shape[-1] == 1 or len(shape) == 4:
        return y_true, y_pred

    y_true = y_true[:, :, :, :, 1:]
    y_pred = y_pred[:, :, :, :, 1:]

    return y_true, y_pred


def compute_metric(
    metric: MetricFunction, average: str = 'macro', domain_adaptation: bool = False,
    exclude_background: bool = False, class_index: int | None = None,
) -> MetricFunction:
    """Wrap a score in the existing masking and class-reduction policy.

    Macro scores pool the whole batch within each class, then average classes.
    Micro scores pool all supplied elements. A class index is interpreted after
    background removal. Masked classes remain in the average; values are zeroed.
    Returned names are preserved for Keras histories and saved-model loading.
    """
    def filter_classes(y_true: tf.Tensor, y_pred: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        """Apply unavailable-entry masking before optional background removal."""
        if domain_adaptation:
            y_true, y_pred = ignore_unavailable_ground_truth(y_true, y_pred)

        if exclude_background:
            y_true, y_pred = ignore_background(y_true, y_pred)

        return y_true, y_pred

    def macro_average(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """Average class scores after pooling batch and spatial dimensions."""
        y_true, y_pred = filter_classes(y_true, y_pred)
        class_count: tf.Tensor = K.shape(y_true)[-1]
        denominator: tf.Tensor = K.cast(class_count, dtype='float32')
        score_sum: tf.Tensor = tf.constant(0.0, dtype=tf.float32)

        for index in tf.range(class_count):
            class_target: tf.Tensor = y_true[:, :, :, :, index]
            class_prediction: tf.Tensor = y_pred[:, :, :, :, index]
            class_metric: tf.Tensor = metric(class_target, class_prediction)
            score_sum += class_metric

        return score_sum / denominator

    def micro_average(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """Compute one score across all filtered elements."""
        y_true, y_pred = filter_classes(y_true, y_pred)

        return metric(y_true, y_pred)

    def index_class(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """Select one class after masking and optional background removal."""
        y_true, y_pred = filter_classes(y_true, y_pred)
        class_target: tf.Tensor = y_true[:, :, :, :, class_index]
        class_prediction: tf.Tensor = y_pred[:, :, :, :, class_index]

        return metric(class_target, class_prediction)

    if class_index is not None:
        index_class.__name__ = f"class_{class_index}_{metric.__name__}"

        return index_class

    if average == 'macro':
        macro_average.__name__ = f"mean_{metric.__name__}"

        return macro_average

    if average == 'micro':
        micro_average.__name__ = metric.__name__

        return micro_average

    raise ValueError('Average not recognized. Choose from: macro, micro')


def binary_segmentation_metrics(
    reference: np.ndarray, prediction: np.ndarray, spacing_mm: Sequence[float],
) -> dict[str, float]:
    """Score aligned binary 3D masks using physical voxel sizes in array-axis order.

    Surfaces are 6-connected boundary voxel centres, including image edges.
    HD95 is the maximum of the two directed 95th percentiles; ASSD is the mean
    of the two directed means (not surface-area weighted). Both distances are
    NaN if either mask is empty. Both-empty Dice/IoU equal one; sensitivity and
    precision are NaN when their respective denominators are zero. RVE is the
    signed percentage (prediction-reference)/reference, NaN for zero reference.
    Inputs must already share an orthogonal physical grid; spacing alone cannot
    represent shear or detect differing origins/orientations.
    """
    reference = np.asarray(reference)
    prediction = np.asarray(prediction)
    spacing: np.ndarray = np.asarray(spacing_mm, dtype=float)
    if reference.ndim != 3 or reference.shape != prediction.shape or reference.size == 0:
        raise ValueError("Masks must be nonempty, matching 3D arrays")
    if spacing.shape != (3,) or not np.isfinite(spacing).all() or np.any(spacing <= 0):
        raise ValueError("spacing_mm must contain three finite positive values")
    for mask in (reference, prediction):
        if not np.isin(mask, [0, 1]).all():
            raise ValueError("Binary masks must contain only zero and one")

    reference = reference.astype(bool)
    prediction = prediction.astype(bool)
    reference_count: int = int(np.count_nonzero(reference))
    prediction_count: int = int(np.count_nonzero(prediction))
    intersection: int = int(np.count_nonzero(reference & prediction))
    total: int = reference_count + prediction_count
    union: int = total - intersection
    voxel_volume: float = float(np.prod(spacing))
    reference_volume: float = reference_count * voxel_volume
    prediction_volume: float = prediction_count * voxel_volume

    dice: float = 1.0
    iou: float = 1.0
    sensitivity: float = float('nan')
    precision: float = float('nan')
    relative_error: float = float('nan')
    hd95: float = float('nan')
    assd: float = float('nan')
    if total:
        dice = 2 * intersection / total
        iou = intersection / union
    if reference_count:
        sensitivity = intersection / reference_count
        relative_error = 100 * (prediction_volume - reference_volume) / reference_volume
    if prediction_count:
        precision = intersection / prediction_count

    if reference_count and prediction_count:
        connectivity: np.ndarray = ndimage.generate_binary_structure(3, 1)
        eroded_reference = ndimage.binary_erosion(reference, structure=connectivity, border_value=0)
        eroded_prediction = ndimage.binary_erosion(prediction, structure=connectivity, border_value=0)
        reference_surface: np.ndarray = reference & ~eroded_reference
        prediction_surface: np.ndarray = prediction & ~eroded_prediction
        outside_reference: np.ndarray = ~reference_surface
        outside_prediction: np.ndarray = ~prediction_surface
        to_reference = ndimage.distance_transform_edt(outside_reference, sampling=spacing)
        to_prediction = ndimage.distance_transform_edt(outside_prediction, sampling=spacing)
        forward: np.ndarray = to_prediction[reference_surface]
        backward: np.ndarray = to_reference[prediction_surface]
        forward_hd95: float = float(np.percentile(forward, 95))
        backward_hd95: float = float(np.percentile(backward, 95))
        hd95 = max(forward_hd95, backward_hd95)
        assd = float((forward.mean() + backward.mean()) / 2)

    return {
        'Dice': dice, 'IoU': iou, 'HD95': hd95, 'ASSD': assd,
        'AVE': abs(prediction_volume - reference_volume), 'RVE': relative_error,
        'Sensitivity': sensitivity, 'Precision': precision,
        'ReferenceVolume': reference_volume, 'PredictionVolume': prediction_volume,
    }


def segmentation_metrics_table(
    reference: np.ndarray, prediction: np.ndarray, spacing_mm: Sequence[float],
    regions: Mapping[int, str], *, subject: str, scan: str = '', dataset: str = '',
    available_labels: Sequence[int] | None = None,
) -> pd.DataFrame:
    """Return long-form full-volume metrics for explicit anatomical label values.

    Label maps must already be on the same physical grid. Select source-mask or
    network label IDs consistently; no remapping/argmax/threshold is inferred.
    Unannotated regions retain rows with NaN values and Status='unavailable'.
    An absent annotated region is distinct from an unavailable annotation.
    """
    reference = np.asarray(reference)
    prediction = np.asarray(prediction)
    if reference.ndim != 3 or reference.shape != prediction.shape or reference.size == 0:
        raise ValueError('Label maps must be nonempty, matching 3D arrays')
    for label_map in (reference, prediction):
        if not np.isfinite(label_map).all() or np.any(label_map != np.rint(label_map)) or np.any(label_map < 0):
            raise ValueError('Label maps must contain finite nonnegative integer labels')
    if not regions or any(not isinstance(label, (int, np.integer)) or label < 0 for label in regions):
        raise ValueError('regions must map nonnegative integer label IDs to names')
    if not subject:
        raise ValueError('A subject identifier is required')

    available: set[int] = set(regions)
    if available_labels is not None:
        available = set(available_labels)
    units: dict[str, str] = {'HD95': 'mm', 'ASSD': 'mm', 'AVE': 'mm3', 'RVE': '%',
                             'ReferenceVolume': 'mm3', 'PredictionVolume': 'mm3'}
    rows: list[dict[str, object]] = []
    for label, region in regions.items():
        reference_mask: np.ndarray = reference == label
        prediction_mask: np.ndarray = prediction == label
        values: dict[str, float] = binary_segmentation_metrics(reference_mask, prediction_mask, spacing_mm)
        status: str = 'ok'
        if label not in available:
            status = 'unavailable'
            values = dict.fromkeys(values, float('nan'))
        elif not reference_mask.any() and not prediction_mask.any():
            status = 'both_empty'
        elif not reference_mask.any():
            status = 'reference_empty'
        elif not prediction_mask.any():
            status = 'prediction_empty'

        for metric, value in values.items():
            unit: str = units.get(metric, 'fraction')
            rows.append({'Subject': subject, 'Scan': scan, 'Dataset': dataset,
                         'Region': region, 'Label': label, 'Metric': metric,
                         'Value': value, 'Unit': unit, 'Status': status})

    return pd.DataFrame(rows)
