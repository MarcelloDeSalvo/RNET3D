"""Loss factories preserving the existing formulas, reductions and masking policy."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TypeAlias

import tensorflow as tf
import tensorflow.keras.backend as K


LossFunction: TypeAlias = Callable[[tf.Tensor, tf.Tensor], tf.Tensor]
Axes: TypeAlias = int | Sequence[int]


def diceLoss(smooth: float = 1e-5, axis: Axes = (1, 2, 3)) -> LossFunction:
    """Return mean spatial Dice loss; preserve the historical doubled smoothing."""
    def loss(target: tf.Tensor, predicted: tf.Tensor) -> tf.Tensor:
        """Reduce spatial axes first, then average samples and classes."""
        overlap: tf.Tensor = predicted * target
        intersection: tf.Tensor = K.sum(overlap, axis=axis)
        union: tf.Tensor = K.sum(predicted, axis=axis) + K.sum(target, axis=axis)

        loss_per_sample: tf.Tensor = 1 - 2 * (intersection + smooth) / (union + smooth)

        return K.mean(loss_per_sample)

    return loss


def batchDiceLoss(smooth: float = 1e-5) -> LossFunction:
    """Return class-averaged Dice loss, pooling each class across the whole batch."""
    def computeScore(target: tf.Tensor, predicted: tf.Tensor, smooth: float) -> tf.Tensor:
        """Compute the original flattened-batch Dice score for one class."""
        predicted = tf.reshape(predicted, [-1])
        target = tf.reshape(target, [-1])

        overlap: tf.Tensor = predicted * target
        intersection: tf.Tensor = K.sum(overlap)
        union: tf.Tensor = K.sum(predicted) + K.sum(target)
        score: tf.Tensor = 2 * (intersection + smooth) / (union + smooth)

        return score

    def loss(target: tf.Tensor, predicted: tf.Tensor) -> tf.Tensor:
        """Accumulate class scores in their original order."""
        score_sum: tf.Tensor = tf.constant(0.0, dtype=predicted.dtype)
        class_count: tf.Tensor = K.shape(target)[-1]
        denominator: tf.Tensor = K.cast(class_count, 'float32')

        for index in tf.range(class_count):
            class_target: tf.Tensor = target[:, :, :, :, index]
            class_prediction: tf.Tensor = predicted[:, :, :, :, index]
            score: tf.Tensor = computeScore(class_target, class_prediction, smooth)
            score_sum += score

        return 1 - score_sum / denominator

    return loss


def batchTverskyLoss(alpha: float | list[float] = 0.5, smooth: float = 1e-5) -> LossFunction:
    """Return class-averaged batch Tversky loss; alpha weights false negatives."""
    def computeScore(
        target: tf.Tensor, predicted: tf.Tensor, alpha: float | tf.Tensor, smooth: float,
    ) -> tf.Tensor:
        """Compute one class score after flattening batch and spatial axes."""
        y_true_pos: tf.Tensor = K.flatten(target)
        y_pred_pos: tf.Tensor = K.flatten(predicted)
        overlap: tf.Tensor = y_true_pos * y_pred_pos
        missed_targets: tf.Tensor = y_true_pos * (1 - y_pred_pos)
        extra_predictions: tf.Tensor = (1 - y_true_pos) * y_pred_pos
        true_pos: tf.Tensor = K.sum(overlap)
        false_neg: tf.Tensor = K.sum(missed_targets)
        false_pos: tf.Tensor = K.sum(extra_predictions)

        denominator: tf.Tensor = true_pos + alpha * false_neg + (1 - alpha) * false_pos + smooth
        score: tf.Tensor = (true_pos + smooth) / denominator

        return score

    def loss(target: tf.Tensor, predicted: tf.Tensor) -> tf.Tensor:
        """Select scalar or per-class alpha and retain equal class weighting."""
        score_sum: tf.Tensor = tf.constant(0.0, dtype=predicted.dtype)
        class_count: tf.Tensor = K.shape(target)[-1]
        denominator: tf.Tensor = K.cast(class_count, 'float32')
        alpha_values: tf.Tensor = tf.convert_to_tensor(alpha, dtype=predicted.dtype)

        for index in tf.range(class_count):
            current_alpha: tf.Tensor = alpha_values
            if isinstance(alpha, list):
                current_alpha = alpha_values[index]

            class_target: tf.Tensor = target[:, :, :, :, index]
            class_prediction: tf.Tensor = predicted[:, :, :, :, index]
            score: tf.Tensor = computeScore(class_target, class_prediction, current_alpha, smooth)
            score_sum += score

        return 1 - score_sum / denominator

    return loss


def tverskyLoss(
    alpha: float = 0.5, smooth: float = 1e-5, sum_axis: Axes = (1, 2, 3),
    focal: float | None = None,
) -> LossFunction:
    """Return spatial Tversky loss, optionally raising its complement to a power."""
    def loss(target: tf.Tensor, predicted: tf.Tensor) -> tf.Tensor:
        """Reduce spatial overlap and then average sample/class scores."""
        overlap: tf.Tensor = predicted * target
        missed_targets: tf.Tensor = target * (1 - predicted)
        extra_predictions: tf.Tensor = (1 - target) * predicted
        true_positives: tf.Tensor = K.sum(overlap, axis=sum_axis)
        false_negatives: tf.Tensor = K.sum(missed_targets, axis=sum_axis)
        false_positives: tf.Tensor = K.sum(extra_predictions, axis=sum_axis)
        denominator: tf.Tensor = true_positives + alpha * false_negatives + (1 - alpha) * false_positives + smooth
        tversky: tf.Tensor = (true_positives + smooth) / denominator

        if focal is not None:
            complement: tf.Tensor = 1 - tversky
            focal_loss: tf.Tensor = tf.math.pow(complement, focal)

            return K.mean(focal_loss)

        return 1 - K.mean(tversky)

    return loss


def weightedBinaryCrossentropy(alpha: float = 0.5, smooth: float = K.epsilon()) -> LossFunction:
    """Return elementwise mean binary CE with positive weight alpha."""
    def loss(target: tf.Tensor, predicted: tf.Tensor) -> tf.Tensor:
        """Clip probabilities and preserve positive/negative weighting."""
        upper_bound: float = 1 - smooth
        predicted = tf.clip_by_value(predicted, smooth, upper_bound)
        complement: tf.Tensor = 1 - predicted
        cross_entropy: tf.Tensor = alpha * target * K.log(predicted) + (1 - alpha) * (1 - target) * K.log(complement)

        return -K.mean(cross_entropy)

    return loss


def focalLoss(gamma: float = 2.0, alpha: float = 1, smooth: float = K.epsilon()) -> LossFunction:
    """Return mean positive-target focal CE, including the class axis in the mean."""
    def loss(target: tf.Tensor, predicted: tf.Tensor) -> tf.Tensor:
        """Apply focal weighting to clipped positive-target log probabilities."""
        upper_bound: float = 1 - smooth
        predicted = tf.clip_by_value(predicted, smooth, upper_bound)
        cross_entropy: tf.Tensor = target * K.log(predicted)
        complement: tf.Tensor = 1 - predicted
        focal_weight: tf.Tensor = K.pow(complement, gamma)
        focal: tf.Tensor = -alpha * focal_weight * cross_entropy

        return K.mean(focal)

    return loss


def crossEntropyLoss(smooth: float = K.epsilon()) -> LossFunction:
    """Return mean -target*log(probability), rather than a sum over classes."""
    def loss(target: tf.Tensor, predicted: tf.Tensor) -> tf.Tensor:
        """Clip probabilities before reducing all elements."""
        upper_bound: float = 1 - smooth
        predicted = tf.clip_by_value(predicted, smooth, upper_bound)
        cross_entropy: tf.Tensor = -target * K.log(predicted)

        return K.mean(cross_entropy)

    return loss


def diceCELoss(
    gamma: float = 0.5, smooth: float = 1e-5, batch_wise: bool = False,
    focal_gamma: float | None = None, domain_adaptation: bool = False,
) -> LossFunction:
    """Blend Dice and CE; optional -1 masking happens after the model activation."""
    if batch_wise:
        dice_function: LossFunction = batchDiceLoss(smooth)
    else:
        dice_function = diceLoss(smooth=smooth)

    if focal_gamma is None:
        ce_function: LossFunction = crossEntropyLoss(smooth)
    else:
        ce_function = focalLoss(focal_gamma, smooth=smooth)

    def loss(target: tf.Tensor, predicted: tf.Tensor) -> tf.Tensor:
        """Keep the existing masking, weighting and class reductions."""
        if domain_adaptation:
            target, predicted = ignore_unavailable_ground_truth(target, predicted)

        dice_loss: tf.Tensor = dice_function(target, predicted)
        ce_loss: tf.Tensor = ce_function(target, predicted)
        total_loss: tf.Tensor = gamma * ce_loss + (1 - gamma) * dice_loss

        return total_loss

    return loss


def tverskyCELoss(
    alpha: float | list[float] = 0.5, smooth: float = 1e-5, gamma: float = 0.5,
    batch_wise: bool = False, focal_gamma: float | None = None,
) -> LossFunction:
    """Blend the existing Tversky and positive-target CE/focal objectives."""
    if batch_wise:
        tversky_function: LossFunction = batchTverskyLoss(alpha, smooth)
    else:
        tversky_function = tverskyLoss(alpha, smooth)

    if focal_gamma is None:
        ce_function: LossFunction = crossEntropyLoss(smooth)
    else:
        ce_function = focalLoss(focal_gamma, smooth=smooth)

    def loss(target: tf.Tensor, predicted: tf.Tensor) -> tf.Tensor:
        """Combine overlap and CE with the historical gamma weighting."""
        tversky_loss: tf.Tensor = tversky_function(target, predicted)
        ce_loss: tf.Tensor = ce_function(target, predicted)
        total_loss: tf.Tensor = gamma * ce_loss + (1 - gamma) * tversky_loss

        return total_loss

    return loss


def diceBCELoss(
    gamma: float = 0.5, alpha: float = 0.5, smooth: float = 1e-5, batch_wise: bool = False,
) -> LossFunction:
    """Blend Dice and weighted binary cross-entropy with unchanged reductions."""
    if batch_wise:
        dice_function: LossFunction = batchDiceLoss(smooth)
    else:
        dice_function = diceLoss(smooth=smooth)

    bce_function: LossFunction = weightedBinaryCrossentropy(alpha, smooth)

    def loss(target: tf.Tensor, predicted: tf.Tensor) -> tf.Tensor:
        """Combine overlap and BCE with the historical gamma weighting."""
        dice_loss: tf.Tensor = dice_function(target, predicted)
        bce_loss: tf.Tensor = bce_function(target, predicted)
        total_loss: tf.Tensor = gamma * bce_loss + (1 - gamma) * dice_loss

        return total_loss

    return loss


def ignore_unavailable_ground_truth(
    y_true: tf.Tensor, y_pred: tf.Tensor, ignore_value: float = -1,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Zero masked target/probability entries without renormalizing probabilities.

    This blocks direct derivatives at masked probabilities, but does not remove
    their logits from an upstream softmax denominator or from class averaging.
    """
    mask: tf.Tensor = tf.equal(y_true, ignore_value)
    zero_targets: tf.Tensor = tf.zeros_like(y_true)
    zero_predictions: tf.Tensor = tf.zeros_like(y_pred)
    y_true = tf.where(mask, zero_targets, y_true)
    y_pred = tf.where(mask, zero_predictions, y_pred)

    return y_true, y_pred

# ------------------------------------------------------------------------------------
# Gradient-preserving loss factories for pre-softmax logits, masking unavailable classes.
# ------------------------------------------------------------------------------------

def masked_softmax_gradient_fix(
    target: tf.Tensor, logits: tf.Tensor, ignore_value: float = -1,
) -> tf.Tensor:
    """Normalize raw logits over available classes only, along the final axis.

    Unavailable logits have exactly zero derivative through this operation.
    An entirely unavailable voxel returns zeros without evaluating softmax on
    an all-negative-infinity vector. Float16 inputs are promoted to float32.
    """
    logits = tf.convert_to_tensor(logits)
    if not logits.dtype.is_floating:
        raise TypeError("logits must have a floating-point dtype")

    if logits.dtype in (tf.float16, tf.bfloat16):
        logits = tf.cast(logits, tf.float32)

    target = tf.cast(target, logits.dtype)
    target_shape: tf.Tensor = tf.shape(target)
    logits_shape: tf.Tensor = tf.shape(logits)
    tf.debugging.assert_equal(target_shape, logits_shape, message="Target and logits shapes must match")
    tf.debugging.assert_all_finite(logits, "Logits must be finite")
    tf.debugging.assert_all_finite(target, "Targets must be finite")

    available: tf.Tensor = tf.not_equal(target, ignore_value)
    has_available: tf.Tensor = tf.reduce_any(available, axis=-1, keepdims=True)
    negative_infinity: tf.Tensor = tf.constant(float('-inf'), dtype=logits.dtype)
    zeros: tf.Tensor = tf.zeros_like(logits)

    masked_logits: tf.Tensor = tf.where(available, logits, negative_infinity)
    safe_logits: tf.Tensor = tf.where(has_available, masked_logits, zeros)
    probabilities: tf.Tensor = tf.nn.softmax(safe_logits, axis=-1)
    probabilities = tf.where(available, probabilities, zeros)

    return probabilities


def diceCELoss_gradient_fix(
    gamma: float = 0.5, smooth: float = 1e-5, batch_wise: bool = False,
    focal_gamma: float | None = None, *, ignore_value: float = -1,
) -> LossFunction:
    """Return partial-label Dice/CE on RAW ROI LOGITS, not softmax outputs.

    Mask unavailable classes before softmax, then reduce only supervised entries
    and sample/class groups. Voxels with no positive available target and batches
    with no supervision contribute zero loss and zero gradient. This conditional
    objective supplies no negative-only supervision at those unknown voxels.

    Dice smoothing and CE's mean-over-classes convention match diceCELoss when
    all classes are available and every voxel has a positive target. Existing
    diceCELoss and inference models are unchanged. Do not compile this callback
    directly against an existing softmax head: expose its pre-activation logits.
    """
    if not 0 <= gamma <= 1:
        raise ValueError("gamma must be between zero and one")

    if not 0 < smooth < 0.5:
        raise ValueError("smooth must be between zero and 0.5")

    if focal_gamma is not None and not 0 <= focal_gamma < float('inf'):
        raise ValueError("focal_gamma must be finite and nonnegative")

    if not float('-inf') < ignore_value < 0:
        raise ValueError("ignore_value must be finite and negative")

    reduction_axes: tuple[int, ...]
    if batch_wise:
        reduction_axes = (0, 1, 2, 3)
    else:
        reduction_axes = (1, 2, 3)

    def loss_gradient_fix(target: tf.Tensor, logits: tf.Tensor) -> tf.Tensor:
        """Compute conditional overlap/CE without dependence on ignored logits."""
        tf.debugging.assert_rank(logits, 5, message="Expected channels-last 3D batches")
        probabilities: tf.Tensor = masked_softmax_gradient_fix(target, logits, ignore_value)
        target = tf.cast(target, probabilities.dtype)
        available: tf.Tensor = tf.not_equal(target, ignore_value)
        zeros: tf.Tensor = tf.zeros_like(probabilities)
        clean_target: tf.Tensor = tf.where(available, target, zeros)
        lower_target_bound: tf.Tensor = tf.constant(0.0, dtype=probabilities.dtype)
        upper_target_bound: tf.Tensor = tf.constant(1.0, dtype=probabilities.dtype)
        tf.debugging.assert_greater_equal(clean_target, lower_target_bound, message="Available targets must be nonnegative")
        tf.debugging.assert_less_equal(clean_target, upper_target_bound, message="Available targets must not exceed one")

        target_mass: tf.Tensor = tf.reduce_sum(clean_target, axis=-1, keepdims=True)
        supervised_voxels: tf.Tensor = target_mass > 0
        valid_entries: tf.Tensor = tf.logical_and(available, supervised_voxels)
        probabilities = tf.where(supervised_voxels, probabilities, zeros)

        overlap: tf.Tensor = probabilities * clean_target
        intersection: tf.Tensor = tf.reduce_sum(overlap, axis=reduction_axes)
        prediction_mass: tf.Tensor = tf.reduce_sum(probabilities, axis=reduction_axes)
        truth_mass: tf.Tensor = tf.reduce_sum(clean_target, axis=reduction_axes)
        union: tf.Tensor = prediction_mass + truth_mass
        dice_terms: tf.Tensor = 1 - 2 * (intersection + smooth) / (union + smooth)

        active_classes: tf.Tensor = tf.reduce_any(valid_entries, axis=reduction_axes)
        zero_dice: tf.Tensor = tf.zeros_like(dice_terms)
        dice_terms = tf.where(active_classes, dice_terms, zero_dice)
        class_weights: tf.Tensor = tf.cast(active_classes, probabilities.dtype)
        dice_sum: tf.Tensor = tf.reduce_sum(dice_terms)
        class_count: tf.Tensor = tf.reduce_sum(class_weights)
        dice_loss: tf.Tensor = tf.math.divide_no_nan(dice_sum, class_count)

        upper_bound: float = 1 - smooth
        clipped: tf.Tensor = tf.clip_by_value(probabilities, smooth, upper_bound)
        log_probabilities: tf.Tensor = tf.math.log(clipped)
        ce_terms: tf.Tensor = -clean_target * log_probabilities

        if focal_gamma is not None:
            complement: tf.Tensor = 1 - clipped
            focal_weights: tf.Tensor = tf.pow(complement, focal_gamma)
            ce_terms = focal_weights * ce_terms

        ce_terms = tf.where(valid_entries, ce_terms, zeros)
        entry_weights: tf.Tensor = tf.cast(valid_entries, probabilities.dtype)
        ce_sum: tf.Tensor = tf.reduce_sum(ce_terms)
        entry_count: tf.Tensor = tf.reduce_sum(entry_weights)
        ce_loss: tf.Tensor = tf.math.divide_no_nan(ce_sum, entry_count)
        total_loss: tf.Tensor = gamma * ce_loss + (1 - gamma) * dice_loss

        return total_loss

    return loss_gradient_fix
