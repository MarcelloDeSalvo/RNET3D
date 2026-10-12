"""Keras batches for ROI segmentation, brain masks, and domain adaptation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, TypeAlias

import numpy as np
import tensorflow as tf
from numpy.typing import NDArray

if TYPE_CHECKING:
    from .preprocessor import Preprocessor
    from ..utils.loader import Rodent, RodentDatasets, SubjectKey


BatchArray: TypeAlias = NDArray[np.float64]
Batch: TypeAlias = tuple[BatchArray, list[BatchArray]]


class MultiTaskGenerator(tf.keras.utils.Sequence):
    """Build full batches with ROI and brain-mask targets.

    Input volumes are repeated across the configured input channels. Targets
    are returned as ``[regions, brain_mask]``. The last incomplete batch is
    dropped by default, matching the existing training notebooks. Set
    drop_last=False to retain every scan. Partial labels and the classifier can
    be enabled independently. ROI values must already
    be mapped to contiguous class indexes by the supplied preprocessor.
    """

    def __init__(
        self,
        ids: Sequence[SubjectKey],
        loader: RodentDatasets,
        config: Mapping[str, Any],
        batch_size: int | None = None,
        shuffle: bool = True,
        preprocessor: Preprocessor | None = None,
        *,
        drop_last: bool = True,
        partial_labels: bool = False,
        with_classifier: bool = False,
    ) -> None:
        """Store batch settings; omit preprocessing for already prepared volumes."""
        self.ids: list[SubjectKey] = list(ids)
        self.loader: RodentDatasets = loader
        self.shuffle: bool = shuffle
        self.preprocessor: Preprocessor | None = preprocessor
        self.drop_last: bool = drop_last
        self.partial_labels: bool = partial_labels
        self.with_classifier: bool = with_classifier

        self.config: dict[str, Any] = dict(config)
        self.labels: list[int] = list(self.config["labels"])
        self.dim: tuple[int, ...] = tuple(self.config["input_shape"])
        self.dataset_path: str = self.config["dataset_path"]
        self.n_input_channels: int = self.config["in_channels"]
        self.n_classes: int = self.config["num_classes"]

        if batch_size is None:
            batch_size = self.config["batch_size"]

        self.batch_size: int = batch_size
        counts: tuple[int, ...] = (self.batch_size, self.n_input_channels, self.n_classes, *self.dim)
        for count in counts:
            if isinstance(count, bool) or not isinstance(count, (int, np.integer)) or count <= 0:
                raise ValueError("Batch size, channels, classes and dimensions must be positive integers")

        if len(self.dim) != 3:
            raise ValueError("input_shape must contain three dimensions")

        self.indexes: NDArray[np.int64]
        self.on_epoch_end()

        description: str = (
            f"Generator configuration:\n"
            f"- Dataset path: {self.dataset_path}\n"
            f"- Dimensions: {self.dim}\n"
            f"- Batch size: {self.batch_size}\n"
            f"- Number of input channels: {self.n_input_channels}\n"
            f"- Number of classes: {self.n_classes}\n"
            f"- Shuffle: {self.shuffle}\n"
        )
        print(description)

    def __len__(self) -> int:
        """Return complete batches, optionally including a shorter final batch."""
        count: int = len(self.ids) // self.batch_size
        remainder: int = len(self.ids) % self.batch_size
        if not self.drop_last and remainder:
            count += 1

        return count

    def __getitem__(self, index: int) -> Batch:
        """Return one valid batch; reject negative and out-of-range indexes."""
        if not isinstance(index, (int, np.integer)):
            raise TypeError("Batch index must be an integer")

        if index < 0 or index >= len(self):
            raise IndexError(f"Batch index out of range: {index}")

        start: int = index * self.batch_size
        stop: int = start + self.batch_size
        indexes: NDArray[np.int64] = self.indexes[start:stop]
        batch_ids: list[SubjectKey] = [self.ids[position] for position in indexes]

        return self._generate_batch(batch_ids)

    def __getcase__(self, index: int) -> SubjectKey:
        """Return an ID in original list order, independently of batch shuffling."""
        return self.ids[index]

    def on_epoch_end(self) -> None:
        """Reset batch indexes and optionally shuffle with NumPy's global RNG."""
        sample_count: int = len(self.ids)
        self.indexes = np.arange(sample_count, dtype=np.int64)

        if self.shuffle:
            np.random.shuffle(self.indexes)

    def _prepare_sample(self, subject: Rodent) -> tuple[BatchArray, NDArray, BatchArray]:
        """Prepare one image/mask group and validate class indexes before encoding."""
        image, roi_mask, brain_mask = subject.get_images()
        if brain_mask is None:
            raise ValueError(f"{subject.id}: a brain mask is required for multitask targets")

        if self.preprocessor is not None:
            masks = [roi_mask, brain_mask]
            image, processed_masks = self.preprocessor.preprocess(image, masks)
            roi_mask, brain_mask = processed_masks

        image_data: BatchArray = image.get_fdata()
        roi_data: BatchArray = roi_mask.get_fdata()
        brain_data: BatchArray = brain_mask.get_fdata()

        for data in (image_data, roi_data, brain_data):
            if data.shape != self.dim:
                raise ValueError(f"{subject.id}: expected shape {self.dim}, received {data.shape}")

            finite_values: NDArray[np.bool_] = np.isfinite(data)
            if not finite_values.all():
                raise ValueError(f"{subject.id}: images and masks must contain finite values")

        rounded_labels: NDArray = np.rint(roi_data)
        integer_labels: bool = np.array_equal(roi_data, rounded_labels)
        labels_in_range: bool = bool(roi_data.min() >= 0 and roi_data.max() < self.n_classes)
        if not integer_labels or not labels_in_range:
            raise ValueError(
                f"{subject.id}: ROI labels must be integer class indexes in [0, {self.n_classes}); "
                "configure MapLabels in the preprocessor"
            )

        # Flatten first: Keras otherwise removes a trailing singleton voxel axis.
        flat_labels: NDArray = roi_data.reshape(-1)
        flat_regions: NDArray = tf.keras.utils.to_categorical(flat_labels, num_classes=self.n_classes)
        region_shape: tuple[int, ...] = (*self.dim, self.n_classes)
        encoded_regions: NDArray = flat_regions.reshape(region_shape)

        return image_data, encoded_regions, brain_data

    def _generate_batch(self, batch_ids: Sequence[SubjectKey]) -> Batch:
        """Load each subject once and fill arrays with the historical float64 dtype."""
        sample_count: int = len(batch_ids)
        image_shape: tuple[int, ...] = (sample_count, *self.dim, self.n_input_channels)
        region_shape: tuple[int, ...] = (sample_count, *self.dim, self.n_classes)
        mask_shape: tuple[int, ...] = (sample_count, *self.dim, 1)

        images: BatchArray = np.zeros(image_shape, dtype=np.float64)
        regions: BatchArray = np.zeros(region_shape, dtype=np.float64)
        brain_masks: BatchArray = np.zeros(mask_shape, dtype=np.float64)
        subjects: list[Rodent] = []

        for position, subject_key in enumerate(batch_ids):
            subject: Rodent = self.loader.get_subject(subject_key)
            subjects.append(subject)
            image_data, region_data, brain_data = self._prepare_sample(subject)

            image_channels: NDArray = np.expand_dims(image_data, axis=-1)
            brain_channel: NDArray = np.expand_dims(brain_data, axis=-1)
            images[position] = image_channels
            regions[position] = region_data
            brain_masks[position] = brain_channel

        targets: list[BatchArray] = self._prepare_targets(subjects, regions, brain_masks)

        return images, targets

    def _prepare_targets(
        self, subjects: Sequence[Rodent], regions: BatchArray, brain_masks: BatchArray,
    ) -> list[BatchArray]:
        """Optionally mask unavailable mapped classes and add condition targets.

        A merged class is unavailable when any contributing source label lacks
        annotation. This is conservative: partial coverage is not a complete
        ground truth for the merged structure. Legacy defaults remain unchanged.
        """
        if self.partial_labels:
            mapping: dict[int, int] = self.config["label_mapping"]
            for position, subject in enumerate(subjects):
                unavailable: set[int] = set(subject.get_unavailable_labels() or ())
                channels: set[int] = set()
                for source, channel in mapping.items():
                    if source in unavailable:
                        channels.add(channel)

                for channel in channels:
                    regions[position, ..., channel] = -1

        targets: list[BatchArray] = [regions, brain_masks]
        if self.with_classifier:
            conditions: BatchArray = np.zeros((len(subjects), 1), dtype=np.float64)
            for position, subject in enumerate(subjects):
                if not subject.get_sham():
                    conditions[position, 0] = 1

            targets.append(conditions)

        return targets


class DomainAdaptationGenerator(MultiTaskGenerator):
    """Add missing-label sentinels and condition targets to multitask batches.

    Output order is ``[regions, brain_mask, condition]``. Unavailable source
    labels are matched against ``config['labels']`` in channel order and their
    entire channels are set to -1. Condition is 0 for SHAM and 1 otherwise.
    Annotation availability is supplied by the loader, never inferred here.
    """

    def _prepare_targets(
        self, subjects: Sequence[Rodent], regions: BatchArray, brain_masks: BatchArray,
    ) -> list[BatchArray]:
        """Preserve existing ignored-channel and SHAM/TBI target conventions."""
        condition_shape: tuple[int, int] = (len(subjects), 1)
        conditions: BatchArray = np.zeros(condition_shape, dtype=np.float64)

        for position, subject in enumerate(subjects):
            unavailable_labels: Sequence[int] | None = subject.get_unavailable_labels()
            if unavailable_labels:
                for channel, label in enumerate(self.labels):
                    if label in unavailable_labels:
                        if channel >= self.n_classes:
                            raise ValueError("Unavailable-label channel exceeds num_classes; check labels and mapping")

                        regions[position, ..., channel] = -1

            if not subject.get_sham():
                conditions[position, 0] = 1

        return [regions, brain_masks, conditions]
