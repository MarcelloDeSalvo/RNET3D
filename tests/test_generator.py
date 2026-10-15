"""Check batch boundaries, target encoding and generator compatibility."""

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

pytest.importorskip("tensorflow")

from src.preprocessing.generator import DomainAdaptationGenerator, MultiTaskGenerator
from src.utils.loader import RodentDatasets


@pytest.fixture
def dataset(tmp_path: Path) -> tuple[RodentDatasets, dict]:
    """Create three small scans, including a singleton spatial axis."""
    shape = (2, 3, 1)
    for index in range(3):
        subject_id = f"mouse{index}"
        folder = tmp_path / "FLASH" / subject_id / "Anat"
        folder.mkdir(parents=True)
        arrays = {
            "N4": np.full(shape, index + 1, dtype=float),
            "Labels": np.full(shape, index % 2, dtype=np.int16),
            "brain_mask": np.ones(shape, dtype=np.uint8),
        }

        for name, values in arrays.items():
            affine = np.eye(4)
            image = nib.Nifti1Image(values, affine)
            path = folder / f"{subject_id}_{name}.nii.gz"
            nib.save(image, path)

    loader = RodentDatasets([0, 1])
    loader.add_dataset(tmp_path, "FLASH", sub_folder="Anat")
    config = {
        "labels": [0, 1], "input_shape": shape, "dataset_path": str(tmp_path),
        "in_channels": 3, "num_classes": 2, "batch_size": 2,
    }

    return loader, config


@pytest.mark.parametrize("generator_type", [MultiTaskGenerator, DomainAdaptationGenerator])
def test_batches_preserve_channels_dtype_and_singleton_axis(dataset, generator_type):
    loader, config = dataset
    ids = loader.get_subjects_list()
    generator = generator_type(ids, loader, config, shuffle=False)

    images, targets = generator[0]

    assert len(generator) == 1  # The third scan remains outside the complete batch.
    assert images.shape == (2, 2, 3, 1, 3)
    assert targets[0].shape == (2, 2, 3, 1, 2)
    assert targets[1].shape == (2, 2, 3, 1, 1)
    assert images.dtype == np.float64
    assert all(target.dtype == np.float64 for target in targets)
    assert np.all(images[0] == 1) and np.all(images[1] == 2)
    assert np.all(targets[0][0, ..., 0] == 1)
    assert np.all(targets[0][1, ..., 1] == 1)
    assert np.all(targets[1] == 1)

    if generator_type is DomainAdaptationGenerator:
        np.testing.assert_array_equal(targets[2], [[1], [1]])
    else:
        assert len(targets) == 2


def test_domain_targets_keep_raw_label_order_and_subject_specific_flags(dataset):
    loader, config = dataset
    config["labels"] = [0, 13]
    ids = loader.get_subjects_list()
    first = loader.get_subject(ids[0])
    first.unavailable_labels = [13]
    first.sham = True
    generator = DomainAdaptationGenerator(ids, loader, config, shuffle=False)

    _, targets = generator[0]

    assert np.all(targets[0][0, ..., 1] == -1)
    assert np.all(targets[0][1, ..., 1] == 1)
    np.testing.assert_array_equal(targets[2], [[0], [1]])


def test_missing_as_background_projects_undeclared_positives_and_keeps_one_hot_targets(dataset):
    loader, config = dataset
    config["label_mapping"] = {0: 0, 1: 1}
    ids = loader.get_subjects_list()
    subject = loader.get_subject(ids[1])
    subject.unavailable_labels = [1]
    generator = MultiTaskGenerator(
        ids, loader, config, shuffle=False, missing_labels_as_background=True,
    )

    _, targets = generator[0]
    assert len(targets) == 2
    assert np.all(targets[0][1, ..., 0] == 1)
    assert np.all(targets[0][1, ..., 1] == 0)
    assert np.all(targets[0].sum(axis=-1) == 1)
    assert np.all(targets[0] >= 0)
    assert np.all(targets[1] == 1)
    assert subject.unavailable_labels == [1]

    with pytest.raises(ValueError, match="not both"):
        MultiTaskGenerator(ids, loader, config, partial_labels=True, missing_labels_as_background=True)


@pytest.mark.parametrize("index", [-1, 1, 50])
def test_invalid_batch_indexes_raise(dataset, index):
    loader, config = dataset
    ids = loader.get_subjects_list()
    generator = MultiTaskGenerator(ids, loader, config, shuffle=False)

    with pytest.raises(IndexError, match="out of range"):
        generator[index]


@pytest.mark.parametrize("batch_size", [0, -1, 1.5, True])
def test_invalid_batch_size_is_rejected(dataset, batch_size):
    loader, config = dataset
    ids = loader.get_subjects_list()

    with pytest.raises(ValueError, match="positive integers"):
        MultiTaskGenerator(ids, loader, config, batch_size=batch_size)


@pytest.mark.parametrize("value", [-1, 0.5, 2, np.nan])
def test_invalid_roi_values_are_not_silently_encoded(dataset, value):
    loader, config = dataset
    ids = loader.get_subjects_list()
    subject = loader.get_subject(ids[0])
    values = np.full(config["input_shape"], value, dtype=float)
    subject.labels = nib.Nifti1Image(values, subject.image.affine)
    generator = MultiTaskGenerator(ids, loader, config, shuffle=False)

    with pytest.raises(ValueError, match="finite|class indexes"):
        generator[0]


def test_missing_brain_mask_is_reported(dataset):
    loader, config = dataset
    ids = loader.get_subjects_list()
    loader.get_subject(ids[0]).brain_mask = None
    generator = MultiTaskGenerator(ids, loader, config, shuffle=False)

    with pytest.raises(ValueError, match="brain mask is required"):
        generator[0]


def test_wrong_preprocessed_shape_is_reported(dataset):
    loader, config = dataset
    ids = loader.get_subjects_list()
    config["input_shape"] = (3, 3, 3)
    generator = MultiTaskGenerator(ids, loader, config, shuffle=False)

    with pytest.raises(ValueError, match="expected shape"):
        generator[0]


def test_generator_owns_ids_and_preserves_original_case_order(dataset, monkeypatch):
    loader, config = dataset
    ids = loader.get_subjects_list()
    original_ids = ids.copy()

    def reverse_indexes(indexes):
        indexes[:] = indexes[::-1]

    monkeypatch.setattr(np.random, "shuffle", reverse_indexes)
    generator = MultiTaskGenerator(ids, loader, config, shuffle=True)
    ids.clear()

    assert generator.ids == original_ids
    assert generator.__getcase__(0) == original_ids[0]
    images, _ = generator[0]
    assert np.all(images[0] == 3)

    generator.on_epoch_end()
    np.testing.assert_array_equal(generator.indexes, [2, 1, 0])
