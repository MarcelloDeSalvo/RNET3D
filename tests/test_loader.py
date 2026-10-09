"""Exercise both filesystem layouts with small, real NIfTI images."""

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from src.utils.loader import RodentDatasets, load_data


def write_scan(folder: Path, prefix: str = "", gz: bool = True) -> None:
    """Write a complete scan with distinguishable image and mask values."""
    folder.mkdir(parents=True, exist_ok=True)
    affine = np.diag([0.1, 0.2, 0.3, 1.0])
    for name, value in [("N4", 2), ("Labels", 1), ("brain_mask", 1)]:
        image = nib.Nifti1Image(np.full((4, 4, 4), value, dtype=np.int16), affine)
        nib.save(image, folder / f"{prefix}{name}{'.nii.gz' if gz else '.nii'}")


def write_normalized(
    root: Path,
    dataset: str = "D1",
    subject: str = "SUBJ_0001",
    timepoints: tuple[str, ...] = (),
    modality: str = "T1w",
    condition: str = "TBI",
) -> Path:
    """Write subject metadata and its single or longitudinal scan folders."""
    folder = root / dataset / subject
    for timepoint in timepoints or (None,):
        write_scan(folder / timepoint if timepoint else folder)
    metadata = {
        "schema_version": 3,
        "dataset": dataset,
        "normalized_id": subject,
        "original_id": "original-animal",
        "modality": modality,
        "condition": condition,
        "subgroup": "example-subgroup",
        "timepoints": list(timepoints),
        "label_configuration": 4,
        "annotated_regions": ["Les", "il-V", "cl-V", "tV"],
    }
    (folder / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    return folder


@pytest.mark.parametrize("gz", [True, False])
def test_legacy_api_and_image_values(tmp_path: Path, gz: bool) -> None:
    folder = tmp_path / "FLASH" / "mouse" / "Anat"
    write_scan(folder, "mouse_", gz)
    datasets = RodentDatasets([0, 1])
    # Exercise the historical positional signature, including optional masks.
    datasets.add_dataset(tmp_path, "FLASH", "Anat", [3], "N4", "Labels", None, True, gz)
    scan = datasets.get_subject(("FLASH", "mouse"))
    assert scan.get_id() == scan.subject_id == scan.original_id == "mouse"
    assert scan.subject_group == ("FLASH", "mouse")
    assert scan.get_modality() == "FLASH"
    assert scan.get_sham() is True
    assert scan.get_unavailable_labels() == [3]
    assert scan.get_brain_mask() is None
    assert scan.get_images() == (scan.get_image(), scan.get_labels(), None)
    assert scan.get_path() == scan.path
    assert isinstance(scan.image.dataobj, nib.arrayproxy.ArrayProxy)
    np.testing.assert_array_equal(scan.image.get_fdata(), np.full((4, 4, 4), 2))
    np.testing.assert_allclose(scan.image.affine, np.diag([0.1, 0.2, 0.3, 1.0]))


def test_normalized_identity_timepoints_and_sham(tmp_path: Path) -> None:
    write_normalized(tmp_path, "D2", timepoints=("7days", "30days"))
    write_normalized(tmp_path, "D3", "SUBJ_0002", condition="SHAM")
    datasets = RodentDatasets()
    datasets.add_normalized_dataset(tmp_path, "D2")
    datasets.add_normalized_dataset(tmp_path, "D3", unavailable_labels=[0, 5])
    scans = [datasets.get_subject(key) for key in datasets.get_subjects_list()]
    assert [s.id for s in scans] == ["SUBJ_0001_7days", "SUBJ_0001_30days", "SUBJ_0002"]
    assert scans[0].subject_group == scans[1].subject_group == ("D2", "SUBJ_0001")
    assert scans[0].timepoint == "7days"
    assert scans[0].original_id == "original-animal"
    assert scans[0].metadata["label_configuration"] == 4
    assert scans[0].get_unavailable_labels() is None
    assert scans[2].sham and not scans[0].sham
    assert scans[2].get_unavailable_labels() == [0, 5]
    assert list(datasets.get_modalities()) == ["D2_T1w", "D3_T1w"]


def test_modality_filter_and_original_ids_do_not_merge_scans(tmp_path: Path) -> None:
    write_normalized(tmp_path, subject="SUBJ_0001")
    write_normalized(tmp_path, subject="SUBJ_0002")
    write_normalized(tmp_path, subject="SUBJ_0003", modality="T2w")
    datasets = RodentDatasets()
    datasets.add_normalized_dataset(tmp_path, "D1", modality="T1w")
    assert datasets.get_subjects_list() == [("D1_T1w", "SUBJ_0001"), ("D1_T1w", "SUBJ_0002")]
    datasets.add_normalized_dataset(tmp_path, "D1", modality="T2w")
    assert len(datasets.get_subjects_list()) == 3


def test_duplicate_registration_does_not_replace_or_partially_add(tmp_path: Path) -> None:
    write_scan(tmp_path / "first" / "FLASH" / "z", "z_")
    write_scan(tmp_path / "second" / "FLASH" / "a", "a_")
    write_scan(tmp_path / "second" / "FLASH" / "z", "z_")
    datasets = RodentDatasets()
    datasets.add_dataset(tmp_path / "first", "FLASH")
    original = datasets.get_subject(("FLASH", "z"))
    with pytest.raises(ValueError, match="Duplicate scan key"):
        datasets.add_dataset(tmp_path / "second", "FLASH")
    assert datasets.get_subjects_list() == [("FLASH", "z")]
    assert datasets.get_subject(("FLASH", "z")) is original


def test_missing_file_keeps_collection_unchanged(tmp_path: Path) -> None:
    write_normalized(tmp_path, subject="SUBJ_0001")
    folder = write_normalized(tmp_path, subject="SUBJ_0002")
    (folder / "Labels.nii.gz").unlink()
    datasets = RodentDatasets()
    with pytest.raises(FileNotFoundError, match="SUBJ_0002"):
        datasets.add_normalized_dataset(tmp_path, "D1")
    assert datasets.get_subjects_list() == []


@pytest.mark.parametrize("field,value", [
    ("normalized_id", "wrong"), ("dataset", "D4"), ("modality", None),
    ("condition", None), ("timepoints", ["../other"]), ("timepoints", "7days"),
])
def test_invalid_normalized_metadata(tmp_path: Path, field: str, value: object) -> None:
    folder = write_normalized(tmp_path)
    path = folder / "metadata.json"
    metadata = json.loads(path.read_text())
    metadata[field] = value
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="metadata.json"):
        RodentDatasets().add_normalized_dataset(tmp_path, "D1")


def test_optional_mask_custom_image_and_raw_labels(tmp_path: Path) -> None:
    folder = write_normalized(tmp_path)
    (folder / "N4.nii.gz").rename(folder / "anat_orig.nii.gz")
    (folder / "brain_mask.nii.gz").unlink()
    labels = nib.Nifti1Image(np.full((4, 4, 4), 13, dtype=np.int16), np.eye(4))
    nib.save(labels, folder / "Labels.nii.gz")
    datasets = RodentDatasets([0, 1])
    datasets.add_normalized_dataset(tmp_path, "D1", image_name="anat_orig", brain_mask_name=None)
    scan = datasets.get_subject(("D1_T1w", "SUBJ_0001"))
    assert scan.brain_mask is None
    assert np.unique(scan.labels.get_fdata()).tolist() == [13]


@pytest.mark.parametrize("normalized", [True, False])
@pytest.mark.parametrize("gz", [True, False])
def test_load_data_both_layouts(tmp_path: Path, normalized: bool, gz: bool) -> None:
    write_scan(tmp_path, "" if normalized else "mouse_", gz)
    images, arrays, paths = load_data(tmp_path, "mouse", ["N4", "Labels", "absent"], gz, normalized=normalized)
    assert set(images) == set(arrays) == set(paths) == {"N4", "Labels"}
    np.testing.assert_array_equal(arrays["N4"], np.full((4, 4, 4), 2))


def test_mixed_layout_collection_and_saved_keys(tmp_path: Path) -> None:
    write_scan(tmp_path / "FLASH" / "mouse", "mouse_")
    write_normalized(tmp_path)
    datasets = RodentDatasets()
    datasets.add_dataset(tmp_path, "FLASH")
    datasets.add_normalized_dataset(tmp_path, "D1")
    path = tmp_path / "ids.txt"
    datasets.save_ids(path)
    assert path.read_text() == "FLASH mouse\nD1_T1w SUBJ_0001\n"
    assert len(datasets.get_subjects_dic()) == 2
    with pytest.raises(ValueError, match="tuple"):
        datasets.get_subject("mouse")


@pytest.mark.parametrize("normalized", [True, False])
@pytest.mark.parametrize("domain_adaptation", [True, False])
def test_existing_generators(tmp_path: Path, normalized: bool, domain_adaptation: bool) -> None:
    """Check real generator batches without requiring training or preprocessing."""
    pytest.importorskip("tensorflow")
    from src.preprocessing.generator import DomainAdaptationGenerator, MultiTaskGenerator

    class IdentityPreprocessor:
        """Keep fixture images unchanged to isolate the loader integration."""

        def preprocess(self, image: object, masks: list) -> tuple[object, list]:
            """Return the generator's image and mask inputs unchanged."""
            return image, masks

    data = RodentDatasets([0, 1])
    if normalized:
        write_normalized(tmp_path, condition="SHAM")
        data.add_normalized_dataset(tmp_path, "D1", unavailable_labels=[1])
    else:
        write_scan(tmp_path / "FLASH" / "mouse", "mouse_")
        data.add_dataset(tmp_path, "FLASH", unavailable_labels=[1], sham=True)
    config = {
        "labels": [0, 1], "input_shape": (4, 4, 4), "dataset_path": str(tmp_path),
        "in_channels": 1, "num_classes": 2, "batch_size": 1,
    }
    generator_type = DomainAdaptationGenerator if domain_adaptation else MultiTaskGenerator
    generator = generator_type(
        data.get_subjects_list(), data, config, shuffle=False, preprocessor=IdentityPreprocessor(),
    )
    images, targets = generator[0]
    assert images.shape == targets[1].shape == (1, 4, 4, 4, 1)
    assert targets[0].shape == (1, 4, 4, 4, 2)
    assert np.all(images == 2) and np.all(targets[1] == 1)
    assert np.all(targets[0][..., 1] == (-1 if domain_adaptation else 1))
    if domain_adaptation:
        assert targets[2].tolist() == [[0.0]]  # Existing generator encodes SHAM as zero.
