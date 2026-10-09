"""Load legacy and normalized rodent scans through one generator-compatible API."""

from __future__ import annotations

import json
from collections.abc import KeysView, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
from nibabel.spatialimages import SpatialImage
from numpy.typing import NDArray

PathLike = str | Path
SubjectKey = tuple[str, str]


def _image_path(folder: Path, name: str, gz: bool) -> Path:
    """Build a NIfTI path using the requested compression convention."""
    return folder / f"{name}{'.nii.gz' if gz else '.nii'}"


@dataclass(eq=False)
class Rodent:
    """One scan, its NIfTI images, and optional subject-level metadata.

    The id identifies a scan; subject_id groups longitudinal scans. Images
    are opened with nibabel's array proxies, without reading all voxel data.
    The original constructor arguments and getters remain supported.
    """

    modality: str
    id: str
    image_path: PathLike
    labels_path: PathLike
    brain_mask_path: PathLike | None = None
    unavailable_labels: Sequence[int] | None = None
    sham: bool = False
    subject_id: str | None = field(default=None, kw_only=True)
    timepoint: str | None = field(default=None, kw_only=True)
    metadata: dict[str, Any] = field(default_factory=dict, kw_only=True)
    image: SpatialImage = field(init=False, repr=False)
    labels: SpatialImage = field(init=False, repr=False)
    brain_mask: SpatialImage | None = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Open images and copy caller-owned metadata and label lists."""
        self.subject_id = self.subject_id or self.id
        self.metadata = dict(self.metadata)
        if self.unavailable_labels is not None:
            self.unavailable_labels = list(self.unavailable_labels)
        self.image = self.load_nifti(self.image_path)
        self.labels = self.load_nifti(self.labels_path)
        self.brain_mask = self.load_nifti(self.brain_mask_path) if self.brain_mask_path else None

    @property
    def path(self) -> str:
        """Return the image path under its historical attribute name."""
        return str(self.image_path)

    @property
    def original_id(self) -> str:
        """Return the original subject ID, or the legacy scan ID if unmapped."""
        return self.metadata.get("original_id", self.id)

    @property
    def subject_group(self) -> SubjectKey:
        """Return a dataset-scoped subject key shared by its timepoints.

        Normalized IDs are used as supplied; ambiguous original IDs are not
        merged automatically. Legacy data falls back to modality and scan ID.
        """
        return self.metadata.get("dataset", self.modality), self.subject_id or self.id

    def load_nifti(self, file_path: PathLike) -> SpatialImage:
        """Open a NIfTI image, including the scan ID in missing-file errors."""
        try:
            return nib.load(str(file_path))
        except FileNotFoundError as error:
            raise FileNotFoundError(f"File not found for {self.id}: {file_path}") from error

    def __str__(self) -> str:
        """Return a readable modality and scan identifier."""
        return f"{self.modality} - {self.id}"

    def get_images(self) -> tuple[SpatialImage, SpatialImage, SpatialImage | None]:
        """Return image, ROI labels, and optional brain mask in generator order."""
        return self.image, self.labels, self.brain_mask

    def get_modality(self) -> str:
        """Return the acquisition modality."""
        return self.modality

    def get_image(self) -> SpatialImage:
        """Return the input image."""
        return self.image

    def get_labels(self) -> SpatialImage:
        """Return the ROI image with its original label values."""
        return self.labels

    def get_brain_mask(self) -> SpatialImage | None:
        """Return the brain mask, when requested during loading."""
        return self.brain_mask

    def get_unavailable_labels(self) -> Sequence[int] | None:
        """Return explicitly configured unavailable source-mask labels."""
        return self.unavailable_labels

    def get_sham(self) -> bool:
        """Return whether the scan belongs to the SHAM condition."""
        return self.sham

    def get_path(self) -> str:
        """Return the input image filename."""
        return self.path

    def get_id(self) -> str:
        """Return the scan ID, including its timepoint when longitudinal."""
        return self.id


class RodentDatasets:
    """A collection of scans indexed by (group, scan_id) tuples.

    Legacy groups retain their modality names. Normalized groups use
    dataset_modality to keep different datasets with the same modality
    separate. Both layouts expose the same records to existing generators.
    """

    def __init__(self, labels: Sequence[int] | None = None) -> None:
        """Store the configured label vocabulary without remapping images."""
        self.labels = list(labels) if labels is not None else None
        self.datasets_dic: dict[str, dict[str, Rodent]] = {}

    def add_dataset(
        self,
        dataset_path: PathLike,
        modality: str,
        sub_folder: str | None = None,
        unavailable_labels: Sequence[int] | None = None,
        image_suffix: str = "N4",
        roi_mask_suffix: str = "Labels",
        brain_mask_suffix: str | None = "brain_mask",
        sham: bool = False,
        gz: bool = True,
    ) -> None:
        """Add root/modality/id/[sub_folder]/id_<suffix>.nii[.gz] scans.

        All historical positional arguments are supported. Set
        brain_mask_suffix=None to load scans without brain masks.
        Existing keys raise an error instead of silently replacing scans.
        """
        scans = []
        for subject in sorted((Path(dataset_path) / modality).iterdir()):
            if not subject.is_dir():
                continue
            folder = subject / sub_folder if sub_folder else subject
            scans.append((modality, Rodent(
                modality, subject.name,
                _image_path(folder, f"{subject.name}_{image_suffix}", gz),
                _image_path(folder, f"{subject.name}_{roi_mask_suffix}", gz),
                _image_path(folder, f"{subject.name}_{brain_mask_suffix}", gz) if brain_mask_suffix else None,
                unavailable_labels, sham,
            )))
        self._register(scans)

    def add_normalized_dataset(
        self,
        dataset_path: PathLike,
        dataset: str,
        *,
        modality: str | None = None,
        image_name: str = "N4",
        roi_mask_name: str = "Labels",
        brain_mask_name: str | None = "brain_mask",
        unavailable_labels: Sequence[int] | None = None,
        gz: bool = True,
    ) -> None:
        """Add root/dataset/subject/[timepoint]/<name>.nii[.gz] scans.

        Subject metadata.json provides IDs, modality, condition, and
        timepoints. Modality optionally selects one acquisition modality.
        Metadata (including annotation availability) is retained;
        label remapping and missing-label policy are not inferred here.
        Unavailable_labels explicitly applies to every selected scan.
        """
        scans = []
        for subject in sorted((Path(dataset_path) / dataset).iterdir()):
            if not subject.is_dir():
                continue
            metadata_path = subject / "metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8-sig"))
            if not isinstance(metadata, dict):
                raise ValueError(f"Expected a metadata object: {metadata_path}")
            if metadata.get("normalized_id") != subject.name or metadata.get("dataset") != dataset:
                raise ValueError(f"Dataset/subject identity does not match folder: {metadata_path}")
            scan_modality = metadata.get("modality")
            if not isinstance(scan_modality, str) or not scan_modality:
                raise ValueError(f"Missing acquisition modality: {metadata_path}")
            if modality is not None and scan_modality != modality:
                continue
            if (
                not isinstance(metadata.get("original_id"), str)
                or not metadata["original_id"]
                or metadata.get("condition") not in ("TBI", "SHAM")
            ):
                raise ValueError(f"Missing original ID or unsupported condition: {metadata_path}")
            timepoints = metadata.get("timepoints", [])
            if not isinstance(timepoints, list) or any(
                not isinstance(t, str) or not t or t in {".", ".."} or any(c in t for c in "/\\:")
                for t in timepoints
            ):
                raise ValueError(f"Invalid timepoint names: {metadata_path}")
            for timepoint in timepoints or [None]:
                folder = subject / timepoint if timepoint else subject
                scan_id = f"{subject.name}_{timepoint}" if timepoint else subject.name
                scans.append((f"{dataset}_{scan_modality}", Rodent(
                    scan_modality, scan_id,
                    _image_path(folder, image_name, gz),
                    _image_path(folder, roi_mask_name, gz),
                    _image_path(folder, brain_mask_name, gz) if brain_mask_name else None,
                    unavailable_labels, metadata["condition"] == "SHAM",
                    subject_id=subject.name, timepoint=timepoint, metadata=metadata,
                )))
        self._register(scans)

    def _register(self, scans: list[tuple[str, Rodent]]) -> None:
        """Validate all keys before adding any scans to the collection."""
        keys = set(self.get_subjects_list())
        for group, scan in scans:
            key = group, scan.id
            if key in keys:
                raise ValueError(f"Duplicate scan key: {key}")
            keys.add(key)
        for group, scan in scans:
            self.datasets_dic.setdefault(group, {})[scan.id] = scan

    def get_subject(self, key: SubjectKey) -> Rodent:
        """Look up a scan using a key from get_subjects_list."""
        if not isinstance(key, tuple) or len(key) != 2:
            raise ValueError("Key must be a tuple with the group and scan id")
        return self.datasets_dic[key[0]][key[1]]

    def get_subjects_list(self) -> list[SubjectKey]:
        """Return scan keys in dataset-registration and sorted-folder order."""
        return [(group, scan_id) for group, scans in self.datasets_dic.items() for scan_id in scans]

    def get_subjects_dic(self) -> dict[str, dict[str, Rodent]]:
        """Return the existing group-to-scans mapping for legacy callers."""
        return self.datasets_dic

    def get_modalities(self) -> KeysView[str]:
        """Return collection group names (historically called modalities)."""
        return self.datasets_dic.keys()

    def save_ids(self, path: PathLike) -> None:
        """Write one 'group scan_id' per line in the historical text format."""
        Path(path).write_text(
            "".join(f"{group} {scan_id}\n" for group, scan_id in self.get_subjects_list()),
            encoding="utf-8",
        )


def load_data(
    patient_folder: PathLike,
    patient_id: str,
    modalities: Sequence[str],
    gz: bool = True,
    *,
    normalized: bool = False,
) -> tuple[dict[str, SpatialImage], dict[str, NDArray[np.float64]], dict[str, str]]:
    """Load named images from one scan folder for notebook inspection.

    Legacy filenames are <patient_id>_<modality>.nii[.gz]; normalized
    filenames omit the ID prefix. For longitudinal data pass the timepoint
    folder. Missing files are printed and skipped, preserving legacy behavior.
    """
    images, data, file_paths = {}, {}, {}
    for modality in modalities:
        name = modality if normalized else f"{patient_id}_{modality}"
        path = _image_path(Path(patient_folder), name, gz)
        if path.is_file():
            image = nib.load(str(path))
            images[modality] = image
            data[modality] = image.get_fdata()
            file_paths[modality] = str(path)
        else:
            print(f"File not found for modality {modality}")
    return images, data, file_paths









