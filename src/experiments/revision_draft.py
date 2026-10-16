"""Tiny, explicitly non-scientific copies of real scans for suite smoke checks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to

from .data import PreparedData


def balanced_sample(data: PreparedData, count: int, seed: int) -> list[str]:
    """Select animals across annotation strata, retaining all their timepoints."""
    groups = data.annotation_groups(include_lesion=False)
    rng = np.random.default_rng(seed)
    buckets: list[list[str]] = []
    for group in sorted(set(groups.values())):
        members = sorted(animal for animal, value in groups.items() if value == group)
        buckets.append(rng.permutation(members).tolist())
    selected: list[str] = []
    while len(selected) < min(count, len(groups)):
        for bucket in buckets:
            if bucket and len(selected) < count:
                selected.append(bucket.pop())
    return sorted(selected)


def _corrected_affine(image: nib.Nifti1Image, resolution: Sequence[float], correct: bool) -> np.ndarray:
    """Use the production CorrectX10 geometry heuristic without changing voxels."""
    affine: np.ndarray = image.affine.copy()
    if correct:
        spacing = nib.affines.voxel_sizes(affine)
        reference = np.asarray(resolution)
        qualifying = np.abs(spacing / reference - 10) < 0.5
        factors = np.ones(3)
        factors[qualifying] = 10
        center = (np.asarray(image.shape) - 1) / 2
        affine[:3, :3] /= factors
        affine[:3, 3] = nib.affines.apply_affine(image.affine, center) - affine[:3, :3] @ center
    return affine


def write_draft_dataset(
    datasets: Sequence[PreparedData], output_dir: Path, *, shape: tuple[int, int, int],
    spacing_mm: tuple[float, float, float], reference: Path,
) -> Path:
    """Resample small central fields into a new normalized tree, leaving inputs intact.

    The third-axis spacing is 1.1 times the requested training spacing so the
    interpolation stage still exercises resampling. MRI uses linear interpolation;
    both annotation masks use nearest neighbour. Source IDs and availability stay
    unchanged. These cropped volumes cannot support biological interpretation.
    """
    output_dir.mkdir(parents=True, exist_ok=False)
    ref = nib.load(str(reference))
    directions = ref.affine[:3, :3] / nib.affines.voxel_sizes(ref.affine)
    draft_spacing = np.asarray(spacing_mm) * np.array([1.0, 1.0, 1.1])
    records: list[dict[str, object]] = []
    seen: set[tuple[str, str]] = set()
    for data in datasets:
        for key in data.loader.get_subjects_list():
            scan = data.loader.get_subject(key)
            dataset, subject = scan.subject_group
            identity = (dataset, scan.id)
            if identity in seen:
                continue
            seen.add(identity)
            subject_dir = output_dir / dataset / subject
            subject_dir.mkdir(parents=True, exist_ok=True)
            metadata_path = subject_dir / "metadata.json"
            metadata_path.write_text(json.dumps(scan.metadata, indent=2), encoding="utf-8")
            folder = subject_dir
            if scan.timepoint:
                folder = folder / scan.timepoint
            folder.mkdir(parents=True, exist_ok=True)

            source_affine = _corrected_affine(
                scan.image, spacing_mm, data.config.preprocessing.correct_units,
            )
            source_center = (np.asarray(scan.image.shape) - 1) / 2
            center_world = nib.affines.apply_affine(source_affine, source_center)
            affine = np.eye(4)
            affine[:3, :3] = directions * draft_spacing
            affine[:3, 3] = center_world - affine[:3, :3] @ ((np.asarray(shape) - 1) / 2)
            target = (shape, affine)
            sources = (
                (data.config.data.image_name, scan.image, scan.image_path, 1),
                (data.config.data.roi_name, scan.labels, scan.labels_path, 0),
                (data.config.data.brain_name, scan.brain_mask, scan.brain_mask_path, 0),
            )
            for name, image, source_path, order in sources:
                if image is None:
                    raise ValueError(f"Draft checks require a brain mask: {scan.id}")
                corrected = _corrected_affine(image, spacing_mm, data.config.preprocessing.correct_units)
                source = nib.Nifti1Image(np.asarray(image.dataobj), corrected)
                sampled = resample_from_to(source, target, order=order)
                dtype = np.float32
                if order == 0:
                    dtype = np.int16
                values = np.asarray(sampled.dataobj, dtype=dtype)
                result = nib.Nifti1Image(values, affine)
                result.header.set_xyzt_units("mm")
                target_path = folder / f"{name}.nii.gz"
                nib.save(result, str(target_path))
                path = Path(source_path)
                records.append({
                    "Source": str(path.resolve()), "Bytes": path.stat().st_size,
                    "ModifiedNs": path.stat().st_mtime_ns, "Draft": str(target_path.resolve()),
                })
    description = {
        "DRAFT": True, "shape": shape, "spacing_mm": draft_spacing.tolist(),
        "purpose": "Execution checks only; central crops and tiny networks are not scientific results",
        "sources": records,
    }
    (output_dir / "draft.json").write_text(json.dumps(description, indent=2), encoding="utf-8")
    return output_dir


def validate_draft_sources(folder: Path) -> None:
    """Reject resumption if source scans changed since the draft was copied."""
    record = json.loads((folder / "draft.json").read_text(encoding="utf-8"))
    for source in record["sources"]:
        path = Path(source["Source"])
        if (path.stat().st_size, path.stat().st_mtime_ns) != (source["Bytes"], source["ModifiedNs"]):
            raise ValueError(f"Draft source changed: {path}")
        if not Path(source["Draft"]).is_file():
            raise FileNotFoundError(source["Draft"])
