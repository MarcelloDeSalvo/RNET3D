"""Small volume bundles and native Plotly displays for diagnostic notebooks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.colors import qualitative
from plotly.subplots import make_subplots

from ..preprocessing.preprocessor import NiftiImage, PreprocessingStep, Preprocessor
from .loader import Rodent, RodentDatasets


@dataclass(frozen=True)
class InspectionVolume:
    """MRI with categorical ROI and optional brain masks on the same grid."""

    image: NiftiImage
    labels: NiftiImage
    brain_mask: NiftiImage | None = None

    def __post_init__(self) -> None:
        """Reject misaligned overlays instead of displaying them as aligned."""
        for mask in (self.labels, self.brain_mask):
            if mask is None:
                continue
            if self.image.ndim != 3 or mask.shape != self.image.shape:
                raise ValueError("MRI and masks must share a 3D grid")
            if not np.allclose(mask.affine, self.image.affine):
                raise ValueError("MRI and masks must share an affine")

    def preprocess(self, steps: Sequence[PreprocessingStep], *, seed: int = 42) -> InspectionVolume:
        """Apply shared transforms reproducibly without changing global RNG state."""
        masks = [self.labels]
        if self.brain_mask is not None:
            masks.append(self.brain_mask)

        pipeline = Preprocessor(steps)
        random_state = np.random.get_state()
        np.random.seed(seed)
        try:
            image, processed_masks = pipeline.preprocess(self.image, masks)
        finally:
            np.random.set_state(random_state)

        brain_mask = None
        if self.brain_mask is not None:
            brain_mask = processed_masks[1]
        return InspectionVolume(image, processed_masks[0], brain_mask)

    def with_masks(self, labels: np.ndarray, brain_mask: np.ndarray | None = None) -> InspectionVolume:
        """Wrap postprocessed arrays on this grid, retaining MRI and binary brain IDs."""
        affine = self.image.affine
        label_image = nib.Nifti1Image(labels, affine)
        brain_image = self.brain_mask
        if brain_mask is not None:
            brain_image = nib.Nifti1Image(brain_mask, affine)
        return InspectionVolume(self.image, label_image, brain_image)


def sample_normalized_scan(root: Path, dataset: str = "D1", seed: int = 42) -> Rodent:
    """Choose one animal uniformly, then one of its available timepoints."""
    collection = RodentDatasets()
    collection.add_normalized_dataset(root, dataset, image_name="N4")
    animals: dict[tuple[str, str], list[Rodent]] = {}
    for key in collection.get_subjects_list():
        scan = collection.get_subject(key)
        animals.setdefault(scan.subject_group, []).append(scan)
    if not animals:
        raise ValueError(f"No scans found in {dataset}")

    rng = np.random.default_rng(seed)
    subjects = sorted(animals)
    subject_index = int(rng.integers(len(subjects)))
    scans = animals[subjects[subject_index]]
    scan_index = int(rng.integers(len(scans)))
    return scans[scan_index]


def postprocessing_example() -> InspectionVolume:
    """Make a 0.1 mm example with known ventricle sides, a brain hole and an island."""
    shape = (64, 64, 64)
    coordinates = np.indices(shape)
    squared_distance = np.sum((coordinates - 32) ** 2, axis=0)
    brain = squared_distance < 23 ** 2
    intensity = brain * (1 - squared_distance / (2 * 23 ** 2))
    image_data = intensity.astype(np.float32)
    labels = np.zeros(shape, dtype=np.int16)
    labels[31:34, 24:41, 24:41] = 21
    labels[18:23, 26:39, 26:39] = 13
    labels[42:47, 26:39, 26:39] = 3
    labels[13:18, 30:35, 30:35] = 1
    brain[30:35, 30:35, 30:35] = False
    brain[3:6, 30:35, 30:35] = True
    brain_data = brain.astype(np.int16)
    affine = np.diag([0.1, 0.1, 0.1, 1.0])
    image = nib.Nifti1Image(image_data, affine)
    roi = nib.Nifti1Image(labels, affine)
    brain_mask = nib.Nifti1Image(brain_data, affine)
    return InspectionVolume(image, roi, brain_mask)


def show_volumes(
    volumes: Mapping[str, InspectionVolume], *, axis: int = 2,
    position: float = 0.5, renderer: str = "vscode",
) -> None:
    """Show MRI/ROI/brain overlays and basic statistics for one or two volumes.

    Each panel uses the same relative slice position in its own grid, rather
    than claiming a common world-space slice after cropping or resampling.
    ROI hover values remain categorical IDs; cyan outlines the binary brain.
    Native Plotly supports zoom and hover without widgets or HTML exports.
    """
    from IPython.display import display

    if axis not in (0, 1, 2) or not np.isfinite(position) or not 0 <= position <= 1:
        raise ValueError("Choose axis 0/1/2 and a relative position between zero and one")
    if not 1 <= len(volumes) <= 2:
        raise ValueError("Show one volume or a before/after pair")

    names = list(volumes)
    label_count = 32
    palette = qualitative.Alphabet
    colorscale = []
    for label_id in range(label_count):
        color = palette[label_id % len(palette)]
        lower = label_id / label_count
        upper = (label_id + 1) / label_count
        colorscale.extend([(lower, color), (upper, color)])

    figure = make_subplots(rows=1, cols=len(volumes), subplot_titles=names)
    rows = []
    for column, (name, volume) in enumerate(volumes.items(), start=1):
        index = int(round(position * (volume.image.shape[axis] - 1)))
        values = volume.image.get_fdata(dtype=np.float32)
        labels = volume.labels.get_fdata()
        if not np.isfinite(values).all() or not np.isfinite(labels).all():
            raise ValueError("Preview data must contain finite values")
        if np.any(labels != np.rint(labels)) or np.any(labels < 0) or np.any(labels >= label_count):
            raise ValueError("Preview ROI labels must be integer IDs between 0 and 31")

        pixels = np.take(values, index, axis=axis).T
        roi = np.take(labels, index, axis=axis).T
        overlay = np.where(roi > 0, roi, np.nan)
        low, high = np.percentile(pixels, [1, 99])
        high = max(high, low + 1e-6)
        spacing = nib.affines.voxel_sizes(volume.image.affine)
        horizontal, vertical = [dimension for dimension in range(3) if dimension != axis]
        x = np.arange(pixels.shape[1]) * spacing[horizontal]
        y = np.arange(pixels.shape[0]) * spacing[vertical]
        image_trace = go.Heatmap(x=x, y=y, z=pixels, colorscale="Gray", zmin=low, zmax=high, showscale=False)
        roi_trace = go.Heatmap(
            x=x, y=y, z=overlay, colorscale=colorscale, zmin=-0.5, zmax=label_count - 0.5,
            opacity=0.5, showscale=False, hoverongaps=False,
            hovertemplate="ROI ID=%{z:.0f}<extra></extra>",
        )
        figure.add_trace(image_trace, row=1, col=column)
        figure.add_trace(roi_trace, row=1, col=column)
        brain_voxels = None
        if volume.brain_mask is not None:
            brain_values = volume.brain_mask.get_fdata()
            if not np.isfinite(brain_values).all():
                raise ValueError("Brain masks must contain finite values")
            brain = brain_values > 0
            brain_slice = np.take(brain, index, axis=axis).T
            contour = go.Contour(
                x=x, y=y, z=brain_slice.astype(int), showscale=False, hoverinfo="skip",
                contours={"start": 0.5, "end": 0.5, "size": 1, "coloring": "none"},
                line={"color": "cyan", "width": 1.5},
            )
            figure.add_trace(contour, row=1, col=column)
            brain_voxels = int(brain.sum())
        anchor = "x"
        if column > 1:
            anchor = f"x{column}"
        orientation = nib.aff2axcodes(volume.image.affine)
        horizontal_title = f"Axis {horizontal}: {orientation[horizontal]} (mm)"
        vertical_title = f"Axis {vertical}: {orientation[vertical]} (mm)"
        figure.update_xaxes(title_text=horizontal_title, row=1, col=column)
        figure.update_yaxes(title_text=vertical_title, scaleanchor=anchor, row=1, col=column)
        rows.append({
            "Step": name, "Shape": volume.image.shape, "Spacing (mm)": np.round(spacing, 4),
            "Axes": nib.aff2axcodes(volume.image.affine), "Mean": float(values.mean()),
            "Std": float(values.std()), "ROI IDs": np.unique(labels).astype(int).tolist(),
            "Brain voxels": brain_voxels,
        })

    figure.update_layout(height=450, template="plotly_white", title_text=f"Axis {axis}, position {position:.0%}")
    figure.show(renderer=renderer)
    table = pd.DataFrame(rows).set_index("Step")
    display(table)
