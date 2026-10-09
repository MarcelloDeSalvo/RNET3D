"""Binary morphology and ventricle-side assignment in voxel coordinates.

Callbacks retain their historical names, parameters and label conventions.
Hemisphere assignment assumes inputs have already been reoriented consistently;
array coordinates alone cannot establish anatomical left/right.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, TypeAlias

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import OptimizeResult, minimize
from skimage import morphology
from skimage.measure import label, regionprops
from sklearn.decomposition import PCA

if TYPE_CHECKING:
    from skimage.measure._regionprops import RegionProperties

Mask: TypeAlias = NDArray[Any]
FloatArray: TypeAlias = NDArray[np.float64]
PostprocessCallback: TypeAlias = Callable[[Mask], Mask]


def morphology_refinement_callback(
    fill_small_holes: bool = True,
    remove_small_objects: bool = True,
    holes_max_area: int = 20000,
    object_min_area: int = 30000,
) -> PostprocessCallback:
    """Fill small holes, then remove small objects in a binary foreground mask.

    Thresholds are voxel counts; library-default connectivity is preserved.
    This does not guarantee a single connected component. Enabled operations
    treat nonzero values as foreground and return float64. With both operations
    disabled, the original array and dtype are returned unchanged.
    """
    for name, value in (("holes_max_area", holes_max_area), ("object_min_area", object_min_area)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer voxel count")

    def post_process(y_pred: Mask) -> Mask:
        """Apply the selected morphology operations without mutating the input."""
        if not remove_small_objects and not fill_small_holes:
            return y_pred

        binary_mask: NDArray[np.bool_] = y_pred.astype(bool)
        if fill_small_holes:
            binary_mask = morphology.remove_small_holes(binary_mask, area_threshold=holes_max_area)

        if remove_small_objects:
            binary_mask = morphology.remove_small_objects(binary_mask, min_size=object_min_area)

        return binary_mask.astype(np.float64)

    return post_process


def ipsi_contra_division_callback(
    visualize_pca: bool = False,
    use_centroids: bool = False,
    default_hemisphere: Literal['right', 'left'] = 'right',
    lesion_voxels_threshold: int = 10,
) -> PostprocessCallback:
    """Split ventricular label 2 relative to the largest valid lesion component.

    Input labels: 0 background, 1 lesion, 2 ventricles, 3 third ventricle.
    Output labels: 0 background, 1 lesion, 2 contralateral ventricles,
    3 ipsilateral ventricles, 4 third ventricle. Input dtype is preserved.

    PCA uses all third-ventricle voxels, with the largest component centroid as
    plane origin, preserving the original algorithm. At least three voxels are
    required when ventricles need splitting. Missing/undersized lesions use a
    hard split even when use_centroids=True. The historical 'right' convention
    means the plane side containing array origin (0, 0, 0); 'left' selects the
    opposite side. Reorient inputs beforehand; no affine is available here.
    """
    if default_hemisphere not in ('right', 'left'):
        raise ValueError("default_hemisphere must be 'right' or 'left'")
    if (isinstance(lesion_voxels_threshold, bool)
            or not isinstance(lesion_voxels_threshold, (int, np.integer))
            or lesion_voxels_threshold < 0):
        raise ValueError('lesion_voxels_threshold must be a nonnegative integer')

    def post_process(y_roi: Mask) -> Mask:
        """Copy and relabel the ROI using the fitted plane and selected lesion side."""
        if y_roi.ndim != 3 or not np.isin(y_roi, [0, 1, 2, 3]).all():
            raise ValueError('Expected a 3D ROI mask with input labels 0, 1, 2, 3')

        final_mask: Mask = y_roi.copy()
        third_ventricle: Mask = y_roi == 3
        ventricles: Mask = y_roi == 2
        final_mask[third_ventricle] = 4
        if not ventricles.any():
            return final_mask

        plane: _SymmetryPlane = _SymmetryPlane.fit(third_ventricle, visualize_pca)
        lesion_mask: Mask = y_roi == 1
        lesion = _largest_region(lesion_mask)
        has_valid_lesion: bool = lesion is not None and lesion.area >= lesion_voxels_threshold

        if lesion is not None and has_valid_lesion:
            lesion_side: bool = bool(plane.side(lesion.centroid))
        else:
            print(f'No lesion above threshold, defaulting to the {default_hemisphere} hemisphere with hard split')
            origin: FloatArray = np.zeros(3)
            lesion_side = bool(plane.side(origin))
            if default_hemisphere == 'left':
                lesion_side = not lesion_side

        if use_centroids and has_valid_lesion:
            components: Mask = label(ventricles)
            for region in regionprops(components):
                if plane.side(region.centroid) == lesion_side:
                    final_mask[components == region.label] = 3
        else:
            coordinates: Mask = np.argwhere(ventricles)
            ipsilateral: NDArray[np.bool_] = plane.side(coordinates) == lesion_side
            ipsi_coordinates: Mask = coordinates[ipsilateral]
            voxel_indices: tuple[Mask, ...] = tuple(ipsi_coordinates.T)
            final_mask[voxel_indices] = 3

        return final_mask

    return post_process


def _largest_region(mask: Mask) -> RegionProperties | None:
    """Return the largest fully connected component, or None for empty masks."""
    components: Mask = label(mask)
    regions: list[RegionProperties] = regionprops(components)
    if not regions:
        return None
    return max(regions, key=lambda region: region.area)


@dataclass(frozen=True)
class _SymmetryPlane:
    """A fitted voxel-space plane with a shared point-side convention."""

    point: FloatArray
    normal: FloatArray

    def side(self, points: Sequence[float] | Mask) -> np.bool_ | NDArray[np.bool_]:
        """Classify one point or many; points on the plane remain on the false side."""
        offsets: FloatArray = np.asarray(points) - self.point
        distances: FloatArray | np.float64 = np.dot(offsets, self.normal)
        return distances > 0

    @classmethod
    def fit(cls, mask: Mask, visualize: bool = False) -> _SymmetryPlane:
        """Preserve PCA over all voxels and the largest-component plane origin."""
        coordinates: Mask = np.argwhere(mask)
        if len(coordinates) < 3:
            raise ValueError('At least three third-ventricle voxels are required to split ventricles')

        region = _largest_region(mask)
        assert region is not None  # At least three foreground voxels were checked above.
        point: FloatArray = np.asarray(region.centroid)
        pca: PCA = PCA(n_components=3)
        pca.fit(coordinates)
        principal_axes: FloatArray = pca.components_
        if visualize:
            _visualize_pca_axes(coordinates, principal_axes, point)

        initial_normal: FloatArray = principal_axes[2]
        optimization_args: tuple[Mask, FloatArray] = (coordinates, point)
        result: OptimizeResult = minimize(
            _symmetry_cost, initial_normal, args=optimization_args, method='Nelder-Mead',
        )
        normal: FloatArray = result.x
        length: float = float(np.linalg.norm(normal))
        if not np.isfinite(length) or length == 0:
            raise ValueError('Symmetry optimization returned an invalid plane normal')
        normal = normal / length
        return cls(point, normal)


def _symmetry_cost(normal: FloatArray, coordinates: Mask, point: FloatArray) -> float:
    """Keep the original absolute voxel-count imbalance objective."""
    length: float = float(np.linalg.norm(normal))
    if not np.isfinite(length) or length == 0:
        return float('inf')

    unit_normal: FloatArray = normal / length
    distances: FloatArray = np.dot(coordinates - point, unit_normal)
    first_half: int = int(np.count_nonzero(distances >= 0))
    second_half: int = int(np.count_nonzero(distances < 0))
    return float(abs(first_half - second_half))


def _visualize_pca_axes(coordinates: Mask, principal_axes: FloatArray, point: FloatArray) -> None:
    """Display up to 20,000 voxel coordinates and the three fitted PCA axes."""
    figure = plt.figure()
    axes = figure.add_subplot(111, projection='3d')
    sampled: Mask = coordinates
    if len(coordinates) > 20000:
        indices: Mask = np.random.choice(len(coordinates), size=20000, replace=False)
        sampled = coordinates[indices]

    axes.scatter(sampled[:, 0], sampled[:, 1], sampled[:, 2], alpha=0.05, c='blue', s=1)
    scale: int = 50
    for index, component in enumerate(principal_axes):
        start: FloatArray = point - scale * component
        stop: FloatArray = point + scale * component
        line_x: list[float] = [start[0], stop[0]]
        line_y: list[float] = [start[1], stop[1]]
        line_z: list[float] = [start[2], stop[2]]
        legend_label: str = f'Principal Component {index + 1}'
        axes.plot(line_x, line_y, line_z, label=legend_label, linewidth=2)

    axes.legend()
    axes.set_title('Principal Components of the third ventricle')
    plt.show()
    plt.close(figure)
