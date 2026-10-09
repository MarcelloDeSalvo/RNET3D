"""Binary morphology and ventricle-side assignment in voxel coordinates.

Callbacks retain their historical names, parameters and label conventions.
Hemisphere assignment assumes inputs have already been reoriented consistently;
array coordinates alone cannot establish anatomical left/right.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
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

    def extract_masks(y_roi: Mask) -> tuple[Mask, Mask, Mask]:
        """Extract binary masks using the existing input label convention."""
        lesion: Mask = (y_roi == 1).astype(np.uint8)
        ventricles: Mask = (y_roi == 2).astype(np.uint8)
        third_ventricle: Mask = (y_roi == 3).astype(np.uint8)
        return lesion, ventricles, third_ventricle

    def get_regions(mask: Mask) -> tuple[list[RegionProperties], Mask]:
        """Label components with the historical full-connectivity default."""
        labeled_mask: Mask = label(mask)
        regions: list[RegionProperties] = regionprops(labeled_mask)
        return regions, labeled_mask

    def perform_pca(coords: Mask) -> tuple[FloatArray, FloatArray]:
        """Fit three voxel-space axes without changing the original PCA settings."""
        pca: PCA = PCA(n_components=3)
        pca.fit(coords)
        return pca.components_, pca.mean_

    def symmetry_cost(normal: FloatArray, coords: Mask, center_of_mass: FloatArray) -> float:
        """Measure the absolute voxel-count imbalance across the proposed plane."""
        plane_normal: FloatArray = np.asarray(normal, dtype=float)
        length: float = float(np.linalg.norm(plane_normal))
        if not np.isfinite(length) or length == 0:
            return float('inf')

        plane_normal = plane_normal / length
        distances: FloatArray = np.dot(coords - center_of_mass, plane_normal)
        first_half: int = int(np.count_nonzero(distances >= 0))
        second_half: int = int(np.count_nonzero(distances < 0))
        return float(abs(first_half - second_half))

    def get_side(centroid: Sequence[float], plane_point: FloatArray, normal: FloatArray) -> bool:
        """Return whether a point is strictly on the normal-facing plane side."""
        vector: FloatArray = np.asarray(centroid) - plane_point
        return bool(np.dot(vector, normal) > 0)

    def create_ventricle_masks_with_centroids(
        labeled_ventricles: Mask, regions: Sequence[RegionProperties], hemisphere_indices: Sequence[bool],
    ) -> tuple[Mask, Mask]:
        """Assign each connected component using its centroid's hemisphere."""
        ipsi: Mask = np.zeros_like(labeled_ventricles)
        contra: Mask = np.zeros_like(labeled_ventricles)
        for region, is_ipsilateral in zip(regions, hemisphere_indices):
            if is_ipsilateral:
                ipsi[labeled_ventricles == region.label] = 1
            else:
                contra[labeled_ventricles == region.label] = 1
        return ipsi, contra

    def create_ventricle_masks_with_hard_split(
        ventricles_mask: Mask, optimal_normal: FloatArray,
        lesion_side: bool, plane_point: FloatArray,
    ) -> tuple[Mask, Mask]:
        """Assign individual voxels, keeping the original strict plane-side rule."""
        coordinates: Mask = np.argwhere(ventricles_mask > 0)
        ipsi: Mask = np.zeros_like(ventricles_mask)
        contra: Mask = np.zeros_like(ventricles_mask)
        for coordinate in coordinates:
            same_side: bool = get_side(coordinate, plane_point, optimal_normal)
            voxel: tuple[int, ...] = tuple(coordinate)
            if same_side == lesion_side:
                ipsi[voxel] = 1
            else:
                contra[voxel] = 1
        return ipsi, contra

    def update_final_roi_mask(y_roi: Mask, ipsi: Mask, contra: Mask) -> Mask:
        """Copy the input before relabeling the third and lateral ventricles."""
        final_mask: Mask = np.copy(y_roi)
        final_mask[final_mask == 3] = 4
        final_mask[contra == 1] = 2
        final_mask[ipsi == 1] = 3
        return final_mask

    def visualize_pca_axes(coords: Mask, principal_axes: FloatArray, plane_point: FloatArray) -> None:
        """Display at most 20,000 voxel coordinates and the fitted PCA axes."""
        figure = plt.figure()
        axes = figure.add_subplot(111, projection='3d')
        sampled_coords: Mask = coords
        if len(coords) > 20000:
            indices: Mask = np.random.choice(len(coords), size=20000, replace=False)
            sampled_coords = coords[indices]

        axes.scatter(sampled_coords[:, 0], sampled_coords[:, 1], sampled_coords[:, 2],
                     alpha=0.05, c='blue', s=1)
        scale_factor: int = 50
        for index, component in enumerate(principal_axes):
            start: FloatArray = plane_point - scale_factor * component
            stop: FloatArray = plane_point + scale_factor * component
            line_x: list[float] = [start[0], stop[0]]
            line_y: list[float] = [start[1], stop[1]]
            line_z: list[float] = [start[2], stop[2]]
            legend_label: str = f'Principal Component {index + 1}'
            axes.plot(line_x, line_y, line_z, label=legend_label, linewidth=2)

        axes.legend()
        axes.set_title('Principal Components of the third ventricle')
        plt.show()
        plt.close(figure)

    def post_process(y_roi: Mask) -> Mask:
        """Split valid 3D input labels; reject insufficient plane information."""
        if y_roi.ndim != 3 or not np.isin(y_roi, [0, 1, 2, 3]).all():
            raise ValueError('Expected a 3D ROI mask with input labels 0, 1, 2, 3')

        lesion_mask, ventricles_mask, third_ventricle_mask = extract_masks(y_roi)
        if not ventricles_mask.any():
            empty_mask: Mask = np.zeros_like(y_roi)
            return update_final_roi_mask(y_roi, empty_mask, empty_mask)

        third_ventricle_coords: Mask = np.argwhere(third_ventricle_mask > 0)
        if len(third_ventricle_coords) < 3:
            raise ValueError('At least three third-ventricle voxels are required to split ventricles')

        lesion_regions, _ = get_regions(lesion_mask)
        ventricle_regions, labeled_ventricles = get_regions(ventricles_mask)
        third_ventricle_regions, _ = get_regions(third_ventricle_mask)
        third_ventricle_region = max(third_ventricle_regions, key=lambda region: region.area)
        third_ventricle_centroid: FloatArray = np.asarray(third_ventricle_region.centroid)
        principal_axes, _ = perform_pca(third_ventricle_coords)

        if visualize_pca:
            visualize_pca_axes(third_ventricle_coords, principal_axes, third_ventricle_centroid)

        initial_normal: FloatArray = principal_axes[2]
        optimization_args: tuple[Mask, FloatArray] = (third_ventricle_coords, third_ventricle_centroid)
        result: OptimizeResult = minimize(symmetry_cost, initial_normal, args=optimization_args, method='Nelder-Mead')
        optimal_normal: FloatArray = result.x
        normal_length: float = float(np.linalg.norm(optimal_normal))
        if not np.isfinite(normal_length) or normal_length == 0:
            raise ValueError('Symmetry optimization returned an invalid plane normal')
        optimal_normal = optimal_normal / normal_length

        lesion_region: RegionProperties | None = None
        if lesion_regions:
            largest_lesion = max(lesion_regions, key=lambda region: region.area)
            if largest_lesion.area >= lesion_voxels_threshold:
                lesion_region = largest_lesion

        if lesion_region is None:
            print(f'No lesion above threshold, defaulting to the {default_hemisphere} hemisphere with hard split')
            upper_right_point: FloatArray = np.zeros(3)
            lesion_side: bool = get_side(upper_right_point, third_ventricle_centroid, optimal_normal)
            if default_hemisphere == 'left':
                lesion_side = not lesion_side
            ipsi, contra = create_ventricle_masks_with_hard_split(
                ventricles_mask, optimal_normal, lesion_side, third_ventricle_centroid,
            )
        else:
            lesion_side = get_side(lesion_region.centroid, third_ventricle_centroid, optimal_normal)
            if use_centroids:
                hemisphere_indices: list[bool] = []
                for region in ventricle_regions:
                    region_side: bool = get_side(region.centroid, third_ventricle_centroid, optimal_normal)
                    hemisphere_indices.append(region_side == lesion_side)
                ipsi, contra = create_ventricle_masks_with_centroids(
                    labeled_ventricles, ventricle_regions, hemisphere_indices,
                )
            else:
                ipsi, contra = create_ventricle_masks_with_hard_split(
                    ventricles_mask, optimal_normal, lesion_side, third_ventricle_centroid,
                )

        return update_final_roi_mask(y_roi, ipsi, contra)

    return post_process
