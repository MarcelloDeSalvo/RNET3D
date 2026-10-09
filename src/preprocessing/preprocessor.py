"""Composable, geometry-aware preprocessing for 3D NIfTI images and masks."""

from __future__ import annotations

import abc
import json
import warnings
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, TypeAlias

import ants
import nibabel as nib
import numpy as np
import SimpleITK as sitk
from nibabel.orientations import io_orientation, ornt_transform
from nibabel.processing import resample_from_to
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter
from scipy.stats import truncnorm

NiftiImage: TypeAlias = nib.Nifti1Image | nib.Nifti2Image
ImageInput: TypeAlias = NiftiImage | list[NiftiImage] | None
PreprocessResult: TypeAlias = ImageInput | tuple[ImageInput, ImageInput]
Coordinates: TypeAlias = tuple[int, int, int]
Padding: TypeAlias = tuple[tuple[int, int], tuple[int, int], tuple[int, int]]


def _shape(values: Sequence[int]) -> Coordinates:
    """Validate a positive, three-dimensional array shape."""
    if len(values) != 3 or any(not isinstance(v, (int, np.integer)) or v <= 0 for v in values):
        raise ValueError("Shape must contain three positive integers")

    return tuple(int(v) for v in values)


def _vector(values: float | Sequence[float], name: str) -> NDArray[np.float64]:
    """Validate positive finite scalar or three-axis physical parameters."""
    vector = np.broadcast_to(np.asarray(values, dtype=float), (3,)).copy()

    if not np.all(np.isfinite(vector)) or np.any(vector <= 0):
        raise ValueError(f"{name} must contain positive finite values")

    return vector


def _range(values: Sequence[float], name: str, minimum: float | None = None) -> tuple[float, float]:
    """Validate a finite, ordered pair of sampling bounds."""
    if len(values) != 2 or not np.all(np.isfinite(values)) or values[0] > values[1]:
        raise ValueError(f"{name} must contain two finite ordered bounds")

    if minimum is not None and values[0] < minimum:
        raise ValueError(f"{name} must be at least {minimum}")

    return float(values[0]), float(values[1])


def _images(value: ImageInput) -> list[NiftiImage]:
    """Return a fresh list while leaving caller-owned lists untouched."""
    return [] if value is None else list(value) if isinstance(value, list) else [value]


def _check_image(image: NiftiImage) -> None:
    """Require a nonempty 3D image with a finite, invertible affine."""
    _shape(image.shape)

    if not np.all(np.isfinite(image.affine)) or abs(np.linalg.det(image.affine[:3, :3])) < 1e-12:
        raise ValueError("Image affine must be finite and invertible")


def _same_grid(a: NiftiImage, b: NiftiImage) -> bool:
    """Compare both array shape and physical voxel coordinates."""
    return a.shape == b.shape and np.allclose(a.affine, b.affine, rtol=0, atol=1e-6)


def _new_image(data: NDArray, reference: NiftiImage, affine: NDArray | None = None) -> NiftiImage:
    """Build an image without changing the source data or header."""
    affine = reference.affine if affine is None else affine
    header = reference.header.copy()
    image = reference.__class__(data, affine, header, dtype=data.dtype)

    # NIfTI-1 stores forms in float32. Update the copied header directly so the
    # in-memory float64 affine is not replaced by its rounded header encoding.
    sform_code: int = int(reference.header["sform_code"]) or 1
    qform_code: int = int(reference.header["qform_code"]) or 1
    image.header.set_sform(affine, code=sform_code)
    image.header.set_qform(affine, code=qform_code)

    return image


def _shift_affine(affine: NDArray, offset: Sequence[int]) -> NDArray[np.float64]:
    """Translate the voxel origin, preserving physical coordinates of retained voxels."""
    result = affine.copy()
    result[:3, 3] = nib.affines.apply_affine(affine, offset)

    return result


def _padding(shape: Sequence[int], target: Sequence[int], random: bool = False) -> Padding:
    """Compute per-axis padding widths; never shrink an image."""
    widths = []

    for size, wanted in zip(_shape(shape), _shape(target)):
        difference = max(0, wanted - size)
        before = int(np.random.randint(difference + 1)) if random else difference // 2
        widths.append((before, difference - before))

    return tuple(widths)


def _pad(image: NiftiImage, widths: Padding, mode: str) -> NiftiImage:
    """Pad data and shift its affine by the amount added before each axis."""
    source_data: NDArray = image.get_fdata()
    data = np.pad(source_data, widths, mode=mode)

    offset: list[int] = [-before for before, _ in widths]
    affine = _shift_affine(image.affine, offset)

    return _new_image(data, image, affine)


def _resample(image: NiftiImage, reference: NiftiImage, is_mask: bool, order: int = 1) -> NiftiImage:
    """Resample to an exact reference grid, using discrete interpolation for masks."""
    _check_image(image)
    _check_image(reference)

    source_data: NDArray = image.get_fdata()

    if _same_grid(image, reference):
        return _new_image(source_data, reference)

    floating_image = _new_image(source_data, image)
    target_grid = (reference.shape, reference.affine)
    interpolation_order: int = 0 if is_mask else order
    result = resample_from_to(floating_image, target_grid, order=interpolation_order)

    result_data: NDArray = result.get_fdata()

    return _new_image(result_data, reference)


def _resize_grid(image: NiftiImage, shape: Coordinates, spacing: Sequence[float], order: int) -> NiftiImage:
    """Resample onto a new grid sharing the source orientation and physical center."""
    affine = image.affine.copy()
    affine[:3, :3] *= np.asarray(spacing) / nib.affines.voxel_sizes(image.affine)

    source_center: NDArray = (np.asarray(image.shape) - 1) / 2
    center = nib.affines.apply_affine(image.affine, source_center)
    affine[:3, 3] = center - affine[:3, :3] @ ((np.asarray(shape) - 1) / 2)

    # Edge extension represents the outer half-voxel without inventing a zero border.
    source_data: NDArray = image.get_fdata()
    floating_image = _new_image(source_data, image)
    target_grid = (shape, affine)
    result = resample_from_to(floating_image, target_grid, order=order, mode="nearest")

    result_data: NDArray = result.get_fdata()

    return _new_image(result_data, image, affine)


def _lps_geometry(image: NiftiImage) -> tuple[tuple, tuple, NDArray[np.float64]]:
    """Convert NIfTI RAS geometry to ITK/ANTs LPS; reject unsupported shear."""
    _check_image(image)

    affine = np.diag([-1.0, -1.0, 1.0, 1.0]) @ image.affine
    spacing = nib.affines.voxel_sizes(affine)
    direction = affine[:3, :3] / spacing

    if not np.allclose(direction.T @ direction, np.eye(3), atol=1e-5):
        raise ValueError("ITK/ANTs transforms require an orthogonal grid; resample sheared images first")

    return tuple(affine[:3, 3]), tuple(spacing), direction


def _configuration(value: Any) -> Any:
    """Convert configuration values to JSON without exporting image voxel data."""
    if isinstance(value, (nib.Nifti1Image, nib.Nifti2Image)):
        return {"shape": list(value.shape), "affine": value.affine.tolist()}

    if isinstance(value, dict):
        return {str(k): _configuration(v) for k, v in value.items()}

    if isinstance(value, (list, tuple, np.ndarray)):
        return [_configuration(v) for v in value]

    if isinstance(value, np.generic):
        return value.item()

    return str(value) if isinstance(value, Path) else value


class PreprocessingStep(abc.ABC):
    """Apply one transform to a single image or list, optionally paired with masks."""

    spatial = False

    def __init__(self, verbose: bool = False, probability: float = 1, global_probability: bool = False) -> None:
        """
        Store execution options without retaining an input image.
        The probability parameter controls the chance of applying this step to each image/mask group.
        The global_probability flag indicates whether this step should be affected by the pipeline-wide probability setting.
        """
        self.name: str = self.__class__.__name__
        self.verbose: bool = verbose
        self.global_probability: bool = global_probability
        self.current_path: str | None = None
        self.set_probability(probability)

    @abc.abstractmethod
    def execute(self, image: NiftiImage, is_mask: bool = False, **parameters: Any) -> NiftiImage:
        """Transform one image using parameters sampled for its image/mask group."""
        raise NotImplementedError

    def sample_parameters(self, image: NiftiImage) -> dict[str, Any]:
        """Return parameters shared by every aligned image and mask."""
        return {}

    def apply(self, imgs: ImageInput = None, masks: ImageInput = None, path: str | None = None) -> PreprocessResult:
        """Validate inputs, reset the current path, and probabilistically apply the step."""
        self.current_path = str(path) if path is not None else None
        images = _images(imgs) + _images(masks)

        if not images:
            raise ValueError("Provide at least one image or mask")

        for image in images:
            _check_image(image)

        if self.spatial and any(not _same_grid(images[0], image) for image in images[1:]):
            raise ValueError(f"{self.name} requires aligned image/mask shapes and affines")

        if np.random.random() >= self.probability:
            return self.return_processed(imgs, masks)

        return self.iterate(imgs, masks)

    def iterate(self, imgs: ImageInput, masks: ImageInput) -> PreprocessResult:
        """Sample once and transform each image/mask without mutating input lists."""
        reference = (_images(imgs) + _images(masks))[0]
        parameters = self.sample_parameters(reference)

        def transform(value: ImageInput, is_mask: bool) -> ImageInput:
            """Apply shared parameters while preserving single-image/list structure."""
            if value is None:
                return None

            result = [self.execute(image, is_mask=is_mask, **parameters) for image in _images(value)]

            return result if isinstance(value, list) else result[0]

        processed_images: ImageInput = transform(imgs, False)
        processed_masks: ImageInput = transform(masks, True)

        return self.return_processed(processed_images, processed_masks)

    def iterate_list(self, input: NiftiImage | list[NiftiImage], callback: Callable, *args: Any) -> NiftiImage | list[NiftiImage]:
        """Compatibility helper: transform one image or a fresh list."""
        return [callback(image, *args) for image in input] if isinstance(input, list) else callback(input, *args)

    @staticmethod
    def return_processed(imgs: ImageInput, masks: ImageInput) -> PreprocessResult:
        """Preserve the historical single-input or image/mask-pair return format."""
        return (imgs, masks) if imgs is not None and masks is not None else masks if masks is not None else imgs

    def get_global_probability(self) -> bool:
        """Return whether pipeline-wide probability updates affect this step."""
        return self.global_probability

    def set_probability(self, probability: float) -> None:
        """Require a finite probability between zero and one."""
        if not np.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("Probability must be between zero and one")

        self.probability: float = float(probability)

    def get_configuration(self) -> str:
        """Serialize parameters, including label mappings and reference geometry."""
        parameters: dict[str, Any] = {k: v for k, v in vars(self).items() if k != "current_path"}
        configuration = _configuration(parameters)

        return json.dumps(configuration, indent=2)


class Preprocessor:
    """Compose ordered steps while preserving the existing notebook-facing API."""

    def __init__(self, steps: Sequence[PreprocessingStep], verbose: bool = False) -> None:
        """Validate and store the settings for this preprocessing step."""
        self.steps: list[PreprocessingStep] = list(steps)
        self.verbose: bool = verbose
        self.global_probability: float = 0

    def preprocess(self, imgs: ImageInput = None, masks: ImageInput = None, path: str | None = None) -> PreprocessResult:
        """Run steps, preserving single-image/list and paired-input return formats."""
        if not _images(imgs) + _images(masks):
            raise ValueError("Provide at least one image or mask")

        for step in self.steps:
            if self.verbose:
                print(step.get_configuration())

            result = step.apply(imgs, masks, path)

            if imgs is not None and masks is not None:
                imgs, masks = result
            elif imgs is not None:
                imgs = result
            else:
                masks = result

        return PreprocessingStep.return_processed(imgs, masks)

    def deprocess(
        self, prediction: NiftiImage, original_image: NiftiImage, mapping: dict[int, int],
        save_path: str | None = None, verbose: bool = False,
        reference_resolution: Sequence[float] | None = None,
        flip_axis: Sequence[int] | None = None, scale_factor: float | Sequence[float] | None = None,
    ) -> NiftiImage:
        """Restore a discrete prediction onto the original voxel grid.

        This uses affine resampling, not shape matching. An explicit flip undoes
        array augmentation on the prediction grid. CorrectX10 is matched to the
        configured pipeline (or the explicitly supplied reference resolution).
        """
        if scale_factor is not None:
            prediction = Scale(scale_factor).execute(prediction, is_mask=True)

        for axis in flip_axis or ():
            prediction = Flip([axis]).execute(prediction, is_mask=True, axis=axis)

        target = original_image

        if reference_resolution is not None:
            target = CorrectX10(reference_resolution).execute(target)
        else:
            for step in self.steps:
                if isinstance(step, CorrectX10):
                    target = step.execute(target)

        restored = _resample(prediction, target, is_mask=True)
        restored = MapLabels(mapping=mapping).execute(restored, is_mask=True)
        # CorrectX10 changes units, not voxel indices: restore the original header
        # only after resampling in the corrected coordinate system.
        restored_data: NDArray = restored.get_fdata()
        restored = _new_image(restored_data, original_image)

        if save_path is not None:
            SaveNifti(path=save_path, verbose=verbose).execute(restored, is_mask=True)

        return restored

    def get_configuration(self) -> str:
        """Return complete, parseable JSON for the ordered pipeline."""
        configuration: list[Any] = []

        for step in self.steps:
            step_json: str = step.get_configuration()
            step_configuration = json.loads(step_json)
            configuration.append(step_configuration)

        return json.dumps(configuration, indent=2)

    def save_configuration(self, path: str) -> None:
        """Write the complete pipeline configuration as JSON."""
        output_path: Path = Path(path)
        configuration: str = self.get_configuration()
        output_path.write_text(configuration, encoding="utf-8")

    def set_global_probability(self, probability: float) -> None:
        """Set the application probability for augmentation steps that opted in."""
        if not np.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("Probability must be between zero and one")

        self.global_probability = float(probability)

        for step in self.steps:
            if step.get_global_probability():
                step.set_probability(probability)


class Resample(PreprocessingStep):
    """Change voxel spacing: linear for images, nearest-neighbor for masks."""

    spatial = True

    def __init__(self, target_resolution: Sequence[float], interpolation: int = 1, verbose: bool = False) -> None:
        """Validate and store the settings for this preprocessing step."""
        super().__init__(verbose)

        self.target_resolution: NDArray[np.float64] = _vector(target_resolution, "Target resolution")

        if interpolation not in (0, 1, 2, 3, 4, 5):
            raise ValueError("Interpolation must be 0 (nearest) or 1-5 (linear to quintic). See scipy.ndimage.affine_transform for details.")

        self.interpolation: int = interpolation

    def execute(self, image: NiftiImage, is_mask: bool = False, **parameters: Any) -> NiftiImage:
        """Resample around the physical center without modifying the input."""
        _check_image(image)

        spacing = nib.affines.voxel_sizes(image.affine)

        if np.allclose(spacing, self.target_resolution, rtol=0, atol=1e-8):
            source_data: NDArray = image.get_fdata()

            return _new_image(source_data, image)

        extent = np.asarray(image.shape) * spacing / self.target_resolution

        # Header float32 rounding must not add a whole voxel on unchanged axes.
        rounded_extent: NDArray = np.rint(extent)
        near_integer: NDArray = np.isclose(extent, rounded_extent, rtol=1e-6, atol=0)
        extent = np.where(near_integer, rounded_extent, extent)

        covering_shape: NDArray = np.ceil(extent)
        shape_values: NDArray = np.maximum(1, covering_shape).astype(int)
        shape: Coordinates = tuple(shape_values)

        interpolation_order: int = 0 if is_mask else self.interpolation

        return _resize_grid(image, shape, self.target_resolution, interpolation_order)


class CorrectX10(PreprocessingStep):
    """Correct only axes whose spacing is close to ten times a known reference.

    This is a dataset-specific units heuristic, not a general resampling step.
    Omit it for data whose physical spacing has already been verified.
    """

    def __init__(self, reference_resolution: Sequence[float] = (0.1, 0.1, 0.1), threshold: float = 0.5, verbose: bool = False) -> None:
        """Validate and store the settings for this preprocessing step."""
        super().__init__(verbose)

        self.reference_resolution: NDArray[np.float64] = _vector(reference_resolution, "Reference resolution")

        if not np.isfinite(threshold) or not 0 < threshold <= 5:
            raise ValueError("Factor tolerance must be positive and at most five")

        self.threshold: float = threshold

    def execute(self, image: NiftiImage, is_mask: bool = False, **parameters: Any) -> NiftiImage:
        """Correct qualifying axes, preserving data and the physical grid center."""
        _check_image(image)

        spacing = nib.affines.voxel_sizes(image.affine)
        ratio_error: NDArray = spacing / self.reference_resolution - 10
        needs_correction: NDArray = np.abs(ratio_error) < self.threshold
        factors = np.where(needs_correction, 10, 1)

        affine = image.affine.copy()
        affine[:3, :3] /= factors
        center = (np.asarray(image.shape) - 1) / 2
        affine[:3, 3] = nib.affines.apply_affine(image.affine, center) - affine[:3, :3] @ center

        source_data: NDArray = image.get_fdata()

        return _new_image(source_data, image, affine)


class MapLabels(PreprocessingStep):
    """Map discrete label values with an explicit policy for unknown values."""

    def __init__(self, labels: Sequence[int] | None = None, mapping: dict[int, int] | None = None,
                 reverse: bool | None = None, verbose: bool = False, *, unknown: str = "error") -> None:
        """Validate and store the settings for this preprocessing step."""
        super().__init__(verbose)

        if mapping is None:
            if labels is None or len(set(labels)) != len(labels):
                raise ValueError("Supply a mapping or a list of distinct labels")

            mapping = {label: index for index, label in enumerate(labels)}

        if not mapping or any(not isinstance(value, (int, np.integer)) for pair in mapping.items() for value in pair):
            raise ValueError("Label mapping must contain integer keys and values")

        if reverse:
            if len(set(mapping.values())) != len(mapping):
                raise ValueError("Cannot reverse a many-to-one label mapping")

            mapping = {value: key for key, value in mapping.items()}

        if unknown not in ("error", "background"):
            raise ValueError("Unknown-label policy must be 'error' or 'background'")

        self.labels: list[int] | None = list(labels) if labels is not None else None
        self.mapping: dict[int, int] = dict(mapping)
        self.unknown: str = unknown

    def execute(self, image: NiftiImage, is_mask: bool = False, **parameters: Any) -> NiftiImage:
        """Map masks only; unlisted labels fail unless background mapping is explicit."""
        if not is_mask:
            return image

        data = image.get_fdata()

        if not np.all(np.isfinite(data)) or np.any(data != np.rint(data)):
            raise ValueError("Masks must contain finite integer label values")

        unknown = set(np.unique(data)) - self.mapping.keys()

        if unknown and self.unknown == "error":
            raise ValueError(f"Unmapped mask labels: {sorted(unknown)}")

        result = np.zeros_like(data)

        for source, target in self.mapping.items():
            result[data == source] = target

        return _new_image(result, image)


class Reorient(PreprocessingStep):
    """Match reference axis orientation while retaining world coordinates."""

    def __init__(self, ref_img: NiftiImage, verbose: bool = False) -> None:
        """Validate and store the settings for this preprocessing step."""
        super().__init__(verbose)

        self.ref_img: NiftiImage = ref_img

    def execute(self, image: NiftiImage, is_mask: bool = False, **parameters: Any) -> NiftiImage:
        """Permute/flip voxel axes and update the affine consistently."""
        source_orientation: NDArray = io_orientation(image.affine)
        reference_orientation: NDArray = io_orientation(self.ref_img.affine)
        transform = ornt_transform(source_orientation, reference_orientation)

        result = image.as_reoriented(transform)
        result_data: NDArray = result.get_fdata()

        return _new_image(result_data, result)


class N4BiasFieldCorrection(PreprocessingStep):
    """Apply N4 in physical LPS coordinates, leaving label masks unchanged."""

    def __init__(self, tolerance: float = 0, iters: Sequence[int] | None = None,
                 spline_parameters: float | Sequence[float] | None = None, spline_order: int = 3,
                 rescale_intensities: bool = False, masking: bool = False, verbose: bool = False) -> None:
        """Validate and store the settings for this preprocessing step."""
        super().__init__(verbose)

        self.iters: list[int] = list(iters) if iters is not None else [50, 50, 50, 50]

        self.spline_parameters: float | list[float] | None = np.asarray(spline_parameters).tolist() if spline_parameters is not None else None

        if self.spline_parameters is None:
            self.spline_parameters = [1, 1, 1]

        if not self.iters or any(not isinstance(i, (int, np.integer)) or i <= 0 for i in self.iters):
            raise ValueError("N4 iteration counts must be positive integers")

        if not np.isfinite(tolerance) or tolerance < 0:
            raise ValueError("N4 tolerance must be finite and nonnegative")

        spline = np.atleast_1d(self.spline_parameters)

        if spline.size not in (1, 3) or not np.all(np.isfinite(spline)) or np.any(spline <= 0):
            raise ValueError("N4 spline parameters must contain one or three positive values")

        if spline.size == 3 and np.any(spline != np.rint(spline)):
            raise ValueError("Three-axis N4 spline mesh sizes must be integers")

        if not isinstance(spline_order, int) or not 1 <= spline_order <= 5:
            raise ValueError("N4 spline order must be an integer between one and five")

        self.tolerance: float = tolerance
        self.spline_order: int = spline_order
        self.rescale_intensities: bool = rescale_intensities
        self.masking: bool = masking

    def execute(self, image: NiftiImage, is_mask: bool = False, **parameters: Any) -> NiftiImage:
        """Pass spacing, origin, and direction into ANTs and restore NIfTI geometry."""
        if is_mask:
            return image

        origin, spacing, direction = _lps_geometry(image)
        source_data: NDArray = image.get_fdata()
        ants_image = ants.from_numpy(source_data, origin=origin, spacing=spacing, direction=direction)

        corrected = self._correct_bias_field(ants_image)

        corrected_data: NDArray = corrected.numpy()

        return _new_image(corrected_data, image)


    def _correct_bias_field(self, image: ants.ANTsImage) -> ants.ANTsImage:
        """Run ANTs N4 using this step's settings, preserving optional masking.

        The backend call allows masking to be disabled and an explicit spline
        order to be supplied. Shrinking remains fixed at four, as before.
        """
        if image.pixeltype != "float":
            image = image.clone("float")

        mask: ants.ANTsImage | None = None
        if self.masking:
            mask = ants.get_mask(image)

        iteration_values: list[str] = [str(value) for value in self.iters]
        iteration_schedule: str = "x".join(iteration_values)
        convergence: str = f"[{iteration_schedule}, {self.tolerance:.10f}]"

        spline_values: NDArray = np.atleast_1d(self.spline_parameters)
        if spline_values.size == 1:
            spacing: float = float(spline_values[0])
            spline_parameters: str = f"[{spacing:g}]"
        else:
            mesh_values: list[str] = [str(value) for value in self.spline_parameters]
            mesh_size: str = "x".join(mesh_values)
            spline_parameters = f"[{mesh_size}, {self.spline_order}]"

        corrected: ants.ANTsImage = image.clone("float")
        bias_field: ants.ANTsImage = image.clone("float")
        corrected_pointer: str = ants.utils.get_pointer_string(corrected)
        bias_pointer: str = ants.utils.get_pointer_string(bias_field)
        output: str = f"[{corrected_pointer},{bias_pointer}]"

        rescale_flag: int = int(self.rescale_intensities)
        verbose_flag: int = int(self.verbose)
        arguments: dict[str, Any] = {
            "d": image.dimension,
            "i": image,
            "w": None,
            "s": "4",
            "c": convergence,
            "b": spline_parameters,
            "x": mask,
            "r": rescale_flag,
            "o": output,
            "v": verbose_flag,
        }

        processed_arguments: list[str] = ants.process_args._int_antsProcessArguments(arguments)
        run_n4: Callable = ants.utils.get_lib_fn("N4BiasFieldCorrection")
        run_n4(processed_arguments)

        return corrected


class Normalize(PreprocessingStep):
    """Normalize finite intensities; map constant images to zeros."""

    def __init__(self, mode: str = "zscore", verbose: bool = False) -> None:
        """Validate and store the settings for this preprocessing step."""
        super().__init__(verbose)

        if mode not in ("minmax", "zscore"):
            raise ValueError("Normalization mode must be 'minmax' or 'zscore'")

        self.mode: str = mode

    def execute(self, image: NiftiImage, is_mask: bool = False, **parameters: Any) -> NiftiImage:
        """Normalize intensities without dividing by zero or modifying masks."""
        if is_mask:
            return image

        data = image.get_fdata()

        if not np.all(np.isfinite(data)):
            raise ValueError("Cannot normalize non-finite intensities")

        offset = data.min() if self.mode == "minmax" else data.mean()
        denominator = data.max() - data.min() if self.mode == "minmax" else data.std()

        result = (data - offset) / denominator if denominator > 0 else np.zeros_like(data)

        return _new_image(result, image)


class Padder(PreprocessingStep):
    """Pad aligned images and masks using shared per-axis widths."""

    spatial = True

    def __init__(self, dim: Sequence[int], fill_mode: str = "constant", pad_mode: str = "center", verbose: bool = False) -> None:
        """Validate and store the settings for this preprocessing step."""
        super().__init__(verbose)

        self.dim: Coordinates = _shape(dim)

        if pad_mode not in ("center", "random"):
            raise ValueError("Padding mode must be 'center' or 'random'")

        if fill_mode not in ("constant", "reflect", "edge", "symmetric", "wrap"):
            raise ValueError("Unsupported padding fill mode")

        self.fill_mode: str = fill_mode
        self.pad_mode: str = pad_mode

    def generate_padding_values(self, img_shape: Sequence[int], target_shape: Sequence[int]) -> Padding:
        """Compute configured padding widths, retaining legacy helper access."""
        random_padding: bool = self.pad_mode == "random"

        return _padding(img_shape, target_shape, random=random_padding)

    def random_padding_values(self, img_shape: Sequence[int], target_shape: Sequence[int]) -> Padding:
        """Compute randomly positioned padding widths."""
        return _padding(img_shape, target_shape, random=True)

    def center_padding_values(self, img_shape: Sequence[int], target_shape: Sequence[int]) -> Padding:
        """Compute centered padding widths."""
        return _padding(img_shape, target_shape)

    def sample_parameters(self, image: NiftiImage) -> dict[str, Any]:
        """Choose one padding placement for the entire group."""
        pad_values: Padding = self.generate_padding_values(image.shape, self.dim)

        return {"pad_values": pad_values}

    def execute(self, image: NiftiImage, is_mask: bool = False, *,
                pad_values: Padding | None = None, fill_mode: str | None = None) -> NiftiImage:
        """Pad one image and update its origin; masks always receive zero padding."""
        widths = pad_values if pad_values is not None else self.generate_padding_values(image.shape, self.dim)
        padding_mode: str = "constant" if is_mask else fill_mode or self.fill_mode

        return _pad(image, widths, padding_mode)


class PadderCutter(PreprocessingStep):
    """Center-pad or center-crop to the exact requested shape."""

    spatial = True

    def __init__(self, img_size: Sequence[int], verbose: bool = False) -> None:
        """Validate and store the settings for this preprocessing step."""
        super().__init__(verbose)

        self.img_size: Coordinates = _shape(img_size)

    def execute(self, image: NiftiImage, is_mask: bool = False, **parameters: Any) -> NiftiImage:
        """Share padding logic and retain world coordinates of the cropped voxels."""
        widths: Padding = _padding(image.shape, self.img_size)
        padded = _pad(image, widths, "constant")

        starts = [(size - wanted) // 2 for size, wanted in zip(padded.shape, self.img_size)]
        slices = tuple(slice(start, start + size) for start, size in zip(starts, self.img_size))

        cropped_data: NDArray = padded.get_fdata()[slices]
        affine: NDArray = _shift_affine(padded.affine, starts)

        return _new_image(cropped_data, padded, affine)


class SaveNifti(PreprocessingStep):
    """Save a single image, optionally resampling onto a reference grid."""

    def __init__(self, path: str | None = None, postfix: str | None = None,
                 reference: NiftiImage | None = None, verbose: bool = False, replace: str | None = None) -> None:
        """Validate and store the settings for this preprocessing step."""
        super().__init__(verbose)

        self.path: str | None = str(path) if path is not None else None
        self.postfix: str | None = postfix
        self.reference: NiftiImage | None = reference
        self.replace: str | None = replace

    def iterate(self, imgs: ImageInput, masks: ImageInput) -> PreprocessResult:
        """Prevent multiple outputs from silently overwriting the same filename."""
        if len(_images(imgs) + _images(masks)) != 1:
            raise ValueError("SaveNifti saves one image at a time")

        return super().iterate(imgs, masks)

    def execute(self, image: NiftiImage, is_mask: bool = False, **parameters: Any) -> NiftiImage:
        """Save to an explicit path or a derived filename supporting .nii and .nii.gz."""
        path = self.path or self.current_path

        if path is None:
            raise ValueError("Provide an output path or the current input path")

        output = Path(path)

        if self.postfix is not None:
            if self.replace is not None:
                if self.replace not in output.name:
                    raise ValueError(f"Filename does not contain replacement text: {self.replace}")

                output_name: str = output.name.replace(self.replace, self.postfix)
                output = output.with_name(output_name)
            else:
                extension = ".nii.gz" if output.name.endswith(".nii.gz") else ".nii" if output.name.endswith(".nii") else None

                if extension is None:
                    raise ValueError("Expected a .nii or .nii.gz filename")

                output_name = output.name[:-len(extension)] + self.postfix + extension
                output = output.with_name(output_name)

        if self.path is None and output == Path(self.current_path):
            raise ValueError("Derived output would overwrite the input; supply an explicit output path")

        if self.reference is not None:
            result = _resample(image, self.reference, is_mask)
        else:
            source_data: NDArray = image.get_fdata()
            result = _new_image(source_data, image)

        output.parent.mkdir(parents=True, exist_ok=True)
        output_path: str = str(output)
        nib.save(result, output_path)

        if self.verbose:
            print(f"Image saved to {output}")

        return result


class Scale(PreprocessingStep):
    """Change sampling density while preserving physical extent and center."""

    spatial = True

    def __init__(self, scale_factor: float | Sequence[float], interpolation: int = 1,
                 verbose: bool = False) -> None:
        """Validate and store the settings for this preprocessing step."""
        super().__init__(verbose)

        if interpolation not in (0, 1, 2, 3, 4, 5):
            raise ValueError("Interpolation must be 0 (nearest) or 1-5 (linear to quintic). See scipy.ndimage.affine_transform for details.")

        self.scale_factor: NDArray[np.float64] = _vector(scale_factor, "Scale factor")
        self.interpolation: int = interpolation

    def execute(self, image: NiftiImage, is_mask: bool = False, **parameters: Any) -> NiftiImage:
        """Resample images linearly and masks discretely onto the scaled grid."""
        scaled_shape: NDArray = np.asarray(image.shape) * self.scale_factor
        rounded_shape: NDArray = np.rint(scaled_shape)
        shape = np.maximum(1, rounded_shape).astype(int)

        spacing = nib.affines.voxel_sizes(image.affine) * np.asarray(image.shape) / shape
        target_shape: Coordinates = tuple(shape)

        interpolation_order: int = 0 if is_mask else self.interpolation

        return _resize_grid(image, target_shape, spacing, interpolation_order)


class Noise(PreprocessingStep):
    """Add Gaussian intensity noise; the range is in image intensity units."""

    def __init__(self, noise_range: Sequence[float], probability: float = 0.3, mean: float = 0) -> None:
        """Validate and store the settings for this preprocessing step."""
        super().__init__(probability=probability, global_probability=True)

        self.noise_range: tuple[float, float] = _range(noise_range, "Noise standard deviation", minimum=0)

        if not np.isfinite(mean):
            raise ValueError("Noise mean must be finite")

        self.mean: float = mean

    def execute(self, image: NiftiImage, is_mask: bool = False, **parameters: Any) -> NiftiImage:
        """Add sampled noise to intensities only."""
        if is_mask:
            return image

        std = np.random.uniform(*self.noise_range)

        source_data: NDArray = image.get_fdata()
        noise: NDArray = np.random.normal(self.mean, std, image.shape)
        noisy_data: NDArray = source_data + noise

        return _new_image(noisy_data, image)


class GaussianBlur(PreprocessingStep):
    """Smooth intensities; sigma is measured in voxels on each axis."""

    def __init__(self, gaussian_blur_range: Sequence[float], probability: float = 0.3) -> None:
        """Validate and store the settings for this preprocessing step."""
        super().__init__(probability=probability, global_probability=True)

        self.gaussian_blur_range: tuple[float, float] = _range(gaussian_blur_range, "Gaussian sigma", minimum=0)

    def execute(self, image: NiftiImage, is_mask: bool = False, **parameters: Any) -> NiftiImage:
        """Apply independently sampled per-axis smoothing to images only."""
        if is_mask:
            return image

        sigma = np.random.uniform(*self.gaussian_blur_range, size=3)

        source_data: NDArray = image.get_fdata()
        blurred_data: NDArray = gaussian_filter(source_data, sigma)

        return _new_image(blurred_data, image)


class RandomAffine(PreprocessingStep):
    """Rotate in degrees and scale about the physical image center on a fixed grid."""

    spatial = True

    def __init__(self, scale_range: Sequence[float] | None = None, rotation_range: Sequence[float] | None = None,
                 probability: float = 0.5, interpolation: int = sitk.sitkLinear, background_label: int = 0) -> None:
        """Validate and store the settings for this preprocessing step."""
        super().__init__(probability=probability, global_probability=True)

        self.scale_range: tuple[float, float] | None = _range(scale_range, "Scale range", minimum=np.finfo(float).eps) if scale_range is not None else None
        self.rotation_range: tuple[float, float] | None = _range(rotation_range, "Rotation range") if rotation_range is not None else None

        if interpolation not in (sitk.sitkLinear, sitk.sitkNearestNeighbor):
            raise ValueError("Use sitkLinear or sitkNearestNeighbor interpolation")

        self.interpolation: int = interpolation
        self.background_label: int = background_label

    def sample_parameters(self, image: NiftiImage) -> dict[str, Any]:
        """Sample one physical transform for all images and masks."""
        angles: NDArray[np.float64]
        if self.rotation_range is not None:
            angles = np.random.uniform(*self.rotation_range, size=3)
        else:
            angles = np.zeros(3)

        scale_factor: float
        if self.scale_range is not None:
            sampled_scale = np.random.uniform(*self.scale_range)
            scale_factor = float(sampled_scale)
        else:
            scale_factor = 1.0

        return {"angles": angles, "scale_factor": scale_factor}

    def execute(self, image: NiftiImage, is_mask: bool = False, *,
                angles: Sequence[float] | None = None, scale_factor: float | None = None) -> NiftiImage:
        """Transform in physical LPS coordinates using the input as output reference."""
        if angles is None:
            sampled = self.sample_parameters(image)
            angles = sampled["angles"]
            scale_factor = sampled["scale_factor"] if scale_factor is None else scale_factor

        if len(angles) != 3 or not np.all(np.isfinite(angles)):
            raise ValueError("Supply three finite rotation angles")

        factor = 1.0 if scale_factor is None else scale_factor

        if not np.isfinite(factor) or factor <= 0:
            raise ValueError("Scale factor must be positive and finite")

        origin, spacing, direction = _lps_geometry(image)
        source_data: NDArray = image.get_fdata().transpose(2, 1, 0)
        itk_image = sitk.GetImageFromArray(source_data)
        itk_image.SetOrigin(origin)
        itk_image.SetSpacing(spacing)

        flattened_direction: NDArray = direction.ravel()
        itk_direction: tuple = tuple(flattened_direction)
        itk_image.SetDirection(itk_direction)

        center_indices: NDArray = (np.asarray(image.shape) - 1) / 2
        center_index: tuple = tuple(center_indices)
        center = itk_image.TransformContinuousIndexToPhysicalPoint(center_index)

        angles_radians: NDArray = np.radians(angles)
        rotation = sitk.Euler3DTransform(center, *angles_radians)

        scale_factors: list[float] = [factor] * 3
        scale = sitk.ScaleTransform(3, scale_factors)
        scale.SetCenter(center)

        transforms: list[sitk.Transform] = [rotation, scale]
        transform = sitk.CompositeTransform(transforms)

        interpolation: int = sitk.sitkNearestNeighbor if is_mask else self.interpolation
        background_value: float = float(self.background_label) if is_mask else 0.0
        result = sitk.Resample(
            itk_image, itk_image, transform,
            interpolation, background_value, sitk.sitkFloat64,
        )

        result_data: NDArray = sitk.GetArrayFromImage(result).transpose(2, 1, 0)

        return _new_image(result_data, image)


class Flip(PreprocessingStep):
    """Reflect anatomy within a fixed grid for augmentation, not reorientation."""

    spatial = True

    def __init__(self, axis_list: Sequence[int], probability: float = 0.5, verbose: bool = False) -> None:
        """Validate and store the settings for this preprocessing step."""
        super().__init__(verbose, probability, global_probability=True)

        if len(axis_list) == 0 or any(not isinstance(axis, (int, np.integer)) or axis not in (0, 1, 2) for axis in axis_list):
            raise ValueError("Flip axes must be chosen from 0, 1, 2")

        self.axis_list: list[int] = list(axis_list)

    def sample_parameters(self, image: NiftiImage) -> dict[str, Any]:
        """Choose a shared augmentation axis."""
        sampled_axis = np.random.choice(self.axis_list)
        axis: int = int(sampled_axis)

        return {"axis": axis}

    def execute(self, image: NiftiImage, is_mask: bool = False, *, axis: int | None = None) -> NiftiImage:
        """Flip voxel values on the unchanged grid; repeating the same flip undoes it."""
        axis = int(np.random.choice(self.axis_list)) if axis is None else axis

        if axis not in (0, 1, 2):
            raise ValueError("Flip axis must be 0, 1, or 2")

        source_data: NDArray = image.get_fdata()
        flipped_data: NDArray = np.flip(source_data, axis=axis)

        return _new_image(flipped_data, image)


class RandomCropping(PreprocessingStep):
    """Sample exact-sized crops, optionally padding small images first."""

    spatial = True

    def __init__(self, dim: Sequence[int], mode: str = "random", std: float | None = None,
                 verbose: bool = False, padding: bool = False) -> None:
        """Validate and store the settings for this preprocessing step."""
        super().__init__(verbose)

        self.dim: Coordinates = _shape(dim)

        if mode not in ("random", "center"):
            raise ValueError("Crop mode must be 'random' or 'center'")

        if std is not None and (not np.isfinite(std) or std <= 0):
            raise ValueError("Cropping standard deviation must be positive and finite")

        self.mode: str = mode
        self.std: float | None = std
        self.pad_if_needed: bool = padding

    def _coordinates(self, shape: Sequence[int], mode: str) -> Coordinates:
        """Sample centers including the last valid crop, for odd and even sizes."""
        coordinates = []

        for size, patch in zip(shape, self.dim):
            maximum = size - patch

            if maximum < 0:
                raise ValueError("Crop exceeds image size; enable padding or add Padder")

            if maximum == 0:
                start = 0
            elif mode == "random":
                start = int(np.random.randint(maximum + 1))
            else:
                mean, std = maximum / 2, self.std or size / 4
                lower_bound: float = -mean / std
                upper_bound: float = (maximum - mean) / std
                distribution = truncnorm(lower_bound, upper_bound, loc=mean, scale=std)
                sampled_start: float = distribution.rvs()
                rounded_start = np.rint(sampled_start)
                start = int(rounded_start)

            coordinates.append(start + patch // 2)

        return tuple(coordinates)

    def get_random_cropping_coordinates(self, img: NiftiImage) -> Coordinates:
        """Return uniformly sampled crop centers on the supplied grid."""
        return self._coordinates(img.shape, "random")

    def get_center_cropping_coordinates(self, img: NiftiImage) -> Coordinates:
        """Return center-biased crop centers on the supplied grid."""
        return self._coordinates(img.shape, "center")

    def select_cropping_coordinates(self, img: NiftiImage) -> Coordinates:
        """Sample crop centers using the configured distribution."""
        return self._coordinates(img.shape, self.mode)

    def sample_parameters(self, image: NiftiImage) -> dict[str, Any]:
        """Choose shared padding and crop coordinates without allocating an image."""
        widths: Padding
        if self.pad_if_needed:
            widths = _padding(image.shape, self.dim)
        else:
            widths = ((0, 0),) * 3

        shape = tuple(size + before + after for size, (before, after) in zip(image.shape, widths))
        center: Coordinates = self._coordinates(shape, self.mode)

        return {"center": center, "pad_values": widths}

    def padding(self, img: NiftiImage) -> NiftiImage:
        """Pad to at least the crop size using the shared padding implementation."""
        widths: Padding = _padding(img.shape, self.dim)

        return _pad(img, widths, "constant")

    def execute(self, image: NiftiImage, is_mask: bool = False, *,
                center: Coordinates | None = None, pad_values: Padding | None = None) -> NiftiImage:
        """Extract the exact requested shape and update the voxel origin."""
        if center is None:
            parameters = self.sample_parameters(image)
            center, pad_values = parameters["center"], parameters["pad_values"]

        image = _pad(image, pad_values, "constant") if pad_values is not None and any(a or b for a, b in pad_values) else image

        starts = [coordinate - size // 2 for coordinate, size in zip(center, self.dim)]

        if len(starts) != 3 or any(start < 0 or start + size > available for start, size, available in zip(starts, self.dim, image.shape)):
            raise ValueError("Crop coordinates are outside the image")

        slices = tuple(slice(start, start + size) for start, size in zip(starts, self.dim))
        cropped_data: NDArray = image.get_fdata()[slices]
        affine: NDArray = _shift_affine(image.affine, starts)

        return _new_image(cropped_data, image, affine)
