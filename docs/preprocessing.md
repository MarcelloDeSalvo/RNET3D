# Preprocessing behavior

The preprocessing steps retain their class-based API and the existing
single-image/list and `(images, masks)` return formats. Shared input handling,
padding calculations, geometry construction, and configuration export now have
one implementation each.

## Interpolation

Intensity images use **linear interpolation** by default in `Resample`, `Scale`,
`RandomAffine`, and reference-grid saving. Segmentation and brain masks always
use **nearest-neighbor interpolation**, preserving discrete class IDs.

The seven explicit `Resample(..., interpolation=0)` settings in the three
existing notebooks were changed to `interpolation=1`. No other notebook logic
was changed, and the notebooks were not executed.

`Resample` and `Scale` use SciPy orders (`0` or `1`). `RandomAffine` retains its
SimpleITK enum argument (`sitkNearestNeighbor` or `sitkLinear`). These numeric
conventions are different; using the defaults is recommended. The legacy
`Scale(..., interpolation=...)` spelling still works with a deprecation warning.

## Geometry and transformations

- Resampling preserves physical center and orientation and constructs an affine
  for the actual output shape. Reference-grid resampling targets the specified
  shape and affine. Input images and headers are not modified.
- Padding and cropping update the voxel origin so retained voxels keep their
  physical coordinates. Odd and even crop dimensions are supported. The last
  valid crop placement can be sampled.
- `RandomCropping(..., padding=True)` uses the shared padding implementation.
  Without it, undersized inputs raise a clear error. `Padder` supports masks
  without an accompanying intensity image. `PadderCutter` now centers both
  padding and cropping.
- A spatial step samples its random parameters once and applies them to the
  entire aligned image/mask group. Mismatched shapes or affines raise an error
  rather than silently applying voxel coordinates to incompatible grids.
- `RandomAffine` rotates about the physical image center in ITK LPS coordinates;
  angles are degrees and scale factors are dimensionless. Spacing, origin,
  direction, and array-axis ordering are passed explicitly to SimpleITK.
- `Flip` is explicitly an **anatomical augmentation on a fixed grid**: it flips
  values while retaining the grid affine. `Reorient` instead changes array axes
  and the affine together, preserving anatomy in physical space.
- `Scale` changes sampling density, not anatomical size: it adjusts spacing
  alongside output shape. Use `RandomAffine` for anatomical scaling on a fixed
  output grid.
- N4 receives the correct LPS spacing, origin, and direction. Sheared grids are
  rejected by ITK/ANTs conversion rather than treated as orthogonal. Gaussian
  blur sigma remains in voxel units; noise magnitude remains in intensity units.

## Restoring and saving predictions

`Preprocessor.deprocess` resamples a discrete prediction onto the original
reference grid, then applies the requested label mapping. It no longer tries to
restore alignment through padding or cutting alone. Explicit `flip_axis` values
undo deterministic array flips before restoration.

If the pipeline includes `CorrectX10`, restoration uses that same correction
to match coordinate systems, then reinstates the original header/grid after
the voxel data has been aligned. Without that step, there is no implicit
spacing correction. Supplying `reference_resolution` explicitly retains the
legacy option to request a correction during restoration.

`SaveNifti(reference=...)` resamples onto the reference grid and uses its header.
Pass `is_mask=True` to direct `execute` calls when saving label predictions.
Pipeline calls via `masks=...` set this automatically.

Derived filenames support both `.nii` and `.nii.gz`. Input paths are reset for
each call, missing replacement text is an error, and a derived filename cannot
overwrite its input. An explicit output path may intentionally overwrite an
existing file. A save step accepts one image at a time so multiple masks cannot
silently overwrite each other.

## Explicit policies

**N4 organization:** `N4BiasFieldCorrection` owns both NIfTI/ANTs geometry
conversion and the private `_correct_bias_field` backend call. The standalone
`bias/N4bias_wrapped.py` script uses this same class. The former
`n4_bias.py` module and `ants_n4_bias_field_wrapper` function have been removed;
call `N4BiasFieldCorrection(...).execute(nifti_image)` instead. The class retains
its existing masking, intensity-rescaling, convergence and spline settings.

**Labels:** `MapLabels` rejects fractional/nonfinite masks and unlisted label
values by default. For a deliberate reduction to a smaller task, request
`unknown="background"` explicitly. This option is not partial-label masking.
Many-to-one mappings are supported in the forward direction; attempting to
reverse one raises an error. No training-loss policy was changed.

**Normalization:** constant intensity images become zeros in either normalization
mode. Unknown modes and nonfinite intensities raise errors. Masks pass through.

**Spacing correction:** `CorrectX10` now corrects only qualifying axes, with
an absolute ratio tolerance of `0.5` around ten (previously `5`). It remains a
dataset-specific heuristic: a legitimate spacing near ten times the reference
cannot be distinguished from a units error using the affine alone. Omit this
step when physical spacing has already been verified. It preserves the grid
center and does not change voxel data.

**Configuration:** probabilities, dimensions, ranges, axes, and N4 settings are
validated. N4 defaults are independent between instances. Configuration export
is now JSON and contains complete label mappings and reference geometry.

## Compatibility and validation

Constructor and pipeline calls used by the existing notebooks are retained.
Direct transform calls now consistently start with `execute(image, is_mask=False)`;
shared geometry parameters are keyword-only. In particular:

```python
padder.execute(image, pad_values=widths)
flip.execute(image, axis=0)
cropper.execute(image, center=(40, 40, 40))
```

`preprocessor` can be imported through either `src.preprocessing` from the
repository root or `preprocessing` when `src` is on the Python path.

Run the regression suite in the existing `rnet` environment:

```text
python -m pytest tests/test_preprocessor.py tests/test_loader.py -q
```

The current combined suite has 65 passing cases. Tests cover numerical failures,
discrete labels, analytic physical-coordinate
checks, reference-grid restoration, file handling, metadata conversion, and
legacy/normalized loader compatibility. A one-iteration N4 smoke run and small
real mouse/rat patch checks were also performed. All 234 normalized scans have
aligned image, ROI-mask, and brain-mask headers under the new grid checks.

These changes alter preprocessing behavior; historical metrics have not been
recomputed. No model training or interpolation sensitivity analysis was run.
