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
conventions are different; using the defaults is recommended.
`Scale(..., interpolation=...)` is the regular image-interpolation argument.

## Geometry and transformations

MRI notebooks use `example/RARE/TBI_fm_19_49/Anat/TBI_fm_19_49_N4.nii.gz`
as the common reorientation reference. Its axis codes are **R, I, A**. This
means LR is axis 0, IS is axis 1, and AP is axis 2. The reference determines
axis ordering/signs, not output voxel size or shape. Spatial parameters are
sampled once per aligned MRI/ROI/brain-mask group. Masks use nearest-neighbour
interpolation even when MRI uses linear interpolation.

`Flip(anatomical_axis="LR")` resolves LR from each input affine, avoiding fixed
voxel-axis assumptions when the reference changes. `Flip(axis_list=[...])`
remains available for an explicitly selected array axis. Reorientation preserves
physical anatomy; augmentation reflects voxel values on a fixed grid.

Flipping is off by default. D1 training/study notebooks opt into LR flipping at
0.5; rat fine-tuning/CV retain 0.5 and MouseFull uses 0.0. Validation and inference never flip
randomly. The historical D1 notebook's validation and inference flips were
removed as well; its training flip remains aligned with the RIA reference.

Ipsi/contra labels in these datasets are relative to the lesion or craniotomy.
Reflecting MRI and masks together also reflects that side, so the class IDs
remain unchanged. Swapping them would reverse their meaning. Class permutation
is a different policy for labels defined as absolute anatomical left/right.
Unavailable-label channels likewise retain their current identities because
relative class IDs do not change during this augmentation.

For a future unmerged ipsi/contra model, an explicit surgery-side signal would
avoid relying on lesion visibility to resolve symmetric anatomy in SHAM. One
option is a normalized LR coordinate channel whose sign is relative to the
known lesion/craniotomy hemisphere, generated on the full scan before cropping
and transformed with the MRI/masks. This is a proposed new input variant; it is
not added to the current one-channel architecture. It would require the actual
surgical hemisphere to be verified, and new training. The existing D1 merged
vocabulary is retained for the established experiments.

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
- N4 receives the correct LPS spacing, origin, and direction, retaining
  nonorthogonal coefficients in the in-memory ANTs image. Gaussian
  blur defaults to voxel sigma and noise to raw input intensity units.
  Optional millimetre blur and image-relative noise scales are explicit.

## Acquisition-style intensity augmentation

The shared training factory reorients/corrects units first, then applies blur and
noise on the native grid before target resampling. Physical affine augmentation
follows on the target grid with its original directions; normalization, cropping and optional LR
flipping come afterward. MRI interpolation remains linear and mask interpolation
nearest-neighbour. Blur and noise leave both masks and all grid headers unchanged.
This order lets interpolation mix an injected noise field, unlike adding fresh
independent noise after resampling. It approximates acquisition variability;
Gaussian noise does not reproduce a complete modality-specific acquisition model.

`GaussianBlur(..., sigma_units="voxels")` samples sigma separately for each axis
in native voxel units. `sigma_units="mm"` divides sampled values by the affine's
voxel sizes before calling the Gaussian filter. `Noise(..., std_units="intensity")`
uses the sampled standard deviation directly; `std_units="image_std"` multiplies
it by the full input MRI standard deviation, including background. The latter
is a per-image scale, not an SNR guarantee or a statistic fitted across subjects.
The noise mean stays in intensity units; constant images with zero mean receive
zero added noise in image-relative mode. Existing positional constructor calls
retain the original unit conventions.

Training notebooks expose ranges and units in `PreprocessingConfig`. Defaults
remain blur sigma 0-0.6 voxels and noise standard deviation 0-0.05 intensity units.
The native grid changes their physical/interpolated effect, so old runs retain
their saved `intensity_augmentation_grid="resampled"` sequence when loaded.
Missing grid fields in older saved configurations resolve to this prior order.
No interpolation sensitivity analysis or spatial-transform fusion is performed.

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

**Spacing correction:** `CorrectX10` uses an absolute ratio tolerance of `0.5`
around ten. When at least two axes match this factor and every axis is at least
`10 - tolerance` times the target spacing, it divides the complete grid's spacing
by ten. Thus an inflated anisotropic T2w scan with `(1, 1, 3)` mm becomes
`(0.1, 0.1, 0.3)` mm, preserving its three-to-one slice anisotropy. Otherwise
only individually qualifying axes are corrected, retaining mixed-axis handling.
Already correct `(0.1, 0.1, 0.3)` mm scans pass through. The grid center, shear,
voxel values and MRI/mask alignment are preserved; this step does not resample.
`corrected_affine()` exposes the same header-only calculation to draft generation.

This remains a dataset-specific heuristic: a legitimate spacing near ten times
the reference cannot be distinguished from a units error using the affine
alone. Omit this step when physical spacing has already been verified.
Saved evaluation-patch contracts record the correction version. Start a new
run/study after changing this policy rather than mixing old crop coordinates
and checkpoints with the corrected geometry. Original NIfTI files are unchanged.

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

## Visual step checks

`notebook/check_preprocessing.ipynb` selects one normalized animal/timepoint and
shows each operation in a short before/after cell. Reorientation to the shared
RARE reference runs first, then units correction and target-spacing resampling.
Separate examples cover normalization, label mapping, padding/cropping, grid
scaling, physical rotation/scaling, blur, noise and anatomical LR flipping.
Blur/noise examples start from the corrected native volume with explicit units,
while spatial examples use the normalized target volume. The custom combined
pipeline demonstrates native blur/noise -> resampling -> affine -> normalization
and cropping. Its relative noise strengths are illustrative, not production defaults.
N4 is optional for inputs that are already corrected. The final cell contains
an editable combined pipeline; masks retain original IDs unless label mapping
is explicitly being demonstrated. Inspection flips do not change production
augmentation settings.

`utils/inspection.py` groups MRI and masks in `InspectionVolume`, applies shared
transforms, and shows aligned native Plotly overlays with numeric summaries.
Seeded previews restore the global NumPy RNG state after processing. Panels
use relative slice positions within their own grids, rather than pretending to
compare a common world-space slice after cropping/resampling.
