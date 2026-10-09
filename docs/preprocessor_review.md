# Preprocessor review

Scope: `src/preprocessing/preprocessor.py`, 2026-10-08.

This document records the findings from the initial annotation-only pass.
The subsequent user-authorized fix pass addressed the eight confirmed defects
and the additional concerns below. See [current preprocessing behavior](preprocessing.md)
for implemented policies, compatibility changes, and validation. The descriptions
and verification results below refer to the earlier version, before those fixes.

## Applied cleanup

- Annotated all 58 functions, including nested coordinate-sampling helpers.
- Annotated constructor attributes and the main array, geometry, configuration,
  and scalar intermediate variables. Ordinary loop indices and unpacked values
  remain inferred rather than adding separate declarations everywhere.
- Added five small aliases for NIfTI images, optional single/list inputs,
  processing results, crop coordinates, and padding widths.
- Added missing function docstrings and corrected documentation describing
  padding, scalar cropping deviation, and the actual nearest-neighbor default
  in `RandomAffine`.
- Wrapped long signatures and formatted annotated default arguments.

The annotations describe the existing interface; they do not introduce runtime
validation. In particular, they do not conceal the incompatible subclass
method signatures or the `RandomCropping.padding` name collision.

## Confirmed defects from the initial review (subsequently fixed)

### 1. Constant images produce NaNs during normalization

`Normalize.execute` divides by zero for a constant image in both `minmax` and
`zscore` modes. A synthetic all-ones image produced only NaNs in both cases.

Proposed correction: define a constant-image policy and guard the denominator.
Unknown normalization modes currently print an error and return unchanged data;
raising a clear exception would also make configuration errors visible.

### 2. Odd crop dimensions produce smaller patches

`RandomCropping.execute` slices from `center - dim // 2` to `center + dim // 2`.
A requested `(5, 5, 5)` crop produced `(4, 4, 4)`. This can break generator batch
assignment. Random coordinate selection also excludes the last valid center
because the upper bound of `randint` is exclusive.

Proposed correction: calculate the start once and use `start + dim` as the end,
with coordinate bounds defined consistently for odd and even sizes.

### 3. Cropping's padding option is broken

Three separate problems exist:

- `self.padding` is a boolean that shadows the method with the same name.
  Calling `cropper.padding(image)` raises `TypeError`.
- The crop pipeline never invokes that method. Requesting a larger patch with
  `padding=True` still fails; the synthetic random-crop example raised
  `ValueError: low >= high`.
- The method's `np.pad(array, (0, diff))` pads every axis. Invoking the method
  through the class to bypass the collision, an `(8, 8, 8)` image with target
  `(10, 8, 8)` became `(10, 10, 10)`.

Proposed correction: reuse `Padder` and remove the duplicate padding algorithm,
with an explicit choice about keeping or retiring the boolean option.

### 4. The optional flip in deprocessing cannot execute

`Preprocessor.deprocess(..., flip_axis=[0])` raises `TypeError` because it passes
`verbose` to `Flip.__init__`, which does not accept that argument. The subsequent
call also omits the required `random_axis` argument to `Flip.execute`.

Proposed correction: define whether this option means a deterministic inverse
flip or a random augmentation, then call the matching operation explicitly.

### 5. Mask-only padding fails

`Padder.iterate` initializes `padding_values` only when images are provided,
then uses it for masks. `Padder((10, 10, 10)).apply(masks=image)` raised
`UnboundLocalError`.

Proposed correction: derive the shape from the first available input, with a
clear error when neither images nor masks are supplied.

### 6. Mask scaling can create fractional class labels

`Scale.execute` ignores `is_mask` when choosing interpolation. Scaling a binary
mask with `interpolation=1` produced values strictly between zero and one.
The constructor keyword is also misspelled, but is left unchanged for callers.

Proposed correction: force nearest-neighbor interpolation for masks. Handle
the spelling correction separately with a compatibility decision.

### 7. Near-spacing resampling mutates the original image

When spacings are within `atol=1e-2`, `Resample.execute` changes the input's
qform/sform and returns it without resampling. A synthetic spacing change from
`0.2` to `0.205` mm modified the original affine while preserving its data.
This matters because the loader retains image objects between calls.

Proposed correction: distinguish a genuine no-op from a geometry change and
avoid mutating caller-owned images.

### 8. The save reference is not applied

`SaveNifti.execute` does not use the supplied reference image's affine or
header. Saving with a reference having a different affine retained the input
affine, contradicting the existing method documentation and the apparent
intent of `deprocess`.

Proposed correction: specify the target grid first. Copying a header alone
cannot replace spatial resampling when grids differ.

## Additional correctness and design concerns

These findings come from source inspection; their effect on historical model
results has not been established.

- **Spatial coordinates:** cropping and padding retain the original affine
  despite shifting the voxel origin. `Scale` and `Flip` also retain geometry;
  distinguish intentional array augmentation from physical-space transforms.
  `Resample` constructs its affine using the input shape before computing the
  output shape. These paths need landmark and round-trip checks before changes.
- **Physical units:** `N4BiasFieldCorrection` calls `ants.from_numpy` without
  passing spacing, origin, or direction. `RandomAffine` likewise constructs
  SimpleITK images from arrays alone. Establish whether parameters are intended
  in voxel units or physical units, especially for anisotropic rat scans.
- **Label policy:** `MapLabels` silently maps every unlisted value to zero.
  Reversing a many-to-one mapping discards original labels. These are scientific
  policy decisions, not merely naming cleanup.
- **Saving state:** `current_path` persists when a later call omits `path`.
  Filename substitution handles `.nii.gz` but not `.nii` when only a postfix
  is supplied, potentially leaving the output equal to the input path.
- **Configuration:** N4 defaults contain shared mutable lists. Configuration
  export records dictionary types instead of their contents, so label mappings
  are not fully captured. Probability, dimension, range, and axis values lack
  consistent validation.
- **Spacing correction:** `CorrectX10` divides every axis by ten if any axis
  triggers its heuristic. Verify the intended scope before applying it to
  already-normalized or anisotropic images.

## Opportunities to reduce code later

1. Share one padding-width calculation across `Padder`, `PadderCutter`, and
   cropping; the same three-axis arithmetic currently appears several times.
2. Use one consistent transform interface. `Padder.execute`, `Flip.execute`,
   and `RandomCropping.execute` currently differ from the abstract base
   signature. Sample random parameters once per image/mask group and apply them
   through a small shared helper, while retaining the readable step classes.
3. Consolidate the repeated single-image/list handling and conditional return
   assembly in `PreprocessingStep` and `Preprocessor`. Decide whether the public
   return format must remain backward compatible before simplifying it.
4. Separate file saving and spatial restoration from the numerical transform
   loop. Retain the existing pipeline composition rather than introducing a
   larger framework.
5. Replace wildcard imports and remove unused imports after checking notebook
   usage; keep the existing namespace unchanged during this annotation pass.

## Verification performed

- Parsed and compiled the annotated module, imported it in the existing Python
  3.10 `rnet` environment, and resolved method annotations with
  `typing.get_type_hints`.
- Compared its executable AST with the version before this change, excluding
  annotations, docstrings, and the added typing imports/aliases: identical.
- Compared original and annotated image/mask arrays and affines with the same
  random seed for 13 transforms: resampling, spacing correction, label mapping,
  reorientation, normalization, padding, padding/cutting, scaling, noise,
  smoothing, random affine, flipping, and cropping. All comparisons matched.
- Compared saved NIfTI data and affines before/after in a temporary directory:
  identical. Reproduced the failure cases described above using synthetic data.

No training, real-dataset writes, or N4 solver run was performed. This is not a
claim that the module passes a full static type checker or that its existing
scientific behavior is correct.
