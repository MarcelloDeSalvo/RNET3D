# Postprocessing callbacks

`src/evaluation/postprocessing.py` retains the existing callback factories and
parameter defaults. Functions and intermediate values now have types, helpers
have docstrings, and logical sections/call arguments are separated for readability.

Morphology still fills holes before removing small objects, using the original
library connectivity defaults and voxel-count thresholds. Enabled operations
return float64; disabling both operations returns the original array. This
operation does not guarantee a single connected component.

Ventricle division retains the input labels 0/1/2/3 and output labels 0/1/2/3/4,
PCA over all third-ventricle voxels, the largest third-ventricle component's
centroid, the original symmetry objective, and Nelder-Mead optimization. Both
centroid assignment and voxel-wise splitting retain their original plane rule.

Small fixes:

- `default_hemisphere='left'` now selects the opposite fallback side from `right`.
  The historical right convention means the side containing array origin.
  This requires consistently reoriented inputs; no anatomical orientation or
  physical affine is inferred from an array.
- PCA visualization no longer samples 20,000 points from fewer available points.
- Missing/undersized lesions are handled explicitly, so unrelated ValueErrors
  are no longer swallowed by the fallback.
- Invalid settings, unsupported label maps and invalid optimized normals raise
  clear errors. Splitting requires at least three third-ventricle voxels; this
  is a sample-count requirement, not a guarantee of reliable anatomy or PCA.
- With no lateral ventricles, plane fitting is unnecessary: return a copy with
  third-ventricle label 3 changed to 4. No missing plane is invented.

## Simplified structure

The callback retains its public signature and now has one nested function.
A private `_SymmetryPlane` dataclass groups the fitted origin, normal, plane
fitting and point-side calculation. Plotting and the symmetry objective are
separate helpers, and a shared largest-component helper serves both masks.

Voxel assignment uses array indexing instead of a Python loop over every voxel.
Labels are written directly into an input copy, eliminating separate ipsilateral
and contralateral output masks and repeated split calls. Ventricular connected
components are computed only when centroid-based assignment is actually used.
The hard split still applies to missing/undersized lesions in either mode.

Verification: 18 focused tests, including crossing components, on-plane points
and empty ipsilateral selections. Exact output comparison against the previous
committed version passed for 96 synthetic cases spanning both split modes,
both fallback hemispheres and absent/tiny/valid lesions.

## Visual step checks

`notebook/check_postprocessing.ipynb` provides a short runnable flow with
separate cells for merged-ventricle separation, brain-hole filling, small-object
removal and restoration of original annotation IDs. A synthetic prediction
with known sides, a brain hole and an island makes the default behavior visible.
Select normalized annotations as a demonstration input or explicit MRI/ROI/brain
files to inspect real outputs; specify whether the ROI file contains original
or raw model IDs. All inputs first follow the shared RIA reference. Real inputs
also use the configured units correction and 0.1 mm orthogonal grid.

Intermediate figures name their model/postprocessed class IDs explicitly. The
last cell uses `FullVolumePostprocessor` with editable `PostprocessingConfig`
settings and restores original IDs. MouseFull keeps its directly predicted
sides. Brain cleanup does not modify categorical ROI masks. Displays remain on
the prepared reference grid; full-volume evaluation restores the original MRI
grid when exporting predictions. No source files or performance results are
rewritten by this diagnostic notebook.
