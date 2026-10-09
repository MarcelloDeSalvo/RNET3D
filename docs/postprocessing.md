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

Verification: 15 focused tests and direct exact-output comparison against the
pre-refactor implementation for four normal/default cases. No inference or
training notebooks were changed.
