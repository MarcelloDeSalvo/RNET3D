# Short D1 interpolation sensitivity study

Open [check_interpolation_sensitiity.ipynb](../notebook/check_interpolation_sensitiity.ipynb)
with the `rnet` kernel. The filename follows the requested spelling. It compares
MRI resampling orders 0, 1, 2 and 3 on normalized D1, using the shared training
and evaluation core. Training is disabled until `RUN_STUDY=True`.

`PreprocessingConfig.resampling_order` stores the SciPy spline order: nearest,
linear, quadratic or cubic in this study. See the
[SciPy interpolation definition](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.affine_transform.html).
It applies to MRI target-grid resampling during training, validation and
inference. Its default is 1, including older saved configurations without the
field. The low-level resampler supports orders zero through five. Masks always
use nearest neighbour, and physical affine MRI augmentation stays linear for
all variants. Do not pass these integers to SimpleITK as interpolation enums.

All runs start from scratch with the original D1 four-class task, 80-cube patches
and filter widths `(16, 32, 64, 128, 256)`. The original multiscale/residual/
attention/fused-head model is retained. Shared preprocessing uses the RIA
reference, native-grid intensity degradation, center-biased random cropping
and LR reflection. Postprocessing separates the merged ventricles by centroid,
cleans the brain head and exports original ROI IDs. Predictions need not be
written: `SAVE_PREDICTIONS=False` retains metrics and checkpoints.

The default pilot uses all D1 animals, ten epochs, four training batches per
epoch, batch size eight, multiplier one and seed 42. A preset cosine learning
rate schedule and fixed epoch count keep optimizer budgets matched without
validation. Callbacks are explicit; the default evaluates the last epoch and
retains recovery checkpoints. `USE_VALIDATION_FOR_SELECTION=True` enables a
separate inner validation set and best-validation selection instead.
Validation/test cover every saved deterministic evaluation patch and every
selected full volume. `STEPS_PER_EPOCH=None` restores complete training epochs.
Use a fresh study folder for changes to budget, orders, seeds or callbacks.

One shared animal split and one saved evaluation-patch collection are used for
every order and seed. `SPLIT_FILE` optionally supplies exact existing memberships.
The study contract, each run's experiment file and the plan record the order.
Grid-resampling order does not change target geometry, so shared crop coordinates
remain valid; MRI intensities are recomputed under each variant's pipeline.
Masks are reprocessed with the same nearest-neighbour policy in every variant.

The notebook audits spacing after the actual reorientation/unit-correction
prefix. Inputs already on the target grid bypass resampling. The training cell
refuses a study where no input is affected. A pooled D1 difference can otherwise
be diluted by unchanged scans: per-sequence and affected/unaffected animal Dice
tables accompany the overall comparison. The audit is saved as
`resampling_audit.csv` when the study finishes.

The affected flag uses the resampler's exact tolerance. Small affine/header
rounding can flag nominally isotropic inputs too; the audit also records actual
spacing, anisotropy and maximum relative spacing change. A read-only header check
of the current D1 inventory found all 112 scans trigger the resampler, including
small spacing deviations in T1w and larger through-plane differences in T2w.
The flag alone is not a measure of interpolation impact.

To resume, retain the folder and exact settings, enable `RESUME_STUDY` and rerun
configuration/data cells before training. The shared runner reuses completed
fits and restores interrupted fits. To inspect completed results after restart,
run setup/configuration and the result cells with explicit `RESULTS_DIR`;
training is unnecessary. Source data, original notebooks and earlier runs are
not modified.

If a run reports a NumPy allocation failure from resampling, this is host RAM
exhaustion, even when TensorFlow wraps it in `ResourceExhaustedError`. Preprocessing
and evaluation now read NIfTI data without filling the source images' persistent
float64 caches. A representative D1 scan previously retained about 60 MiB across
its MRI and two masks; keeping every scan cached consumed several GiB. Resampling
still uses the same float64 calculations and interpolation orders. Generator
batches use float32, matching the conversion Keras already performed before
fitting, and completed fits release their model/callback references.

After updating from the caching implementation, restart the notebook kernel to
release arrays and TensorFlow state retained by the failed execution. Keep the
same `STUDY_DIR`, enable `RESUME_STUDY=True`, and rerun the setup and training
cells with the same experiment settings. Recovery restarts the interrupted
epoch from the last committed checkpoint; completed variants are reused. If
no epoch has finished yet, that variant starts again from its initialization.

The anisotropic x10 geometry correction is a separate fix. Eight D1 T2w inputs
(`SUBJ_0089`, `SUBJ_0090`, `SUBJ_0091`, `SUBJ_0108` through `SUBJ_0112`) store
`(1, 1, 3)` mm spacing. The older per-axis rule incorrectly left the third
spacing at 3 mm, producing 1,110 slices. These scans now use `(0.1, 0.1, 0.3)` mm,
giving 111 slices at the 0.1-mm target spacing. If a study used the older rule,
select a fresh `STUDY_DIR` and set `RESUME_STUDY=False`: its saved patches and
training geometry are obsolete. Keep its existing files for provenance. The
loader rejects saved patch contracts with the old correction version.

Validation lesion Dice supplies an exploratory ranking. Held-out tables contain
native patch measurements separately from postprocessed full-volume ROI/brain
Dice, IoU, HD95, ASSD, sensitivity, precision and volume measurements. Paired
comparisons use order 1 as reference: candidate minus linear, with animal
bootstrap intervals and Holm-adjusted Wilcoxon p-values. Timing records retain
their existing workload definitions. Optional fresh-folder exports include
validation/test paper workbooks, CSVs, the validation ranking and sequence table.

This is a short-budget convergence/sensitivity screen, not a universal ranking of
interpolators. Different seeds can change outcomes; a single split does not
measure retraining uncertainty. Selecting a setting from test performance makes
that cohort part of development. Higher spline orders may overshoot input
intensities; no extra clipping is silently introduced.

Tests cover configuration validation/default loading, routing at every stage,
fixed affine interpolation, identical discrete GT/grid across orders, shared
evaluation coordinates with recomputed MRI, a disabled notebook with no fitting,
and a four-model synthetic training/evaluation pass. No real D1 training is
launched while creating the notebook.
