# Experiment core and fixed-split training

Start with `notebook/01_training.ipynb` in the **rnet** environment. Review the
configuration, animal splits and validation batch, then set `RUN_TRAINING = True`.
The notebook defaults to normalized D1 and the historical four-class task:
background, lesion, merged lateral ventricles, and third ventricle.

## Ownership

- `experiments/config.py`: typed settings, validation, JSON serialization and
  configuration fingerprint. It does not initialize TensorFlow.
- `experiments/data.py`: explicit dataset registration, annotation availability,
  animal splits, preprocessing and generator factories.
- `experiments/training.py`: runtime setup, compilation, fixed-split execution,
  K-Fold adapters, inference export, provenance and Plotly history inspection.
- `evaluation/k_fold.py`: the shared `train_fold` implementation and K-Fold
  orchestration. Both run types use the same callbacks and selected checkpoint.
- `evaluation/inference.py`: typed prediction results, full-volume computation,
  reporting classes and `ExperimentVolumeEvaluator`. Computation and reporting
  stay separate responsibilities within this single module.

No architecture defaults or dropout rates are changed. Model settings with
`residual=None` and `attention=None` use the selected builder's existing defaults.
The current core accepts single-channel images and the existing model heads.

## Data and annotation policy

Register only requested datasets using `DataConfig.datasets`. `DatasetSpec`
supports normalized cohorts and legacy modalities. Legacy sources require
`modality` and `available_labels`, expressed as source-mask values including zero.
Original IDs and normalized animal identities are preserved by `SubjectIndex`.

For normalized schema-v3 metadata, `annotated_regions` and
`has_lesion_annotation` determine availability. The metadata's `label_mapping`
is a vocabulary, not a declaration that every class is annotated. SHAM scans
without a lesion annotation therefore have an unavailable lesion class.
Explicit `DatasetSpec.available_labels` overrides are available for other schemas
or a verified annotation policy. Unknown anatomical abbreviations fail clearly.

`LabelConfig.mapping` defines source values and contiguous output classes.
The default rejects unlisted voxel labels. The D1 notebook explicitly uses
`unknown="background"`: nine normalized D1 scans include extra anatomy labels,
which are projected to background for the selected four-class task. Availability
is checked only for the selected mapped source labels. For an expanded task,
define every source-to-class mapping and class name rather than relying on this
projection. Source values 0 and 1 retain their indexes so the shared preprocessing
pipeline also preserves binary brain masks.

Partially annotated samples require `dice_ce_gradient_fix`. The generator marks
unavailable output channels with -1, independently of the optional classifier.
A merged class is unavailable if any contributing source label is unavailable.
This does not infer missing annotation from a structure's absence in the mask.

The corrected objective retains its documented supervision policy: a voxel with
no positive available target contributes no ROI supervision. It does not invent
annotations or establish that background is anatomically complete in partially
annotated masks. Annotation/background semantics need a verified dataset policy.

## Preprocessing and splits

Every stage reorients to the same saved reference first. Optional CorrectX10
performs the existing unit heuristic; omit it when geometry has been verified.
Images use linear interpolation and masks nearest-neighbour interpolation.
Existing N4 inputs are loaded without repeating bias correction.

Before augmentation, all stages resample to an orthogonal grid. Small shear terms
occur in normalized D1 affines and cannot be used directly by ITK transforms or
spacing-based distance metrics. Resampling uses the closest orthogonal axis basis,
keeps the physical centre and covers the source field of view. It preserves
reference axis orientation and obliquity rather than changing headers alone.
The original grid remains available for prediction restoration.

Only training applies affine, blur, noise and flip augmentation. Training uses
a deterministic centre crop by default, or uniform random cropping when requested.
Validation always uses `PadderCutter`, not the stochastic centre-biased
`RandomCropping(mode="center")`. Inference keeps the complete volume.

Both training and validation retain their last incomplete batch. The existing
generator's default remains `drop_last=True` for historical callers. An optional
training `data_multiplier` repeats only training scan selections; validation and
test animal counts are never inflated. All selected timepoints remain together.

`make_split` uses a local RNG and fractions of the complete animal cohort. It is
unstratified. Use `SubjectSplit.load` for predefined or matched scientific splits,
especially the small longitudinal D2 cohort. Loading validates exact cohort
coverage, uniqueness, and disjoint train/validation/test animals. The split seed
is separate from the training seed.

## Execution and output contracts

Call `configure_runtime` before models or generators initialize devices. It sets
GPU memory growth, precision, and Python/NumPy/TensorFlow seeds. Accelerator-level
bitwise reproducibility is not guaranteed. The resolved runtime is recorded.

`run_training(prepared, split)` starts a fresh fixed-split run.
`run_cross_validation(prepared, ...)` supplies the same `ExperimentFactories` to
the existing K-Fold runner. The shared runner restores validation-selected best
weights before saving or evaluating the model. Early stopping does not separately
restore competing checkpoints.

For `dice_ce_gradient_fix`, training emits `regions_logits` and does not attach
historical probability-based ROI metrics. The brain head still emits probabilities.
Exports always expose `regions` softmax probabilities and omit optimizer/loss
serialization dependencies. Load new exports using Keras `load_model(...,
compile=False)`. Best weights belong to the fitting graph; the exported inference
model is the intended deployment artifact.

Fixed-split runs save:

```text
results/<experiment>/<run>/
  experiment.json
  metadata.json
  subjects.csv
  splits.json
  fold_0/
    split.json
    best.weights.h5
    model.keras
    history.csv
    timing.json
    evaluations/test/regions/
      inference_timings.csv
      predictions/<group>/<scan>/{regions,brain_mask}.nii.gz
  regions/full_volume_metrics.{csv,xlsx}
  summary/
    selected_epoch_metrics.csv
    average_metrics.json
    fold_timings.csv
    all_evaluations.csv
```

Optional evaluation artifacts are written only when inference is enabled.
The `.keras` filename follows the existing runner; TensorFlow 2.10 uses its
legacy serialization behavior. An inference export does not retain optimizer
state. Existing run folders are never overwritten.

Provenance records the configuration fingerprint, Git revision and dirty state,
environment, precision, reference-file SHA-256 and label/output conventions.
The complete configuration saves the source mapping, class names and input paths.

## Full-volume evaluation

`ExperimentVolumeEvaluator` predicts each scan once using full softmax outputs.
ROI measurements exclude background and use the complete aligned, reoriented,
resampled orthogonal grid in millimetres. Sheared grids or mismatched image/reference
geometry fail rather than producing misleading distances. No anatomical
postprocessing is implicitly applied.

Saved ROI and brain masks are restored to the original image's voxel grid.
ROI files contain **output class IDs**, not source-mask values; a many-to-one
mapping cannot recover left/right anatomy. The mapping and names are saved in
the experiment configuration. Offline metric rows retain subject, scan, region,
units, source dataset and annotation/empty-mask status.

Device warm-up is excluded from per-volume inference time. Prediction uses Keras
`predict`, which materializes NumPy results before the timer stops. Preprocessing
and prediction durations are reported separately. Timings include sliding-window
reconstruction and are not isolated GPU-kernel benchmarks.

K-Fold validation currently serves checkpoint selection and diagnostic evaluation.
It is not an independent outer test estimate. Supply a disjoint evaluation cohort
for final performance, and saved splits for seed-matched comparisons. Repeated
external-animal predictions across folds do not increase biological sample size.

## Subsequent steps

Fine-tuning transfer audits, interrupted-run resumption, independent outer/inner
CV splits, controlled comparison automation, and manuscript statistics notebooks
remain subsequent work. Best weights alone are not a resumable optimizer state.
This first workflow performs no sensitivity analysis or automatic statistical
model comparison. Notebook plots use Plotly's native VS Code renderer; the shared
runner also writes standalone HTML history artifacts for optional offline use.
