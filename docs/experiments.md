# Experiment core and training notebooks

Start with `notebook/01_tbi_mouse_training.ipynb` in the **rnet** environment. Review the
configuration, animal splits and validation batch, then set `RUN_TRAINING = True`.
The notebook defaults to normalized D1 and the historical four-class task:
background, lesion, merged lateral ventricles, and third ventricle.
`notebook/02_tbi_mouse_kfold.ipynb` uses the same factories for grouped K-Fold,
with separate checkpoint-selection validation and independent outer holdout
animals. Training is disabled until its configuration and split counts are reviewed.
For the historical rat CV protocol, use `notebook/04_rat_kfold.ipynb` and set
`MOUSE_CHECKPOINT`. Its complete D2 configuration uses 112-cube patches, five folds,
100 epochs, batch size four and training multiplier five. Each fold initializes
from the same mouse weights, trains on four rats, and evaluates every available
timepoint of the remaining rat. That rat also selects the checkpoint, matching
the old `3DRNET_Kfold_rats.ipynb` split; results have the `validation` role, not
an independent `holdout` role. Validation reuses five saved center-biased patches per scan and retains
incomplete batches. The original notebook's random validation
augmentation and nearest image interpolation are not reproduced. Full-volume
postprocessing retains the historical ventricle separation and brain cleanup.
`notebook/03_rat_finetuning.ipynb` adapts a pretrained mouse model to normalized
D2 using the fixed-split runner. All timepoints of each of its five rats stay
within one split role, including the additional 45-day scan when present.
The exploratory default is three training, one validation and
one test rat; load a chosen split for manuscript work. Set `MOUSE_CHECKPOINT`
explicitly before enabling training. The default complete transfer retains both
pretrained heads and fine-tunes all layers with a fresh optimizer. Optional
backbone transfer reinitializes heads; optional freezing includes batch
normalization. Data previews work without a checkpoint, but training rejects
that scratch preview configuration. Resume automatically reloads the saved split.

`notebook/05_mouse_full_training.ipynb` trains and evaluates the combined D1+D3
task. It keeps the historical eleven-class channel order, separate bilateral
anatomy and SHAM/TBI classifier. It explicitly selects the corrected logits-based
partial-label loss and displays annotation coverage before fitting. Evaluation
predicts the complete test scans and reports missing annotations separately from
empty structures. Extra anatomy outside the configured task maps to background.
The current original-width model configuration is explicit; it is not inferred
from the small-filter prototype in the old domain-adaptation notebook.
Notebook 05 and MouseFull K-Fold notebook 06 expose `USE_TBI_PRETRAINING=True`,
`TBI_RUN`, `TBI_FOLD` and `FREEZE_BACKBONE=False`. Set the source run explicitly;
an interrupted TBI run with one committed epoch is enough. Without a source,
data previews remain available but the training cell cannot silently start from
scratch. Set `USE_TBI_PRETRAINING=False` to choose scratch training deliberately.
Pretraining transfers encoder/decoder feature weights, while the eleven-class
ROI head, binary brain head and SHAM/TBI classifier receive fresh initialization.
All transferred features are fine-tuned unless freezing is explicitly enabled.
Each CV fold reloads the same selected source features and gets fresh heads and
a fresh optimizer. Training resumption restores the target run's fitting state.

MouseFull contains the same D1 animals as TBI training. A source that trained or
selected weights on later held-out animals compromises independent evaluation.
For fixed splits, select target validation/test animals unseen in source training
and source selection-validation. A shared CV source must exclude every evaluated
animal across the target folds; otherwise fold-specific pretraining sources and
matching exclusions are needed. Source selection is explicit; these notebooks
do not infer that a checkpoint is independent of their new splits.

`notebook/06_mouse_full_kfold.ipynb` uses the same complete MouseFull settings
with five grouped outer folds. Each remaining pool is split into training and
checkpoint-selection validation; every CV animal is evaluated in one outer
holdout. Its default `FOLD_MODE="balanced"` stratifies newly generated outer
holdouts by complete versus partial anatomical annotation coverage, excluding
lesion annotation from this balance. An animal is complete only when all selected
anatomical regions are available in every scan. SHAM animals with complete anatomy
can therefore count as complete despite unavailable lesion annotations. Both
groups are spread approximately evenly across folds, preserving their cohort
ratio without oversampling. `FOLD_MODE="random"` retains unstratified outer folds;
checkpoint-selection validation remains random in either mode. Saved or resumed
folds are loaded unchanged and take priority over generation settings.
Per-fold complete/partial animal counts and annotation coverage are displayed
before fitting. Optional saved
fixed-split cohorts reserve a separate shared test set, and saved folds support
matched comparisons and resumption. Other cohort features are not stratified.

`notebook/07_ablation.ipynb` defines a complete D1 reference configuration and
single-setting residual, attention and downsampling variants. Optional training
strategy rows are separate from architectural changes. The same fixed split or
independent outer folds are reused across variants and training seeds. The study
saves its complete contract, matrix, shared selection, run index and raw metrics.
Explicit study resumption delegates optimizer/callback recovery and completed-run
reuse to the core. Descriptive plots average longitudinal scans within each
animal while retaining variant, seed and fold identity; no automatic significance
test or interpolation sensitivity analysis is performed.

`notebook/07f_ablation.ipynb` preconfigures that same ablation workflow for
MouseFull D1+D3, retaining notebook 05's eleven-class order, classifier,
corrected partial-label loss, filter widths and preprocessing. It has its own
study folder and supports matched fixed splits or grouped outer folds. The
default component variants retain the classifier; an optional commented row
removes that head alone. Training remains disabled until explicitly enabled.

The notebook directory now separates complete experiment configurations instead
of selecting rat or mouse presets inside a generic notebook. Renaming the
executed TBI notebook preserves its saved cells and outputs. See
[the notebook index](../notebook/README.md) for all implemented workflows.

`notebook/08_unet_comparison.ipynb` uses the same study executor as ablations.
It compares the existing original-width 3DR-Net with the existing conventional
U-Net, disabling residual/attention options for U-Net and omitting the classifier
for both. Scratch initialization, labels, loss, preprocessing, animal selection,
training seeds and maximum budgets match. Identical widths do not imply equal
capacity; actual epochs, parameter counts and durations are saved by the runner.

`notebook/09_inference_evaluation.ipynb` accepts saved `experiment.json` and
`fold_N/model.keras` files plus explicit animal IDs or a single saved split role.
Optional input registrations/reference paths relocate inputs or define external
cohorts. The shared evaluator checks head/input shapes, predicts full volumes,
restores original-grid masks and records physical metrics, statuses and timings.
It saves source/evaluation configurations, a checkpoint hash and the data contract
into a fresh folder. Output tensor shapes cannot verify label semantics; choose
the experiment configuration belonging to the checkpoint. This workflow measures
ROI segmentation; it does not compute classifier accuracy. Explicit settings
apply historical postprocessing and restore original annotation IDs.
Offline data preparation retains annotation masks but skips the training-only
loss restriction, allowing an older probability-loss model to be measured on
partially annotated external data. The training restriction remains enabled.

`notebook/10_revision_results.ipynb` consumes explicit saved metric CSVs.
It audits statuses, averages model repetitions within scans and then scans within
animals, and bootstraps biological animals. Volume agreement applies existing
MAE, Bland–Altman, ICC(A,1), CCC and diagnostic methods to animal mean scan volumes.
Optional paired model comparisons use common finite scans and animal pairs,
with mean-difference intervals and exploratory Holm-adjusted Wilcoxon tests.
Results retain cohort/role/region/units, counts and raw sources. Missing labels
and undefined distances are never zero-filled. Exports use a fresh folder.

## Ownership

- `experiments/config.py`: typed settings, validation, JSON serialization and
  configuration fingerprint. It does not initialize TensorFlow.
- `experiments/data.py`: explicit dataset registration, annotation availability,
  animal splits, preprocessing and generator factories.
- `experiments/training.py`: runtime setup, compilation, fixed-split execution,
  K-Fold adapters, inference export, provenance and Plotly history inspection.
- `experiments/initialization.py`: strict complete weight loading or validated
  backbone transfer, with optional freezing and a saved transfer audit.
- `experiments/ablation.py`: validated single-setting matrices, matched study
  execution, recovery routing, raw result exports and descriptive animal means.
- `experiments/comparison.py`: explicit matched 3DR-Net/U-Net matrices and
  parameter inspection, delegating execution to the existing study runner.
- `evaluation/revision.py`: saved-metric source validation, biological-animal
  summaries and bootstrap intervals, common-scan model pairing, volume agreement
  and auditable manuscript CSV exports.
- `evaluation/k_fold.py`: the shared `train_fold` implementation and K-Fold
  orchestration. Both run types use the same callbacks and selected checkpoint.
- `evaluation/training_state.py`: epoch recovery, optimizer and standard callback
  state, atomic recovery records, and a process lock for each run directory.
- `evaluation/inference.py`: typed prediction results, full-volume computation,
  reporting classes and `ExperimentVolumeEvaluator`. Computation and reporting
  stay separate responsibilities within this single module.

No network builder defaults or dropout rates are changed. `ModelConfig` uses
explicit boolean flags: `residual=True` and `attention=True` match the original
3DR-Net. For the conventional U-Net baseline, explicitly select
`ModelConfig(architecture="unet", residual=False, attention=False)`.
The current core accepts single-channel images and the existing model heads.

## Data and annotation policy

Register only requested datasets using `DataConfig.datasets`. `DatasetSpec`
supports normalized cohorts and legacy modalities. Legacy sources require
`modality` and `available_labels`, expressed as source-mask values including zero.
Original IDs and normalized animal identities are preserved by `SubjectIndex`.
`DataConfig.animal_ids` maps existing canonical animal IDs to confirmed shared
identities, including duplicates across cohorts. All scans move together; unknown
IDs fail. No cross-cohort biological identity is inferred automatically.

For normalized schema-v3 metadata, `annotated_regions` and
`has_lesion_annotation` determine availability. The metadata's `label_mapping`
is a vocabulary, not a declaration that every class is annotated. SHAM scans
without a lesion annotation therefore have an unavailable lesion class.
Explicit `DatasetSpec.available_labels` overrides are available for other schemas
or a verified annotation policy. Unknown anatomical abbreviations fail clearly.

`DatasetSpec.annotation_scope="full_volume"` declares that every available
foreground ROI has been fully delineated throughout each selected scan. Its
mask defines positive voxels and known negatives outside the ROI. An annotated
empty ROI is still available; zero voxel counts do not establish missing
annotation or known absence. SHAM status alone does not establish lesion
annotation. Availability always comes from metadata or an explicit override.
Sparse slices, scribbles and incomplete contours need spatial validity masks
and are rejected when declared through `annotation_scope`. Existing schema-v3
metadata lacks this field, so its scope comes from the explicit dataset policy;
the code cannot verify annotation completeness from voxel values. Metadata that
declares a conflicting scope is rejected, including when availability is
overridden. These policies are stored in `experiment.json`.

Source value zero is reserved for the task remainder. It may include true
background, unselected anatomy or unavailable regions. Including zero in
`available_labels` is an encoding requirement, not proof of complete background
annotation. `annotation_summary()` exposes this distinction. Background
supervision is determined separately by the configured objective; the version-2
partial-label loss has no explicit background Dice or CE term.

`LabelConfig.mapping` defines selected source values and contiguous output classes.
The default `unknown="background"` projects additional anatomy to task background,
including extra annotations present in MouseFull masks. These source classes do
not create output channels or evaluation regions. Their voxels remain background
targets; this is not exclusion of those voxels from the loss. Use `unknown="error"`
explicitly for strict source-vocabulary checks. Availability is checked only for
the selected mapped source labels. For an expanded task,
define every source-to-class mapping and class name rather than relying on this
projection. Source values 0 and 1 retain their indexes so the shared preprocessing
pipeline also preserves binary brain masks.

Partially annotated samples require `dice_ce_gradient_fix`. The generator marks
unavailable output channels with -1, independently of the optional classifier.
A merged class is unavailable if any contributing source label is unavailable.
This does not infer missing annotation from a structure's absence in the mask.

Version 2 of the corrected objective averages Dice over available foreground
ROIs, including their known negative voxels and annotated empty masks. CE uses
only positive available foreground targets and averages over available foreground
entries there. Background remains a softmax competitor without an explicit Dice
or CE term. Unavailable anatomical logits are excluded before normalization and
have zero direct ROI-loss gradient. This depends on full-volume ROI delineation;
it does not assert that zero source voxels are verified background. Dice smoothing
uses `(2 * intersection + smooth) / (union + smooth)` so correctly empty masks
have zero loss. Legacy loss computation remains unchanged.

MouseFull notebooks 05/06/07f explicitly record `gradient_fix_version=2` in
`LossConfig`. Older saved partial-label configurations without this marker are
version 1 and can be evaluated or used for initialization, but cannot resume
fitting or an ablation/comparison study under the changed objective. Start a new
run for retraining. Configured version-1 objectives are refused by training
preparation/compilation; offline preparation accepts them without compiling.

## Preprocessing and splits

Every stage reorients to the same saved reference first. Optional CorrectX10
performs the existing unit heuristic; omit it when geometry has been verified.
Images use linear interpolation and masks nearest-neighbour interpolation.
Existing N4 inputs are loaded without repeating bias correction.

Before spatial affine augmentation, all stages resample to an orthogonal grid. Some D1 sforms
contain small nonorthogonal coefficients consistent with rounded headers; this
does not establish physical shear in the acquired images. The current physical
transform and distance checks require orthogonal axes. Resampling uses the closest orthogonal axis basis,
keeps the physical centre and covers the source field of view. It preserves
reference axis orientation and obliquity rather than changing headers alone.
The original grid remains available for prediction restoration.

New training configurations explicitly set `intensity_augmentation_grid="native"`:
reorientation and unit correction precede MRI-only blur/noise on the native
acquisition grid, followed by linear target resampling, physical affine
augmentation, normalization, cropping and optional LR flipping. Interpolation
therefore acts on the injected noise field as it acts on acquisition noise.
Mask interpolation stays nearest-neighbour for both resampling and affine.
No spatial-transform fusion or interpolation sensitivity study is introduced.

The notebook configuration exposes `blur_sigma_range=(0.0, 0.6)` with
`blur_sigma_units="voxels"`, and `noise_std_range=(0.0, 0.05)` with
`noise_std_units="intensity"`. These retain the previous default magnitudes;
moving them to the native grid changes their effect. Native voxel blur follows
the acquisition's anisotropic spacing. Select `blur_sigma_units="mm"` to convert
sampled per-axis millimetre sigmas using input voxel sizes. Select
`noise_std_units="image_std"` to multiply sampled noise standard deviations by
the full input MRI standard deviation, including background. A value of 0.05
then means 5% of that scale, not a specified MRI SNR. No dataset-level statistics
are fitted. Noise means remain in intensity units. These switches are explicit;
no default strength is silently retuned for a modality.

Loading older configurations without the grid field resolves it to `resampled`,
retaining their previous resampling -> affine -> blur -> noise sequence and
default units. Newly constructed configurations default to `native`. Resuming
requires the saved order/ranges/units, and a changed configuration is rejected.
Evaluation has no augmentation, and old model weights remain usable. For a new
native-order experiment, start a fresh run or explicitly initialize from weights.

Only training applies affine, blur, noise and flip augmentation. Training uses
`RandomCropping(mode="center", std=None)` by default. This samples patches
around the centre; it is stochastic rather than a fixed centre crop.
`crop_mode="random"` selects uniform training crops, and `random_crop=False`
uses a fixed `PadderCutter` crop instead.
Validation always uses `PadderCutter`, not the stochastic centre-biased
`RandomCropping(mode="center")`. Inference keeps the complete volume.

Both training and validation retain their last incomplete batch. The existing
generator's default remains `drop_last=True` for historical callers. An optional
training `data_multiplier` repeats only training scan selections; validation and
test animal counts are never inflated. All selected timepoints remain together.

`ExperimentConfig.fixed_split` stores `FixedSplitConfig(validation_fraction=0.2,
test_fraction=0.1)`, defaulting to 70/20/10 train/validation/test proportions of
the complete animal cohort. `prepared.make_split()` uses this policy and
`split_seed`; shared study selection uses the same method. Fractions are visible
in fixed-split notebooks and saved in `experiment.json`. Validation and test
counts are rounded up independently, with the remainder used for training;
small cohorts will therefore have different realized proportions. Notebook 03
explicitly retains 20% validation and 20% test for its five-rat cohort (3/1/1).
`split_summary()` reports actual animal fractions and scan counts. The low-level
`make_split` still accepts explicit fractions and uses a local RNG.

Splits are unstratified. Use `SubjectSplit.load` for predefined or matched scientific splits,
especially the small longitudinal D2 cohort. Loading validates exact cohort
coverage, uniqueness, and disjoint train/validation/test animals. The split seed
is separate from the training seed. Saved/manual memberships take precedence
over generation fractions: loading never regenerates or resizes a split.
Saved configurations predating these explicit fields load with the previous
defaults. During resumption, older files that did not record generation fractions
use the current declaration for configuration comparison, while their exact
saved memberships remain mandatory. Newly recorded fractions are checked
strictly. The same normalization applies to saved study configurations;
neither resumption path rewrites or regenerates existing split files.

`make_cv_splits` assigns every animal to one outer holdout fold. Each remaining
pool is partitioned into training and separate checkpoint-selection validation,
using `cv_validation_fraction` (default 0.2). Optional `strata` maps selected
animal IDs to annotation groups for stratified outer assignment. `annotation_groups`
uses declared availability, with all scans grouped and conservative merged-class
rules. `fold_annotation_summary` audits complete/partial counts by fold and role.
Roles are disjoint and cover the
whole pool; every outer holdout animal is evaluated once. These
splits are not a nested hyperparameter search. Small or imbalanced cohorts need
scientifically chosen split files. Legacy two-role files remain usable by the
core for diagnostic validation, but the mouse K-Fold notebook requires outer holdouts.
Alternatively, `make_leave_one_animal_out_splits` constructs sorted two-role
folds for the old rat protocol. Notebook 04 uses these explicitly and evaluates
validation volumes, preserving every timepoint. Notebook 02 retains independent
outer test animals.

## Execution and output contracts

Call `configure_runtime` before models or generators initialize devices. It sets
GPU memory growth, precision, and Python/NumPy/TensorFlow seeds. Accelerator-level
bitwise reproducibility is not guaranteed. The resolved runtime is recorded.

`run_training(prepared, split)` starts a fresh fixed-split run.
`run_cross_validation(prepared, ...)` supplies the same `ExperimentFactories` to
the existing K-Fold runner. The shared runner restores validation-selected best
weights before saving or evaluating the model. Early stopping does not separately
restore competing checkpoints.

Every training/study notebook has a **Configure callbacks directly** cell. Its
actual Keras callback list is passed with `callbacks=callbacks`, and
`TrainingConfig.default_callbacks=False` prevents another automatic list.
Fixed-split notebooks explicitly use ReduceLROnPlateau (`val_loss`, factor 0.5,
patience 10, minimum LR 1e-7, verbosity 1), TensorBoard (histograms every epoch),
EarlyStopping (patience 20, mode min, verbosity 1), and CSVLogger. K-Fold
notebooks use only ReduceLROnPlateau by default, with no early stopping.
Ablation/model-comparison notebooks select the same policies according to `MODE`.

EarlyStopping uses `restore_best_weights=False` because the runner already
reloads its validation-selected checkpoint before exporting/evaluating. Restoring
weights inside the epoch-end early-stopping hook would pair earlier weights with
the latest optimizer slots in recovery state. The final selected-best behavior
is preserved. Checkpoint selection remains controlled by the notebook's explicit
`TrainingConfig.monitor`/`monitor_mode`, and best/recovery checkpoints are saved
even when the editable callback list is empty.

Each fit receives fresh copies of the unused callback instances; folds, seeds
and variants never share their fitting state. Relative TensorBoard/CSV paths
resolve within the current fold directory, and CSV logs append on resume.
`callbacks.json` saves standard callback constructor settings. Resume validates
those settings as well as the existing callback classes/order/state; study
contracts also include them. Editing patience, LR reduction settings, monitors,
or callback order defines a new protocol and requires a new run. Older runs
without an explicit callback contract retain their existing recovery checks.

`InitializationConfig` defaults to scratch. `mode="weights"` loads a complete
compatible weight file without partial matching; its output label semantics must
match the target. `mode="backbone"` loads a saved inference model, or reconstructs
a weights-only source graph from `source_experiment=Path(".../experiment.json")`.
Reconstruction uses the saved architecture, class count and fitting-output view;
it does not load source data, initialize transfers, compile or restore its optimizer.
Transfer compares feature topology and every weight shape before copying, and leaves ROI,
brain and classifier heads newly initialized. `freeze_backbone=True` freezes
feature layers including batch normalization. For a repository run, use:

```python
initialization = InitializationConfig.from_run(
    run_dir=TBI_RUN, fold=TBI_FOLD, freeze_backbone=False,
)
```

This selects the weight snapshot paired with `training_state.json`, falling back
to `best.weights.h5` only when there is no recovery record. A completed fit or
inference export is not required. Source checkpoint/configuration hashes and
copied/reinitialized/frozen layer lists are saved. Resume also verifies the source
configuration hash. Original model architecture and dropout defaults stay unchanged.

Pass `resume_run=Path(...)` to either runner to continue an interrupted run.
The core validates the saved configuration, animal/scan manifest, annotations,
input geometry and source-file size/modification times, reference/checkpoint
hashes, precision, TensorFlow version and splits. Completed folds reuse saved
results. Incomplete fitting restores model and optimizer slots from the last
committed epoch, plus history, Python/NumPy RNG state and standard checkpoint,
early-stopping and learning-rate callback state. An interrupted epoch restarts.
Custom callback state and exact stochastic augmentation replay are not promised.
Epoch budgets and settings must remain unchanged; this is recovery, not a new
experiment with an extended budget. Existing best-only runs cannot recover an
optimizer state. A per-run process lock rejects concurrent writers.

For `dice_ce_gradient_fix`, training emits `regions_logits` and attaches
`PartialLabelDice`, logging `regions_logits_mean_dice` and its validation counterpart.
It converts logits to full softmax probabilities, excludes background and missing
annotations, and averages class Dice after pooling available ROI entries across
the epoch, including known negatives with no positive available target. Classes
without annotation are excluded. This soft patch metric is
separate from full-volume animal evaluation and does not change the loss.
Rebuild/recompile the model to enable a newly added metric; an active `fit` does
not pick up source changes. Recover interrupted runs with `RESUME_RUN` pointing
to their existing run folder. History leaves earlier Dice observations blank
when that metric was not logged before recovery; it does not reconstruct them.
The brain head still emits probabilities.
Exports always expose `regions` softmax probabilities and omit optimizer/loss
serialization dependencies. Load new exports using Keras `load_model(...,
compile=False)`. Best weights belong to the fitting graph; the exported inference
model is the intended deployment artifact.

The final section of training/study notebooks 01–08 compares saved test or outer
holdout predictions with mapped GT over the MRI. Rat leave-one-animal-out uses
its validation role. `VIEW_RUN` can target an existing matching run without
fitting. A scan manifest preserves animal, timepoint and fold; native Plotly
menus choose loaded scans/planes and a slider chooses slices. All inputs follow
the configured reference orientation, and the title identifies unavailable GT
regions. Previews may sample slices/pixels for display; exported masks and
evaluation measurements remain unchanged. Prediction saving must have been
enabled; viewing does not run inference again.

To evaluate a stopped run before its final export exists, use the last optional
cell in notebooks 01–08. Set `CHECKPOINT_RUN`, `CHECKPOINT_FOLD` and
`RUN_CHECKPOINT_EVALUATION=True`. Only repository setup and that cell are needed
after a fresh kernel start. `best_checkpoint_settings` resolves saved animals,
configuration and best fitting weights; `evaluate_saved_model` reconstructs the
fitting graph without initialization transfer, compilation or an optimizer,
then exposes probability outputs for full-volume inference. The committed
selected-weight snapshot is preferred when recovery state exists. Historical
best-only runs can also be evaluated even though they cannot resume fitting.
Outputs must be outside the source training run; fitting state stays untouched.

Notebook 09 provides the same path through `RUN_DIR` and
`CHECKPOINT_KIND="weights"`, while `"model"` retains exported-model loading.
Its final native Plotly viewer includes MRI, mapped GT and prediction. Set
`VIEW_EVALUATION_DIR` to reopen a prior evaluation, including one created by a
training notebook, without predicting again. Checkpoint hashes and kind are
recorded in `evaluation.json`; saved annotation policies remain in effect.

MRI notebook configurations now share the RARE `TBI_fm_19_49_N4.nii.gz`
orientation reference, whose axes are R, I, A. LR augmentation resolves its
axis from the image affine. D1 configurations explicitly enable it at 0.5;
rat fine-tuning/CV retain 0.5, and MouseFull disables it at 0.0. Relative ipsi/contra IDs are not permuted
because the lesion/craniotomy side moves with a paired reflection. The MRI,
ROI and brain masks share all spatial transforms; validation/inference do not
perform augmentation.

Notebook 09 and the final checkpoint cells explicitly override older saved
references with this common reference, as requested. That changes the input
convention for models trained with another orientation; saved/evaluation
references remain recorded separately. Reopening old evaluation tables only
changes the viewing orientation, not the already computed metrics. Resume
continues to require its saved configuration; changed preprocessing settings
are for a new training run.

Fixed-split runs save:

```text
results/<experiment>/<run>/
  experiment.json
  data_contract.json
  metadata.json
  subjects.csv
  splits.json
  fold_0/
    split.json
    best.weights.h5
    initialization.json
    training_state.json
    latest/  # model/optimizer recovery and committed selected-weight snapshot
    completed.json
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
state. Fresh runs do not overwrite existing run folders; explicit recovery
updates the matching run. Recovery state is separate from the best inference
checkpoint, and restores the committed selected weights after a partial write.

Provenance records the configuration fingerprint, Git revision and dirty state,
environment, precision, reference-file SHA-256 and label/output conventions.
The complete configuration saves the source mapping, class names and input paths.

## Full-volume evaluation

`ExperimentVolumeEvaluator` predicts each scan once using full softmax outputs.
ROI measurements exclude background and use the complete aligned, reoriented,
resampled orthogonal grid in millimetres. Sheared grids or mismatched image/reference
geometry fail rather than producing misleading distances. All training/study
notebooks and notebook 09 explicitly configure full-volume postprocessing.

Saved ROI and brain masks are restored to the original image's voxel grid.
ROI files contain **original annotation IDs** with notebook defaults. Merged
D1/rat ventricles are separated with the historical PCA/lesion-side algorithm
before restoring IDs 3 (contra), 13 (ipsi), and 21 (third ventricle). MouseFull
preserves its separately predicted sides and maps its eleven channels back to
0, 1, 2, 3, 5, 6, 12, 13, 15, 16, 21. GT retains the original selected annotations;
prediction postprocessing never determines the ground-truth sides.

`InferenceConfig.original_label_ids=True` controls this export vocabulary.
Its `postprocessing: PostprocessingConfig` exposes brain-mask hole filling and
small-object removal, voxel thresholds (20,000 and 30,000), ventricle splitting,
centroid/voxel assignment, default hemisphere and lesion threshold. Morphology
acts on the binary brain head only; it does not overwrite or gate categorical
ROI predictions. Both masks receive the same original-geometry restoration.
Thresholds apply on the processed 0.1 mm grid before restoring geometry.
An unsplit merged class has no unique inverse: requesting original IDs without
separating it raises a configuration error.

The historical `default_hemisphere="right"` means the fitted-plane side containing
array origin, rather than anatomical right derived from the affine. This fallback
is used without a sufficiently large predicted lesion. Splitting requires at
least three predicted third-ventricle voxels when lateral ventricles are present;
the evaluator reports a scan-specific error if the midline cannot be fitted.
The algorithm is unchanged from the original repository.

Evaluation writes `prediction_labels.json` with source-GT, prediction and export
mappings and region names. The interactive viewer reads it, including sparse
original IDs such as 13 and 21. Older exports without this metadata retain their
recorded model-channel convention; rerun checkpoint evaluation to produce
postprocessed original-ID masks. Notebook 09 uses `VENTRICLE_MODE="auto"` to split
only the recognized merged vocabulary; all options remain explicit and editable.
Standalone checkpoint cells expose these same settings and override older saved
evaluation defaults without changing training weights or recovery artifacts.
Library defaults retain raw model IDs for old configurations; notebook defaults
explicitly select the full-volume workflow above. Offline metric rows retain subject, scan, region,
units, source dataset and annotation/empty-mask status.

Device warm-up is excluded from per-volume inference time. Prediction uses Keras
`predict`, which materializes NumPy results before the timer stops. Preprocessing
prediction and postprocessing durations are reported separately. Timings include sliding-window
reconstruction and are not isolated GPU-kernel benchmarks.

K-Fold's inner validation serves checkpoint selection. Independent outer
measurements have `DatasetRole="holdout"`; separate shared test/external cohorts
retain their own roles. Repeated external-animal predictions across folds do not
increase biological sample size. Reuse saved splits for seed-matched comparisons.

## Subsequent steps

Multiscale-input and fused deep-supervision
ablations require additional explicit builder switches; current component flags
do not represent those removals. nnU-Net training remains separate; its saved
results can enter the revision CSV contract with explicit canonical animal IDs,
region names, units and annotation statuses. This workflow performs no sensitivity
analysis. Animal bootstrap uncertainty conditions on supplied trained models;
it does not estimate longitudinal trajectories or training-seed uncertainty.
Notebook plots use Plotly's native VS Code renderer; the shared
runner also writes standalone HTML history artifacts for optional offline use.

## Saved validation and test patches

`ExperimentConfig.evaluation_patches` separates evaluation coverage from
`TrainingConfig.data_multiplier`. Training continues to draw fresh center-biased
crops. The notebooks use three evaluation patches per scan (five in rat K-Fold),
sampled once with their own seed and reused for every epoch and evaluation role.
The center-biased distribution matches training's default crop distribution.
`sampling="center", patches_per_scan=1` restores a single fixed center patch.

`evaluation_patches.json` saves the crop centers on each processed, padded grid,
padding widths, scan geometry and source/sampling contract. Runs save it before
fitting; studies save one shared collection and copy it into every member run.
Resumption loads the saved coordinates and rejects missing or modified multi-patch
collections. Older saved settings without the new field retain one center patch.
Changing evaluation sampling requires a new run. Training seeds and multipliers
do not change the evaluation collection.

`left_right_pairs=True` adds an original/reflected pair for each saved D1/rat crop.
The same anatomical LR reflection is applied to MRI, ROI and brain targets;
lesion/craniotomy-relative label identities remain unchanged. MouseFull keeps this
option disabled. Evaluation has no noise, blur, affine augmentation or shuffling,
and keeps the last incomplete batch. Full-volume evaluation remains uncropped.
Several patches do not increase animal counts; small scans may yield identical
coordinates when no additional crop positions exist.
