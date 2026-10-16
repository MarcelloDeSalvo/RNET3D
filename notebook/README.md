# Experiment notebooks

Each experiment has its own complete, editable configuration. Shared modules
own data loading, preprocessing, splits, fitting, recovery and evaluation.
Use the **rnet** kernel; new notebooks keep training disabled until enabled.
Your executed TBI notebook retains its existing settings and outputs.

For sequential execution without notebook cells, use
[run_revision_experiments.py](../scripts/run_revision_experiments.py).
The [suite guide](../docs/revision_suite.md) describes shared folds, a small draft
mode, recovery and the compact paper workbook with four sheets.

MRI workflows share `example/RARE/TBI_fm_19_49/Anat/TBI_fm_19_49_N4.nii.gz`
as their orientation reference. Its voxel axes are **R, I, A**: LR is axis 0,
IS is axis 1, and AP is axis 2. Reorientation preserves world coordinates; it
does not resize images to the reference shape. New configuration cells explicitly
set flipping to 0.5 for D1 training/studies and rat fine-tuning/CV, and 0.0 for MouseFull.
Validation/inference use no random flips. Ipsilateral/contralateral labels are
relative to the lesion/craniotomy and retain their IDs under a paired reflection.

Every training/study notebook exposes actual Keras callbacks in a separate
configuration cell and passes the list explicitly to training. Fixed-split runs
use LR reduction, TensorBoard histograms, early stopping and CSV logging.
Notebooks 01/03/05 also expose `K_FOLD_COMPATIBLE=False`: enable it to run every
outer fold with the same complete configuration. Studies 07/07f/08 expose the
same boolean to select CV. The corresponding callback cell then defaults to
LR reduction only; source CV checkpoints can be reused for matching MouseFull folds.
Standard K-Fold uses only LR reduction by default, with no early stopping; study
notebooks follow their selected `MODE`. Rat notebook 04 has no LR callback and
selects weights using training loss. Edit the constructors/list directly to change this.
Relative logging paths are scoped to each run/fold. Fresh callback copies prevent
state sharing across folds/seeds/variants, and saved `callbacks.json` settings
are checked on resume. The runner handles best/recovery checkpoints separately
and restores best weights before evaluation, so EarlyStopping itself uses
`restore_best_weights=False` to keep recovery weights and optimizer state aligned.

| Notebook | Responsibility | Status |
|---|---|---|
| [01_tbi_mouse_training.ipynb](01_tbi_mouse_training.ipynb) | D1 training and volume evaluation; optional full CV for reusable pretraining sources | Implemented; `K_FOLD_COMPATIBLE` switch |
| [02_tbi_mouse_kfold.ipynb](02_tbi_mouse_kfold.ipynb) | D1 grouped CV with separate checkpoint validation and outer holdouts | Implemented |
| [03_rat_finetuning.ipynb](03_rat_finetuning.ipynb) | D2 fine-tuning and independent evaluation; optional grouped outer CV | Implemented; `K_FOLD_COMPATIBLE` switch |
| [04_rat_kfold.ipynb](04_rat_kfold.ipynb) | Historical leave-one-rat-out fine-tuning with all longitudinal scans | Implemented |
| [05_mouse_full_training.ipynb](05_mouse_full_training.ipynb) | D1+D3 training and volume evaluation, eleven classes and partial labels; optional full CV | Implemented; `K_FOLD_COMPATIBLE` switch |
| [06_mouse_full_kfold.ipynb](06_mouse_full_kfold.ipynb) | D1+D3 grouped CV with anatomy-balanced outer folds and separate checkpoint validation | Implemented |
| [07_ablation.ipynb](07_ablation.ipynb) | Matched residual, attention, multiscale, head-fusion and downsampling ablations; optional loss/training rows | Implemented |
| [07f_ablation.ipynb](07f_ablation.ipynb) | MouseFull component/classifier ablations with eleven classes; optional v1/v3 loss comparison | Implemented |
| [08_unet_comparison.ipynb](08_unet_comparison.ipynb) | Matched 3DR-Net and standard 3D U-Net runs | Implemented |
| [09_inference_evaluation.ipynb](09_inference_evaluation.ipynb) | Stopped-checkpoint/export inference, physical metrics and interactive GT comparison | Implemented |
| [10_revision_results.ipynb](10_revision_results.ipynb) | Animal uncertainty, native patch tables, costs, coverage/failure review and manuscript exports | Implemented; independent D4 manual-volume analysis deferred |

The inference notebook accepts a stopped/completed `RUN_DIR` and selects its
saved best weights and cohort. Set `EVALUATION_ROLE="holdout"` for ordinary CV,
or `"validation"` for historical rat CV. Explicit experiment/checkpoint files
and external cohorts remain supported. Set `CHECKPOINT_KIND="model"` for an
exported model, or `"weights"` for fitting weights, before enabling
`RUN_EVALUATION`. Revision results consume explicit
saved per-scan CSVs, preserving annotation availability and metric statuses.
Set `METRIC_SOURCES` before enabling `RUN_ANALYSIS`; exports use a fresh folder.
Both workflows preserve biological animal grouping and use native Plotly displays.

The U-Net comparison reuses the study executor and the existing builders with
matched widths, preprocessing, loss, splits and seeds. It supports fixed splits
and grouped outer CV. Classifiers are disabled for both models because the
existing U-Net does not support that head; neither model's dropout is changed.

Rat CV retains the old animal split: train on four rats and evaluate the fifth.
For new runs its learning rate is fixed and the checkpoint uses training loss,
so the held-out rat does not control automatic scheduling or weight selection.
Its curves remain diagnostic; saved memberships/viewers retain the `validation`
role. Older runs retain their original validation-driven selection. Evaluation
uses five saved center-biased patches per scan. Every fold starts from the same
explicitly chosen mouse checkpoint and a fresh optimizer.

MouseFull preserves the old source-label/channel order and SHAM/TBI classifier.
Notebooks 05/06 expose TBI feature transfer through `USE_TBI_PRETRAINING`,
`TBI_RUN`, `TBI_FOLD` and `FREEZE_BACKBONE`. A saved epoch in a stopped source run
is sufficient; the committed best weights are selected automatically. Features
transfer, output heads remain fresh, and each target model uses a fresh optimizer.
Fresh fixed splits inherit source animal roles and fill the overall requested
fractions with new animals. Source selection-validation can be shared with
target selection-validation; fitted animals cannot become validation/test, and
source selection animals cannot become final test. Initialization ancestors are
checked too. Matching CV checkpoints are exposed through `TBI_FOLD_SOURCES`;
the planner reuses their source folds and splits additional animals. A shared
overlapping D1 checkpoint cannot provide independent full MouseFull CV.
Scratch training remains explicit. See [pretraining splits](../docs/pretraining_splits.md).
Unavailable classes follow normalized annotation metadata and use the corrected
logits-based loss. Confirmed cross-cohort duplicates can be grouped explicitly;
the notebook does not infer biological identity from similar original IDs.
Notebook 06 defaults to balancing complete/partial anatomical coverage across
new outer folds, excluding lesion availability. Set `FOLD_MODE="random"` for
unstratified generation; saved/resumed folds remain unchanged.

The diagnostic notebooks are:

- [check_interpolation_sensitiity.ipynb](check_interpolation_sensitiity.ipynb):
  a short D1 training comparison of MRI spline orders 0/1/2/3 with shared animals,
  seeds, evaluation patches and original model widths. Includes a grid audit,
  validation/test reports, T1w/T2w summaries and optional exports. Defaults to
  ten epochs/four training batches per epoch, with training disabled. See the
  [interpolation guide](../docs/interpolation_sensitivity.md).
- [check_preprocessing.ipynb](check_preprocessing.ipynb): one random scan, one
  preprocessing/augmentation per short cell, and an editable combined pipeline
  at the end. MRI/ROI/brain overlays share the RIA reference; changes in shape,
  spacing, intensities and categorical IDs are displayed after each operation.
  Blur/noise use the corrected native grid with explicit units; the combined
  example applies them before linear target resampling and affine augmentation.
- [check_postprocessing.ipynb](check_postprocessing.ipynb): a visible synthetic
  example by default, with normalized-scan and explicit-file options. Separate
  cells demonstrate ventricle separation, brain hole filling, small-object removal
  and restoration of original ROI IDs. The last cell uses the production helper.
- [check_partial_label_loss_gradient.ipynb](check_partial_label_loss_gradient.ipynb):
  version-2 foreground loss/gradient comparisons, missing-logit invariance,
  background competition and known-negative false-positive penalties.

Diagnostic plots use native Plotly zoom/hover. Set `VIEW_AXIS` and
`VIEW_POSITION`, then rerun a step to view another slice. Before/after panels
use the same relative position within their own grids; they do not imply a
shared physical slice after cropping. These notebooks do not train models or
overwrite dataset files. N4 is optional because normalized inputs already use N4.

Every full-volume workflow explicitly configures `PostprocessingConfig` and
restores original annotation IDs. Brain masks use historical hole filling and
small-object removal. D1/rat merged ventricles are separated before restoring
contra/ipsi IDs 3/13; MouseFull keeps its predicted sides. Ground truth retains
the selected source annotations. The settings are visible both in the complete
training configuration and in the independent best-checkpoint cell. Notebook 09
automatically recognizes the merged vocabulary and exposes the same controls.
Exported `prediction_labels.json` tells the viewer which IDs and names to use;
select an original annotation ID with `VIEW_REGION` or `CHECKPOINT_REGION`.
Previously saved masks without that file still display their old model IDs;
evaluate their checkpoint again for postprocessed original-ID exports.

Training/study notebooks 01–08 end with a saved-prediction viewer: MRI, mapped
ground truth and prediction share the reference orientation and class colors.
Native Plotly menus select scans/planes, and a slider selects slices. Use the
manifest row indices to load other animals, timepoints or folds; study notebooks
also select a variant/seed. Missing annotations remain explicit. `VIEW_RUN` can
inspect a matching existing run without fitting. Predictions must have been
saved during evaluation; no inference is repeated. Rat CV displays validation
predictions, while ordinary CV displays independent outer holdouts.

The final **Evaluate the best checkpoint after stopping training** cell in each
training/study notebook works independently of its interrupted `result`. Set
`CHECKPOINT_RUN`, choose a fold and enable `RUN_CHECKPOINT_EVALUATION`. After a
kernel restart, run only repository setup and that final cell. Saved settings
and cohorts are loaded automatically; no fit or optimizer restoration occurs.
When available, the committed selected-weight snapshot is used instead of a
possibly interrupted best-file overwrite. Predictions/metrics go to a fresh
folder outside the training run; checkpoint history and completion state are
unchanged. A saved epoch is required. Notebook 09 can reopen those outputs via
`VIEW_EVALUATION_DIR` without running inference again.

Checkpoint-only evaluation also explicitly uses the shared RARE reference,
including older checkpoints, as requested. Older weights trained with another
orientation receive a changed input convention; evaluation provenance records
both references. Reopening saved results can change the viewing orientation
without recomputing the stored metrics. Resume still requires the saved training
configuration; changing reference/augmentation settings defines a new run.

Validation/test patch coverage is configured separately from training repetitions
with `EvaluationPatchConfig`. Saved coordinates are shared across epochs and study
members. Each training notebook also exposes the three task loss weights. D1/rat
full-volume evaluation and checkpoint cells use centroid ventricle separation;
MouseFull preserves its already separate labels. Missing or small lesions retain
the documented voxel-wise fallback.

MouseFull 05/06/07f default to loss v1, with available empty SHAM lesion targets.
07f exposes reference, variant-specific and fold-specific TBI sources.
Architectural variants need matching source backbones; no-classifier can reuse
the reference. Final result sections display native patch measurements and
full-volume brain/region scores. Runs/studies export `evaluation_metrics.xlsx`
with separate sheets, alongside CSV tables. Checkpoint cells and 09 provide the
same measurements without fitting. See
[the configuration, leakage and evaluation guide](../docs/reviewer_workflow_steps_1_3.md).

Auxiliary classifier decisions are collected during the same full-volume pass.
Run/study/checkpoint workbooks include animal-level rates, confidence intervals
and confusion counts, plus independent SHAM lesion-volume diagnostics. Notebook
10 reads explicit `CLASSIFIER_FILES` and displays a Plotly confusion matrix without
retraining. Classifier decisions do not suppress lesions.
See [classifier evaluation](../docs/classifier_evaluation.md).

Revision notebook 10 also accepts explicit `PATCH_SOURCES` and `COST_SOURCES`.
Its export combines CSVs and `revision_tables.xlsx`, retaining patch, full-volume
and timing measurements as separate statistical units. Coverage and failure
review tables preserve missing/empty statuses and never remove low-scoring
animals. Input SHA256 hashes make the selected evidence auditable. The additional
baseline experiments and independent D4 manual-volume analysis remain deferred.
See [revision reporting](../docs/revision_reporting.md).

New full-volume workbooks also separate anatomy-only versus lesion/brain-inclusive
scores, condition-specific annotation coverage, acquisition/timepoint subgroups,
participant/split counts and structure-size measurements. Notebook 10 displays
these tables and a score-versus-GT-volume plot. Older result sources can attach
an explicit metadata manifest; unknown metadata is not guessed.

- [check_dataset_volumes.ipynb](check_dataset_volumes.ipynb): select any normalized
  Dx folder for native annotated volumes, SHAM ventricle ratios, distributions,
  and unclipped corrected/uncorrected lesion-volume exports.
- [check_inter_rater.ipynb](check_inter_rater.ipynb): explicit paired-reader mask
  manifests, spatial agreement and animal-level volume ICC, with the common RIA
  reference for paired displays. Independent-reader masks are required.

See [cohort and annotation reports](../docs/cohort_annotation_reports.md).
