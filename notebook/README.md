# Experiment notebooks

Use the **rnet** kernel. Every training notebook exposes its complete configuration,
actual Keras callbacks, data preview, recovery settings and evaluation options.
Shared modules implement loading, preprocessing, training and reporting.

## Training and split policy

`USE_VALIDATION_FOR_SELECTION` is visible in every notebook that trains models:

- **False:** train without validation passes, use a preset cosine learning-rate
  schedule, and evaluate final-epoch weights after the entire configured budget.
  ReduceLROnPlateau and early stopping are disabled. Test animals never select
  epochs or learning rates.
- **True:** reserve separate validation from development animals, reduce LR using
  `val_loss`, and evaluate the best validation checkpoint. Fixed mode also enables
  early stopping. The outer test never becomes fitting-time validation.

`INITIAL_LEARNING_RATE`, `MINIMUM_LEARNING_RATE` and `training.epochs` define the
cosine schedule. Its parameters are saved and checked on resume. Callback lists
remain explicit; the runner additionally manages selected/recovery checkpoints.
Recovery works every epoch, including in fixed-budget mode. Scientific test
evaluation of that mode requires the complete budget.

D1 CV restores the exact memberships from the supplied original split. MouseFull
extends them with D3 holdouts balanced for anatomy coverage. These defaults use
only training/test animals. The final-model fixed files instead target 70/20/10,
using a test subset of original fold 4. See [reference splits](../docs/reference_splits.md)
for exact memberships, modality counts, annotation coverage, backups and filenames.

## Main notebooks

| Notebook | Default role and split |
|---|---|
| [01_tbi_mouse_training.ipynb](01_tbi_mouse_training.ipynb) | D1 final model: 79 train / 22 validation / 11 test; validation selection enabled |
| [02_tbi_mouse_kfold.ipynb](02_tbi_mouse_kfold.ipynb) | Original D1 five-fold CV: 89–90 train / 22–23 test; cosine schedule and final epoch |
| [03_rat_finetuning.ipynb](03_rat_finetuning.ipynb) | D2 fixed fine-tuning: 3 train / 1 validation / 1 test rat; choose a mouse checkpoint first |
| [04_rat_kfold.ipynb](04_rat_kfold.ipynb) | Leave one complete rat out: 4 train / 1 test; cosine schedule and final epoch |
| [05_mouse_full_training.ipynb](05_mouse_full_training.ipynb) | MouseFull final model: 139 train / 39 validation / 20 test; validation selection enabled |
| [06_mouse_full_kfold.ipynb](06_mouse_full_kfold.ipynb) | Original D1 plus balanced D3 CV: 158–159 train / 39–40 test; cosine schedule and final epoch |
| [07_ablation.ipynb](07_ablation.ipynb) | D1 matched component ablations on original fold 4: 90 train / 22 test |
| [07f_ablation.ipynb](07f_ablation.ipynb) | MouseFull component/classifier/masking controls on matching fold 4: 159 train / 39 test |
| [08_unet_comparison.ipynb](08_unet_comparison.ipynb) | Matched 3DR-Net/U-Net comparison on original D1 fold 4 |
| [09_inference_evaluation.ipynb](09_inference_evaluation.ipynb) | Explicit saved run/checkpoint; no new split or training |
| [10_revision_results.ipynb](10_revision_results.ipynb) | Explicit saved metric, classifier, patch and timing sources; manuscript tables |

01/03/05 and studies 07/07f/08 expose `K_FOLD_COMPATIBLE`. Enabling it runs every
outer fold. Mouse CV and ablations start from scratch; rat folds reload the same
predefined independent mouse source. Notebook 05 alone retains optional TBI
backbone initialization for its separate fixed-mode final model. This requires
matching fixed memberships and the existing source/target leakage checks.

Study mode `fixed_split` means one original reference CV fold; it does not use
the smaller final-model test subset. All variants/seeds within a study reuse its
animal memberships and deterministic evaluation patches.

## Inspection and reporting

| Notebook | Purpose and input |
|---|---|
| [plot_training_results.ipynb](plot_training_results.ipynb) | Plotly loss/metric curves, combined fold validation curves when available, condition/modality boxplots, hover IDs/strain, exports |
| [check_interpolation_sensitiity.ipynb](check_interpolation_sensitiity.ipynb) | Matched D1 interpolation study on original reference fold 4; configurable orders and budget; no validation ranking when validation is disabled |
| [check_preprocessing.ipynb](check_preprocessing.ipynb) | One scan, one preprocessing/augmentation per cell, editable combined pipeline |
| [check_postprocessing.ipynb](check_postprocessing.ipynb) | Brain cleanup, ventricle splitting and restoration of original IDs on an explicit example |
| [check_partial_label_loss_gradient.ipynb](check_partial_label_loss_gradient.ipynb) | Synthetic gradient/masking checks; no animal split |
| [check_attention.ipynb](check_attention.ipynb) | Saved-model attention, classifier Grad-CAM and segmentation Grad-CAM |
| [check_feature_space.ipynb](check_feature_space.ipynb) | Saved encoder features, animal/scan aggregation and PCA/optional UMAP |
| [check_dataset_volumes.ipynb](check_dataset_volumes.ipynb) | Selected normalized cohort, native annotation volumes, SHAM ventricle ratios and lesion-volume corrections |
| [check_inter_rater.ipynb](check_inter_rater.ipynb) | Explicit paired-reader masks, spatial metrics and volume agreement/ICC |

Inspection notebooks do not define a training split. Saved-model exploration
retains the selected model's provenance; it is not an independent test simply
because a scan is displayed. Notebook 09 defaults to `test`; choose `holdout` for
CV results. No saved run adopts newly generated reference memberships.

## Shared image and evaluation conventions

The orientation reference is
`example/RARE/TBI_fm_19_49/Anat/TBI_fm_19_49_N4.nii.gz`, with R/I/A voxel axes.
MRI uses linear interpolation; categorical masks use nearest neighbour. Paired
spatial transforms keep MRI and masks aligned. D1/rats retain LR augmentation;
MouseFull disables flipping. Training uses center-biased random crops; evaluation
coordinates are deterministic and saved separately from the training multiplier.

Full-volume evaluation explicitly postprocesses and restores original annotation
IDs. D1/rat merged ventricles use centroid splitting; MouseFull predicts its
separate sides directly. Ground truth is never postprocessed from predictions.
Thresholds and morphology settings remain visible in each configuration.

MouseFull uses gradient-fix loss v1 and available empty lesion targets for verified
SHAM scans. Reports include native-patch and postprocessed full-volume metrics,
brain/region summaries, anatomy-only coverage, condition/modality/timepoint
subgroups and auxiliary-classifier confusion matrices when that head is active.
No validation rows or curves are invented for two-way training.

For sequential execution use [run_revision_experiments.py](../scripts/run_revision_experiments.py).
The [suite guide](../docs/revision_suite.md) explains planning, draft execution,
recovery and compact paper exports. Existing user configuration edits and stored
notebook outputs are retained; rerun configuration and selection cells for a fresh
experiment after changing the protocol.
