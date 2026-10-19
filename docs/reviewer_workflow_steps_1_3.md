# Reviewer workflow: configuration, pretraining and complete evaluation

## MouseFull configuration

Notebooks 05, 06 and 07f select `dice_ce_gradient_fix` version 1. SHAM lesion
masks are available empty annotations in every version; no absence flag is used.
All mouse training, CV and ablations start from scratch on shared reference
memberships. V3 remains an explicit experimental alternative. Network defaults
are unchanged. Notebook 05 alone exposes `USE_TBI_PRETRAINING=False` for a
separate fixed-mode final model. Its `TBI_RUN` and `TBI_FOLD` select committed
best source weights; transferred features get fresh output heads and optimizer.
That option is unavailable in mouse CV and ablations. Rat fine-tuning retains
its independent D1 source. Previews do not fit.

## Pretraining leakage and inventory

Before any fold starts, `pretraining_audit` checks source training against target
validation, holdout, independent test and external cohorts. Source
checkpoint-selection validation may be shared with target selection-validation;
it must remain excluded from final holdout/test/external cohorts. Initialization
ancestors are checked too. Fresh splits now inherit source roles through the
[common split helpers](pretraining_splits.md).
An unused source test cohort does not cause a pretraining overlap by itself.

Training requires saved source `experiment.json`, the selected fold's
`split.json`, and `subjects.csv`. Standalone weights without provenance cannot
establish independent evaluation and are rejected by training runners. Low-level
weight loading and checkpoint-only evaluation remain available.
Legacy source names need explicit `animal_ids` links when the target uses normalized
IDs; differing folder conventions cannot prove that cohorts are independent.

Matching uses canonical animal identities, image paths, and original IDs qualified
by dataset/subgroup. Explicit biological identity overrides are respected. Reused
original names across SHAM subgroups identify independent animals, as confirmed
by the user; names alone never merge them. Older manifests recover subgroup from
their saved data contract. `pretraining_audit.json` records source/protected animals
and file hashes; resume rejects changed source provenance.

When present, `mapping.xlsx` is cross-checked against normalized cohort/subject,
timepoint and original identity. Duplicate inventory scan entries fail preparation.
The current D1–D3 inventory has 218 scans from 203 normalized animals:

| Cohort | Scans | Animals |
| --- | ---: | ---: |
| D1 | 112 | 112 |
| D2 | 20 | 5 |
| D3 | 86 | 86 |

The selected source `d1_3drnet/RUN_20261013_125509_781588/fold_0`, audited against
the saved MouseFull split `mouse_full_3drnet/RUN_20261013_175412_728737`, used
**28 D1 animals from target validation/test**. Of these, 15 target validation
animals were fitted by the source and 10 target test animals contributed to source
fitting/selection; those assignments remain invalid. Three target validation
animals only contributed to source selection-validation and may be shared.
Existing files were not modified. A fresh automatic split preserves the source's
77/23/12 D1 roles and allocates D3 to reach 138/40/20 overall, passing the audit
without retraining the source or reserving extra animals. This remains an option
for notebook 05's fixed-mode final model. Mouse CV instead starts from scratch
with the frozen reference animal roles.

## Best-checkpoint measurements

Runners restore committed selected weights before final evaluation. Native patch
evaluation uses the compiled training outputs/objectives/metrics, every saved
deterministic patch, and retained final batches. No augmentation, optimizer updates
or postprocessing occur. Class-wise soft Dice comes from existing partial-label
metric accumulators; missing class supervision remains explicit.

Patch rows have `EvaluationStage=patch`, `Aggregation=cohort_pooled`, and
animal/scan/patch counts. They are cohort measurements, not independent animal
observations. Full-volume rows have `EvaluationStage=full_volume`,
`Aggregation=scan`, and `Head=regions` or `brain_mask`. Both heads share one
prediction per scan and the same physical grid. Metrics compare postprocessed
predictions with unpostprocessed valid GT. Original ROI IDs and unavailable/empty
statuses are preserved. Brain is evaluated as a separate binary mask.
Final validation remains a checkpoint-selection measurement.

Each run's `summary`, and each study folder, contains:

- `patch_metrics.csv`: native patch measurements.
- `full_volume_metrics.csv`: brain/region physical metrics.
- `evaluation_metrics.csv`: both stages with explicit aggregation columns.
- `evaluation_metrics.xlsx`: separate patch/full-volume sheets.

Training rows identify fold, seed, selected epoch (one-based), role and model/run.
Study exports also identify the variant. Completed folds reload saved patch rows.
Completed older folds without patch artifacts use separate checkpoint evaluation.

Final checkpoint cells and notebook 09 request patch evaluation explicitly.
`SavedModelEvaluationConfig.evaluate_patches` defaults False so inference-only
callers need not compile an old objective. Patch evaluation compiles for
measurement, never fits, and does not restore the training optimizer. Saved patch
coordinates are reused when the complete contract matches; otherwise fresh
deterministic coordinates are saved with their sampling provenance. A changed
orientation or annotation policy is therefore not misreported as exact replay.
Evaluation outputs are outside the source run.

Real reviewer experiments still require eligible sources and matched splits.
No full dataset training was launched for these implementation steps.
