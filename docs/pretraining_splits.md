# Splits shared with pretraining

Pretraining is one stage of the complete training pipeline. Reserve validation
and test animals once, then keep their roles across stages. There is no need for
an additional permanent pretraining holdout.

## Fixed splits

`resolve_training_split` is shared by fixed-split notebooks and study selection.
`PreparedData.make_split()` also delegates to it.

For a fresh pretrained run, it reads the selected source's saved memberships
and scan manifest, matching canonical animals, original IDs qualified by
dataset/subgroup, and exact image paths. All timepoints inherit their animal's
role. Matching source training, validation and test animals keep those roles.
Additional animals fill the requested proportions of the complete target cohort
using the target split seed. Source cohorts unrelated to target animals retain
the ordinary splitting behavior.

```python
split = resolve_training_split(
    data=prepared, resume_run=RESUME_RUN, split_file=SPLIT_FILE,
)
```

For the existing TBI source `RUN_20261013_125509_781588` and MouseFull inventory,
the generated split is:

| Cohort | Training | Validation | Test |
|---|---:|---:|---:|
| D1: inherited source roles | 77 | 23 | 12 |
| D3: newly allocated roles | 61 | 17 | 8 |
| Total | 138 | 40 | 20 |

A read-only check against that source passed the audit with all 23 D1 validation
animals shared between stages. No training was launched or saved run modified.

Explicit `SPLIT_FILE` and resumed memberships remain authoritative and are never
regenerated. Unsafe memberships fail with a leakage error. If inherited roles
already exceed a requested fraction, they are retained; if too few independent
animals remain to fill the requested counts, feasible counts are used while
keeping all roles nonempty. An impossible independent test/validation split is
rejected. Inspect the displayed counts rather than assuming exact proportions.

For ablations, every selected source contributes constraints to the same split.
Conflicting source roles are rejected; architecture-matched sources should also
use matching animal memberships. Initialization ancestors contribute their fitting
and selection roles, with cycles/conflicts rejected.

## Audit policy

- Source training may initialize target training, but must not overlap target
  validation, test, outer holdout or external evaluation animals.
- Source selection-validation may be shared with target selection-validation.
  It must not overlap target test, outer holdout or external evaluation animals.
- Source test animals did not contribute to fitting or checkpoint selection;
  automatic splits keep them in the target test role.

Both fitting stages can use the same development validation set. Its scores
remain model-selection measurements; they are not independent final performance.
The same policy applies recursively to initialization ancestors. The runner
records `shared_validation_animals` in `pretraining_audit.json` when present.
Low-level audits keep strict protection unless `validation_ids` is explicitly
provided; training and study runners pass their exact selection-validation IDs.

## Cross-validation

### One notebook switch

Notebooks 01, 03 and 05 expose:

```python
K_FOLD_COMPATIBLE: bool = False
K_FOLD_COUNT: int = 5
CV_VALIDATION_FRACTION: float = 0.2
```

`False` retains their fixed-split behavior. `True` runs every outer fold through
the existing CV runner, saving one selected checkpoint per fold. The common
`resolve_training_selection` returns either the fixed split or the complete
fold set; generator previews use the first fold. Source CV runs use a separate
model-group name ending in `_kfold`. No extra permanent test cohort is reserved.

The callback cell remains explicit: fixed mode retains LR reduction, TensorBoard,
early stopping and CSV logging; CV defaults to LR reduction only. Managed best
and recovery checkpoints remain active in both modes. Result viewers and
checkpoint evaluation default to `test` in fixed mode and `holdout` in CV mode.
Override the displayed role when inspecting a run from a different mode.

Enable the switch in notebook 01 and run all folds to create a TBI source usable
for full MouseFull CV. Then choose that source `TBI_RUN` in notebook 05 and enable
its switch, or use notebook 06. MouseFull selects same-index committed source
checkpoints automatically unless `TBI_FOLD_SOURCES` supplies explicit overrides.
Notebook 06 exposes `AUTO_MATCH_TBI_FOLDS=True` for this automatic selection.
Source runs without every required committed checkpoint fail before fitting.

The same boolean controls `MODE` in studies 07, 07f and 08. With it enabled, every
variant/seed trains all folds. Notebook 07f automatically maps variant-specific
source CV runs to target folds when explicit variant fold overrides are absent.

`SPLIT_FILE` applies to fixed mode; `FOLD_FILE` applies to CV. Resume retains exact
saved memberships and rejects a mode mismatch. Use a fresh run when changing modes.
Notebook 03's optional CV has independent inner validation and outer holdouts;
historical rat notebook 04 keeps its original leave-one-rat-out selection protocol.

### Matching source memberships

`make_pretrained_cv_splits` reads the source selected for each target fold.
Matching source animals keep their training, selection-validation and outer
holdout roles. Additional target animals receive grouped outer folds; notebook
06 optionally balances their anatomical annotation coverage, excluding lesion.
The complete fold set must evaluate every CV animal exactly once.

For full MouseFull CV, the recommended source is a D1 CV run with one matching
checkpoint per fold. In notebook 06, after choosing `TBI_RUN` from notebook 02:

```python
TBI_FOLD_SOURCES = {
    index: (TBI_RUN, index)
    for index in range(training.k)
}
```

The notebook already converts this map into initialization settings. Fresh folds
are then generated from those source memberships. A shared overlapping D1
checkpoint is rejected because it cannot provide independent D1 holdouts in all
folds. A source trained on a completely independent cohort can be shared.
Explicit/resumed fold files are not regenerated and remain audited before fitting.

Notebook 07f accepts corresponding source maps for each architecture variant;
their memberships must agree within each target fold. Historical rat CV retains
its original leave-one-rat-out selection protocol. Independent mouse pretraining
does not force rat animals into mouse source roles.

Existing suitable D1 fold checkpoints can be reused. Otherwise train a D1 source
inside each target fold's training partition, using its inner validation for
checkpoint selection, then fine-tune MouseFull with those same roles. This adds
computation but no extra permanent held-out cohort. Training from scratch remains
an explicit alternative. Cross-validating only D3 with a shared D1 checkpoint
would evaluate a narrower target population and must be described accordingly.

Automated checks use synthetic sources and tiny datasets. They cover membership
inheritance, annotation balancing, longitudinal grouping, source renumbering,
saved splits, conflicting sources and rejection of shared-checkpoint leakage.

The mode switch does not alter preprocessing or metric calculations. The shared
full-volume evaluator now measures nonorthogonal grids using their full physical
affine, without an additional resampling step. See [cohort reports](cohort_annotation_reports.md).
