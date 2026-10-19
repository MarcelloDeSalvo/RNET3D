# Shared splits and optional pretraining

All mouse training, cross-validation and ablations start from scratch on the
same frozen animal memberships. Notebook 01 uses the derived fixed fold-zero
file; mouse CV uses every explicit three-role fold of `D1_splits_final.json`.
The D1 reference retains every historical outer test animal; validation comes
only from the original training pools.
See [reference splits](reference_splits.md) for filenames and exact counts.
Sharing memberships does not share weights.

## Where pretraining remains available

Rat notebooks 03/04 fine-tune an explicitly chosen D1 checkpoint. Their D2
animals are independent of D1, so one mouse checkpoint can initialize every rat
fold. Choose that source before inspecting rat test results. The sequential
launcher uses the predefined `RAT_SOURCE_FOLD` of its D1 reference CV run.

Notebook 05 alone retains optional TBI backbone initialization for a separate
fixed-mode final MouseFull model:

```python
K_FOLD_COMPATIBLE: bool = False
USE_TBI_PRETRAINING: bool = False
TBI_RUN: Path | None = None
TBI_FOLD: int = REFERENCE_FOLD
FREEZE_BACKBONE: bool = False
```

Set `TBI_RUN` and enable `USE_TBI_PRETRAINING` to use it. Features transfer;
output heads and optimizer start fresh. The model-group name gains
`_tbi_pretrained` to separate it from the scratch reference. CV rejects this
option. Notebooks 06/07f have no mouse source-checkpoint settings, and all mouse
jobs in the sequential launcher, including deployment refits, start from scratch.

A single D1 checkpoint cannot provide independent initialization for all mouse
outer folds: every D1 animal is a test animal once. Corresponding source-fold
support remains in the core for historical run contracts and explicit custom
workflows, but it is not the review experiment protocol.

## Fixed-mode membership and provenance

`resolve_training_split` is shared by fixed-split notebooks and study selection.
`PreparedData.make_split()` also delegates to it. For a fresh pretrained run,
it reads the selected source's saved memberships and scan manifest. Matching
canonical animals, original IDs qualified by dataset/subgroup, and exact image
paths retain their source roles. Every timepoint stays with its animal.
Additional animals fill the requested proportions using the target split seed.
A completely independent source cohort does not constrain target memberships.

```python
split = resolve_training_split(
    data=prepared, resume_run=RESUME_RUN, split_file=SPLIT_FILE,
)
```

Explicit split files, reference plans and resumed memberships remain
authoritative; they are never silently replaced to accommodate a checkpoint.
Incompatible source/target roles fail the audit. Without explicit memberships,
inherited source roles may make requested fractions approximate. The planner
keeps all roles nonempty and rejects an impossible independent evaluation.
Inspect displayed memberships before fitting. Resume uses the target run's
saved fitting state rather than initializing it again.

## Audit policy

- Source fitting animals must not overlap target validation, outer holdout,
  independent test or external evaluation animals.
- Source checkpoint-selection validation may be shared with target selection
  validation, but must exclude target test, outer holdout and external animals.
- Source test animals did not contribute to fitting or selection. Automatic
  fixed splits retain them as target tests.

Both fitting stages may use the same development validation set. Its scores
remain model-selection measurements. Audits check initialization ancestors too,
rejecting cycles/conflicts, and record shared validation animals and source hashes
in `pretraining_audit.json`. Training requires saved source `experiment.json`,
`subjects.csv` and fold memberships; weights alone cannot establish independence.
Legacy identities require explicit links where normalized IDs cannot be matched.
Similar names do not prove biological identity or independent cohorts.

## Cross-validation and callbacks

`K_FOLD_COMPATIBLE=True` in notebooks 01/03/05 runs every outer fold. Mouse
folds start fresh; rat folds reload their independent mouse source with a fresh
optimizer. The same switch selects CV for studies 07/07f/08. Fixed ablations
are the default, using reference fold zero. `SPLIT_FILE` applies to fixed mode;
`FOLD_FILE` applies to CV. Resume keeps exact saved memberships and rejects a
mode mismatch. Use a fresh run when changing mode or initialization.

Callback cells remain explicit. Fixed mode uses LR reduction, TensorBoard,
early stopping and CSV logging; standard CV uses LR reduction only. Best and
recovery checkpoints are managed in both modes. Viewers default to `test` in
fixed mode and `holdout` in CV; override the role when inspecting older runs.
Notebook 03's grouped CV has independent inner validation and outer holdouts.
Historical rat notebook 04 holds out every scan of one rat, with no LR callback
and training-loss checkpoint selection; its saved diagnostic role is `validation`.

The core still tests optional per-fold source mappings, longitudinal grouping,
annotation balancing, inherited memberships, conflicting sources and leakage
rejection. Those capabilities do not enable pretraining in the scratch mouse
notebooks. Neither the protocol switch nor the split filename changes
preprocessing, loss computation or evaluation metrics.
