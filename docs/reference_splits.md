# Shared reference animal splits

D1 training, K-Fold, ablation, U-Net comparison and interpolation notebooks,
plus notebook 05, default to `DATA_ROOT.parent / "D1_splits_final.json"`:

```text
D:/Other files/TESI_CNN/MANUSCRIPT_DATASET/D1_splits_final.json
```

The sequential launcher uses the same file. MouseFull notebooks 06/07f instead
use the dedicated balanced plan described below. Source JSON files are read only. Shared
parsing and cohort extension live in `src/experiments/splits.py`; the existing
selection helpers apply pretraining checks before returning memberships.

## Explicit three-way D1 reference

The file now stores five explicit `train`/`validation`/`test` folds covering
112 D1 animals, using canonical IDs such as `D1/SUBJ_0001`. Every animal from
each historical `val` list remains in the same fold's outer `test` list, in
the same order. No test animals were taken for validation. Inner validation
was sampled only from historical training, with seed `42 + fold` and fraction
`0.2` of that pool.

These are exactly the memberships the refactored loader already resolved from
the two-role file; the migration makes them visible in the JSON. Loading does
not split validation again or change roles when the generation seed/fraction
changes. Original bytes are archived under `split_backups/`; the companion
`D1_splits_final.metadata.json` records the backup, file digests and settings.

| Fold | Training | Inner validation | Outer test |
|---|---:|---:|---:|
| 0 | 71 | 18 | 23 |
| 1 | 71 | 18 | 23 |
| 2 | 72 | 18 | 22 |
| 3 | 72 | 18 | 22 |
| 4 | 72 | 18 | 22 |

All 112 animals are tested exactly once. Roles are disjoint within each fold,
and test animals never guide checkpoint selection or LR scheduling. The loader
still supports historical two-role inputs, but preserves explicit three-role
plans exactly, including order. Validation rejects repeated holdouts,
overlapping roles and incomplete coverage.

### Checkpoint selection and final test metrics

Mouse K-Fold notebooks 02/06 and CV ablations already enable an explicit
`ReduceLROnPlateau` monitoring inner `val_loss` (factor 0.5, patience 10,
minimum LR 1e-7). The runner saves the lowest inner-validation-loss weights
and reloads that checkpoint before native patch and full-volume outer-test
evaluation. CV uses its configured epoch budget without early stopping by default.
The launcher follows the same selection policy for its three-way CV jobs.

This is grouped five-fold evaluation with an internal validation holdout, not
nested CV. Freeze the experiment settings before inspecting outer-test results;
use only training/inner validation for tuning, scheduling and checkpoint selection.
Report those decisions and exact fold memberships. This addresses Reviewer 2's
test-driven selection concern; nested CV or a separate fixed tuning set was
described as the preferred design. Historical rat notebook 04 has no inner
validation and therefore retains training-loss selection and a fixed LR.

With the current inventory, reference fold zero has:

| Task | Train animals | Validation animals | Test animals |
|---|---:|---:|---:|
| D1 | 71 | 18 | 23 |
| MouseFull D1+D3 | 125 | 32 | 41 |

MouseFull extends the D1 plan with separate SHAM folds balanced by anatomical
annotation coverage, excluding lesion. Every D1 animal retains its role in
every fold. The same helpers, seed and fraction produce identical notebook and
launcher memberships. Rat splits remain separate and grouped by animal; D4
remains independent. A D1 file cannot define rat or D4 animal memberships.

## Balanced MouseFull reference for 06 and 07f

```text
D:/Other files/TESI_CNN/MANUSCRIPT_DATASET/MouseFull_5fold_balanced_splits.json
```

This separate three-role plan covers all 198 D1+D3 animals. It balances outer
tests by **dataset and complete/partial anatomical annotation coverage**;
background and lesion are excluded from the completeness definition. Completeness
requires every selected anatomical annotation in every registered scan. The
current inventory contains nine complete D1 animals and 79 complete D3 animals.

| Fold | Train | Inner validation | Outer test | Complete D1 in outer test | Complete D3 in outer test |
|---|---:|---:|---:|---:|---:|
| 0 | 126 | 32 | 40 | 2 | 16 |
| 1 | 126 | 32 | 40 | 2 | 15 |
| 2 | 126 | 32 | 40 | 2 | 16 |
| 3 | 127 | 32 | 39 | 2 | 16 |
| 4 | 127 | 32 | 39 | 1 | 16 |

Generation uses the shared `balanced_folds` helper, seed 42, and strata formed
from dataset plus anatomy coverage. Inner validation is a seeded random 20%
sample of each outer training pool; it is separate from fitting and outer tests.
Every animal is an outer test animal exactly once. Companion `.metadata.json`,
`.balance.csv` and `.membership.csv` files record settings, complete D1 IDs,
hashes, all role counts and scan memberships.

Notebook 06 loads all five folds with `FOLD_MODE="balanced"`. Notebook 07f
defaults to fixed fold zero from this exact plan; `K_FOLD_COMPATIBLE=True`
selects all five folds. All models start from scratch. Saved/resumed roles remain
authoritative, so changing a reference does not relabel an existing model's tests.
Use a fresh run/study for the new plan; 07f has a separate balanced study folder.

Historical D1 outer test memberships are unchanged. Their complete D1 counts are 3/2/2/1/1,
so preserving those exact roles would not achieve the requested 2/2/2/2/1 balance.
Notebook 05 and the launcher retain their existing defaults. To use this joint
plan for the entire launcher, pass `--folds-file` with the new path and a fresh
`--run-name`; its D1 experiments will then use the joint plan's D1 projection.

## Notebook settings

```python
REFERENCE_SPLITS: Path | None = DATA_ROOT.parent / "D1_splits_final.json"
REFERENCE_FOLD: int = 0
K_FOLD_COMPATIBLE: bool = False
```

Fixed training and studies use `REFERENCE_FOLD`; enabling K-Fold uses every fold
of the same reference plan. Set `K_FOLD_COUNT`/`training.k` to its actual fold
count for CV. `SPLIT_FILE` or `FOLD_FILE` can supply exact task-specific roles.
Resuming uses the run's saved membership; it does not regenerate or replace it.
Set `REFERENCE_SPLITS=None` to generate a new split instead.

Notebooks 07 and 07f default to fixed splits. Their result directories, plus the
comparison/interpolation study directories, include the reference fold to avoid
mixing new default runs with earlier studies. To resume an earlier study, retain
its original folder, mode, configuration and saved membership. Inference and
result notebooks read a model's saved splits, rather than assigning new test
animals to existing weights. Inspection notebooks do not generate splits.

All mouse experiments start from scratch, including every CV fold and ablation.
The shared plan coordinates animal roles; it does not share model weights. Rat
notebooks 03/04 fine-tune a chosen D1 checkpoint because D2 animals are independent
of D1. Notebook 05 alone retains optional mouse pretraining for a separate
fixed-mode final model, with the existing source leakage checks.

## Exported notebook 01 baseline

The fixed-split file is available beside the dataset:

```text
D:/Other files/TESI_CNN/MANUSCRIPT_DATASET/D1_fixed_split_from_kfold_fold_0.json
```

It contains canonical `train`, `validation` and `test` arrays with 71, 18 and 23
animals. Its companion `.metadata.json` records the source JSON digest, fold,
seed, inner-validation fraction and compatibility scope. Its role lists are
unchanged by the D1 reference migration and still match fold zero exactly.

Notebook 01 loads this file by default with `K_FOLD_COMPATIBLE=False`. Start a
new run with `RESUME_RUN=None`. Its checkpoint can initialize either rat notebook
without reserving extra D1 animals for D2. For the optional fixed-mode final
MouseFull model in notebook 05, select its run as `TBI_RUN`, enable
`USE_TBI_PRETRAINING`, and keep `TBI_FOLD=0` and `REFERENCE_FOLD=0`. Source
validation may be shared with target validation; target tests must remain
excluded from source fitting and checkpoint selection.

One D1 baseline cannot provide independent pretraining for all mouse outer folds:
every D1 animal is an outer test animal once. Notebook 02, MouseFull CV and
ablations therefore each train fresh models. Notebook 01's optional CV also
starts fresh in every fold; it clears the fixed `SPLIT_FILE` and uses the complete
reference plan. For another fixed reference fold, set `SPLIT_FILE=None` and
choose `REFERENCE_FOLD`. One D1 checkpoint can initialize every independent rat
fold, provided its choice does not use rat test performance.

## Sequential launcher

The visible settings are `REFERENCE_SPLITS`, `REFERENCE_FOLD` and
`ABLATION_MODE="fixed_split"`. Main D1/MouseFull/rat reference studies retain CV;
component and classifier/masking ablations fit once on the selected reference
fold. Every mouse fit starts from scratch; only rat fine-tuning loads the
predefined D1 reference fold. Summary
tables reuse only that CV reference fold as their ablation baseline; its other
folds remain in the main CV results. The baseline is not fitted twice. Fixed
results report one observed fold and do not acquire an invented fold SD.

```powershell
# Validate inputs and export the resolved plan without fitting.
python scripts/run_revision_experiments.py --plan-only

# Repeat all ablations across all reference folds instead.
python scripts/run_revision_experiments.py --ablation-mode cross_validation --run-name reference_cv --plan-only

# A small automatic draft without enforcing the 112-animal reference file.
python scripts/run_revision_experiments.py --quick --no-reference-splits --run-name draft_generated
```

`--reference-splits` and `--reference-fold` override the visible defaults.
`FOLDS_FILE`/`ANCHOR_SPLIT_FILE` take precedence over the default reference file.
When using an explicit reference or supplied fold file, a draft retains all its
animals and folds. It still reduces images, filters and fitting steps; it does
not silently alter the reference membership to reach the draft sample count.
Changed settings need a fresh suite name. Generated three-role plans, expanded
scan memberships and the source JSON fingerprint are exported with the suite.
