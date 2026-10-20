# Original CV references and final-model fixed splits

References live in `D:/Other files/TESI_CNN/MANUSCRIPT_DATASET/`.
CV restores the exact D1 memberships supplied in
`C:/Users/MDesalvo/Downloads/D1_splits_final_original.json`. Only ID spelling and
the historical `val` role name are normalized: `D1_SUBJ_0001` becomes
`D1/SUBJ_0001`, and the original outer `val` becomes final `test`.
No D1 animal is reassigned or rebalanced across original folds.

| File | Purpose |
|---|---|
| `D1_splits_final_original.json` | Byte-for-byte copy of the supplied original file |
| `D1_splits_final.json` | Original five-fold D1 train/test memberships with canonical IDs |
| `MouseFull_5fold_balanced_splits.json` | Original D1 folds extended with balanced D3 holdouts |
| `D1_fixed_split_from_kfold_fold_4.json` | Approximately 70/20/10 final-model D1 split |
| `MouseFull_fixed_split_from_kfold_fold_4.json` | Matching final-model D1+D3 split |

CV files contain only `train` and `test` arrays. Fixed files contain separate
`train`, `validation` and `test` arrays. Companion metadata, balance CSVs and
scan-membership CSVs describe the actual roles and counts.

## Cross-validation and matched ablations

Every original animal appears in one outer test fold. All scans of an animal
stay together. With `USE_VALIDATION_FOR_SELECTION=False`, no held-out scan is
supplied to Keras validation, no early stopping or ReduceLROnPlateau is used,
and the final configured epoch is evaluated. A cosine schedule decreases LR
from the initial rate to `MINIMUM_LEARNING_RATE`, using only the epoch index.

The schedule is visible in each notebook's callback cell. Its parameters and
epoch budget are recorded in callback/configuration contracts; resuming uses
absolute epoch indexes and rejects changed settings. Recovery checkpoints are
still committed every epoch. Scientific test evaluation requires completion of
the declared budget. The historical managed filename `best.weights.h5` contains
the selected final weights in this mode, rather than a loss-selected epoch.

| Original fold | D1 train T1w / T2w | D1 test T1w / T2w | MouseFull train T1w / T2w | MouseFull test T1w / T2w |
|---|---:|---:|---:|---:|
| 0 | 66 / 23 | 20 / 3 | 135 / 23 | 37 / 3 |
| 1 | 66 / 23 | 20 / 3 | 135 / 23 | 37 / 3 |
| 2 | 69 / 21 | 17 / 5 | 137 / 21 | 35 / 5 |
| 3 | 71 / 19 | 15 / 7 | 140 / 19 | 32 / 7 |
| 4 | 72 / 18 | 14 / 8 | 141 / 18 | 31 / 8 |

D1 therefore uses 89/23 or 90/22 training/test animals. MouseFull uses 158/40
or 159/39. Modality differences between D1 folds intentionally match the original
plan. D3 contains only T1w scans and cannot rebalance D1's T2w holdouts.

D3's complete/partial anatomical coverage is spread across its five holdouts;
their order minimizes differences in combined coverage and total fold sizes.
Combined complete-anatomy test counts are 18/18/18/17/17. D1 includes nine
complete animals and D3 includes 79. Completeness requires every selected
anatomical annotation, excluding background and lesion. D1 roles are identical
in both plans, and extending the D1 file produces the saved MouseFull plan.

Set `USE_VALIDATION_FOR_SELECTION=True` to reserve separate inner validation
from each original training pool, use ReduceLROnPlateau, and restore the best
validation-selected checkpoint. Outer test memberships remain unchanged. This
option never turns the outer test into fitting-time validation.

## Final-model fixed splits

Notebook 01 defaults to the D1 fixed file; notebook 05 defaults to the MouseFull
fixed file. Original fold 4 was chosen before model evaluation because its
holdout contains eight T2w animals. Keeping its entire holdout would make a 10%
test fraction impossible. The approved fixed plan therefore retains a **subset**
of that holdout; the other animals return to the development pool.

Within each cohort, about 10% of all animals are selected for test from the
reference holdout, and about 20% are reserved for validation from the remaining
animals. The D1 subset keeps five T2w test animals. Selection also preserves
rare complete anatomy when available. The same D1 train/validation/test roles
are used in both final-model tasks.

| Task | Train | Validation | Test | Train T1w / T2w | Validation T1w / T2w | Test T1w / T2w |
|---|---:|---:|---:|---:|---:|---:|
| D1 | 79 | 22 | 11 | 63 / 16 | 17 / 5 | 6 / 5 |
| MouseFull | 139 | 39 | 20 | 123 / 16 | 34 / 5 | 15 / 5 |

These are 70.5/19.6/9.8% for D1 and 70.2/19.7/10.1% for MouseFull. The final
test intentionally contains a higher T2w proportion than the development pool.
It is a separate experiment from evaluation of the complete original CV folds.

Final-model fixed mode defaults to `USE_VALIDATION_FOR_SELECTION=True` for
validation-based scheduling and checkpoint selection. Setting it to False adds
those validation animals back to fitting, uses the cosine schedule, and evaluates
the last epoch; its test subset is unchanged. A fixed single-run checkpoint is
stored in `fold_0` regardless of the source reference fold. Consequently notebook
05 uses `TBI_FOLD=0` when optionally initializing from notebook 01's final model.

## Notebook defaults and saved runs

01/05 load their final-model fixed files. Enabling `K_FOLD_COMPATIBLE` clears the
fixed file and uses all original CV folds. Rat notebook 03 retains its separate
three-way fixed split; rat CV uses complete-animal holdouts. Notebooks 02/04/06,
07/07f/08 and interpolation default to the fixed-budget cosine/final-epoch
protocol. Fixed ablations use the complete original reference fold 4, not the
smaller final-model test subset, so they remain matched to that CV baseline.

Saved/resumed memberships remain authoritative. Changing selection policy,
schedule, budget or references requires a fresh run/study. Inference and result
notebooks evaluate the selected checkpoint using its saved roles. D4 remains
independent of all development splits.

## Backup and reproduction

All twelve previously active reference/audit files were verified and archived in
`split_backups/original_protocol_20261020_123316_385052/`. Its manifest contains
their SHA-256 hashes. Superseded fold-zero fixed files were removed from the
active folder only after successful backup. Earlier backups remain available.

```powershell
python scripts/regenerate_reference_splits.py
python scripts/regenerate_reference_splits.py --apply
python scripts/run_revision_experiments.py --plan-only
```

The regeneration script previews by default. `--apply` backs up existing files
before replacement. `--original-splits` selects another explicitly supplied
original plan. The launcher's defaults are `REFERENCE_FOLD=4`,
`USE_VALIDATION_FOR_SELECTION=False`, and `RUN_NAME="revision_original_two_way_v1"`.
