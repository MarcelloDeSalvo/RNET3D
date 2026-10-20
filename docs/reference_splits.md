# Shared reference animal splits

The reference files live in
`D:/Other files/TESI_CNN/MANUSCRIPT_DATASET/`. All mouse experiments preserve
the cohort's modality proportions, retain every animal, and keep every scan of
an animal together. There is no oversampling or forced 50/50 T1w/T2w subset.

| File | Purpose |
|---|---|
| `D1_splits_final.json` | Five explicit D1 train/validation/test folds |
| `D1_fixed_split_from_kfold_fold_0.json` | Exact D1 fold-zero roles for notebook 01 |
| `MouseFull_5fold_balanced_splits.json` | Five explicit D1+D3 folds for 06/07f |
| `MouseFull_fixed_split_from_kfold_fold_0.json` | Exact MouseFull fold-zero roles |

Fixed runs and ablations use reference fold zero by default. CV uses all five
folds. D1 animals have **identical roles in both reference plans**, so notebook
05 and the launcher can extend the D1 reference with SHAM animals and recover
the same MouseFull assignments as notebooks 06/07f. Model initialization remains
independent: mouse experiments start from scratch except optional pretraining
for notebook 05's separate final model.

## Modality and anatomy balance

The 2026-10-20 regeneration stratifies by dataset, modality and complete/partial
anatomical coverage, with seed 42. Generated folds 0 and 1 are then exchanged
so reference fold zero has six T2w test animals. All three roles move together;
the recorded generation order is `[1, 0, 2, 3, 4]`.
Completeness requires every selected anatomical
annotation in every scan of the animal; background and lesion are excluded.
There are nine complete D1 animals and 79 complete D3 animals.

Outer tests distribute each stratum across the five folds. Inner validation
uses proportional allocation from the remaining training pool, reserving 20%
rounded up within each cohort. Rare strata retain fitting and validation examples
when feasible; integer counts prevent every proportion from being exact.

The D1 inventory contains 86 T1w and 26 T2w animals:

| Fold | Train T1w / T2w | Validation T1w / T2w | Test T1w / T2w | Complete D1 in test |
|---|---:|---:|---:|---:|
| 0 | 55 / 16 | 14 / 4 | 17 / 6 | 2 |
| 1 | 54 / 17 | 14 / 4 | 18 / 5 | 2 |
| 2 | 55 / 17 | 14 / 4 | 17 / 5 | 2 |
| 3 | 55 / 17 | 14 / 4 | 17 / 5 | 2 |
| 4 | 55 / 17 | 14 / 4 | 17 / 5 | 1 |

MouseFull adds 86 T1w SHAM animals, preserving every D1 assignment:

| Fold | Train T1w / T2w | Validation T1w / T2w | Test T1w / T2w | Complete D3 in test |
|---|---:|---:|---:|---:|
| 0 | 110 / 16 | 28 / 4 | 34 / 6 | 16 |
| 1 | 108 / 17 | 28 / 4 | 36 / 5 | 16 |
| 2 | 110 / 17 | 28 / 4 | 34 / 5 | 16 |
| 3 | 110 / 17 | 28 / 4 | 34 / 5 | 16 |
| 4 | 110 / 17 | 28 / 4 | 34 / 5 | 15 |

Every animal occurs in exactly one outer test. Training, validation and test are
disjoint within each fold. The two fixed files match their corresponding fold
zero exactly: D1 has 71/18/23 animals; MouseFull has 126/32/40 animals.

Each CV file has `.metadata.json`, `.balance.csv` and `.membership.csv`
companions recording generation settings, hashes, complete D1 IDs, per-role
modality/coverage counts and scan memberships. Fixed-file metadata records the
source CV filename, digest and fold index.

## Backups and regeneration

Before replacement, all eight previous reference and companion files were copied
byte-for-byte into:

```text
split_backups/modality_balanced_20261020_101808_358542/
```

The subsequent fold swap backed up all twelve current reference/audit files to
`split_backups/modality_balanced_20261020_102828_555561/` before replacement.
Backup manifests record SHA-256 hashes. Earlier backup folders remain
available. The new references intentionally replace historical memberships to
improve modality balance; no previously saved model/run memberships are changed.

To preview or deliberately regenerate these references without model fitting:

```powershell
python scripts/regenerate_reference_splits.py
python scripts/regenerate_reference_splits.py --apply
```

The script uses the normalized loader and shared split helpers. It verifies all
backups before replacing references, stages new files, and restores originals if
a replacement fails. Each invocation with `--apply` creates a fresh backup folder.

## Notebook and launcher settings

Most mouse notebooks, notebook 05 and the launcher use:

```python
REFERENCE_SPLITS: Path | None = DATA_ROOT.parent / "D1_splits_final.json"
REFERENCE_FOLD: int = 0
K_FOLD_COMPATIBLE: bool = False
```

Notebooks 06/07f instead use `MouseFull_5fold_balanced_splits.json`; their D1
roles match the D1 reference. Notebook 01 loads the exported D1 fixed file in
fixed mode. Set `SPLIT_FILE=None` and choose `REFERENCE_FOLD` to select another
fixed fold. K-Fold mode uses every reference fold and requires the matching
`training.k`/`K_FOLD_COUNT`.

Explicit split/fold files and resumed memberships take precedence. Loading a
reference preserves its explicit roles, independent of later generation seeds
or fractions. **Use fresh runs/studies for the updated reference**: existing
checkpoints retain their original fitting and holdout animals. Saved-model
evaluation reads their saved splits rather than assigning new test animals.
Set `REFERENCE_SPLITS=None` to generate fresh roles instead of using the plan.

The launcher's new default name is `revision_modality_balanced_v2`. Its main
D1/MouseFull studies retain CV, while architecture and classifier/masking
ablations default to fixed reference fold zero. Their summary baseline uses
only the matching reference fold. A fixed study has one observed fold and no
invented fold SD.

```powershell
python scripts/run_revision_experiments.py --plan-only
python scripts/run_revision_experiments.py --ablation-mode cross_validation --run-name modality_cv --plan-only
python scripts/run_revision_experiments.py --quick --no-reference-splits --run-name draft_generated
```

`--reference-splits`/`--reference-fold` override the visible defaults;
`FOLDS_FILE`/`ANCHOR_SPLIT_FILE` take precedence over the reference. Drafts with
an explicit reference retain all its animals and folds while reducing model and
training budgets. Plans export exact roles, scan memberships and input digests.

## Checkpoint selection and independent cohorts

Mouse CV uses `ReduceLROnPlateau` monitoring inner `val_loss` and reloads the best
inner-validation checkpoint before native-patch and full-volume outer-test
evaluation. CV uses its configured epoch budget without early stopping by default.
This is grouped CV with an internal validation holdout, not nested CV. Outer
tests must not guide tuning, LR changes or checkpoint selection.

Rats retain their separate grouped protocol; D4 remains independent. Rat notebook
04 uses a fixed LR and epoch budget without inner validation and evaluates final
weights on each held-out rat. Rat notebooks can fine-tune a predefined D1
checkpoint because D2 animals are independent of D1. Optional MouseFull
pretraining in notebook 05 still enforces the source/target leakage checks.
