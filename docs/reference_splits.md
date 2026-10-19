# Shared reference animal splits

The mouse training, K-Fold, ablation, U-Net comparison and interpolation notebooks
default to `DATA_ROOT.parent / "D1_splits_final.json"`:

```text
D:/Other files/TESI_CNN/MANUSCRIPT_DATASET/D1_splits_final.json
```

The sequential launcher uses the same file. The source JSON is read only. Shared
parsing and cohort extension live in `src/experiments/splits.py`; the existing
selection helpers apply pretraining checks before returning memberships.

## Interpretation of the existing file

The file contains five historical `train`/`val` folds covering 112 D1 animals.
`D1_SUBJ_0001` is translated to the canonical `D1/SUBJ_0001` spelling. Each saved
`val` group is retained as that fold's independent outer test group. Inner
validation is drawn from the saved training pool only, with seed `42 + fold`
and fraction `0.2` of that pool. The remaining pool is used for fitting.

This preserves the original outer tests and avoids using those animals for
checkpoint selection, LR scheduling or early stopping. A supplied three-role
`train`/`validation`/`test` plan is retained exactly, including list order.
Validation rejects repeated holdouts, overlapping roles and incomplete coverage.

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

An old TBI checkpoint may be incompatible with the new reference roles. The
leakage guard remains active: it must not have fitted the target validation/test
animals or selected its checkpoint using target test animals. Train compatible
D1 source checkpoints with notebook 01 in K-Fold mode or notebook 02, then use
their matching folds for MouseFull. A changed default does not make old weights
independent of previously seen animals.

## Sequential launcher

The visible settings are `REFERENCE_SPLITS`, `REFERENCE_FOLD` and
`ABLATION_MODE="fixed_split"`. Main D1/MouseFull/rat reference studies retain CV;
component and classifier/masking ablations fit once on the selected reference
fold. Each fixed MouseFull control loads the matching D1 source fold. Summary
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
