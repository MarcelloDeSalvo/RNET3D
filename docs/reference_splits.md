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
seed, inner-validation fraction and compatibility scope. The original reference
file remains unchanged.

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
