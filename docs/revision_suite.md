# Sequential revision experiments

Edit the variables at the top of
[`scripts/run_revision_experiments.py`](../scripts/run_revision_experiments.py).
The script uses the same data preparation, training, checkpoint recovery,
initialization and evaluation code as the notebooks. It never executes notebook
cells, chooses a latest run, or launches a collaborator's nnU-Net training.

From the repository root, with the `rnet` environment active:

```powershell
# Validate the real inventory and save the complete full-study plan. No fitting.
python scripts/run_revision_experiments.py

# Run every stage on tiny real-data copies, with one epoch and one fitting batch.
python scripts/run_revision_experiments.py --quick

# Start a new full study without an earlier saved plan.
python scripts/run_revision_experiments.py --run

# If the full plan already exists, or training was interrupted:
python scripts/run_revision_experiments.py --run --resume

# Resume a draft. Completed folds are reloaded rather than fitted again.
python scripts/run_revision_experiments.py --quick --resume

# Rebuild the compact paper workbook from saved results, without fitting.
python scripts/run_revision_experiments.py --summarize-only
python scripts/run_revision_experiments.py --quick --summarize-only
```

Use `--output-root`, `--run-name`, `--data-root` and `--reference` to choose paths.
`--quick --plan-only` saves a draft plan without copying images or fitting.
An existing suite needs `--resume`. Changed settings, callbacks, reference,
metadata, source-file stamps or split memberships require a new run name.
Draft and full runs occupy different directories and cannot be resumed into
one another. Scientific training is never launched by running the file without
an execution flag.

## Early configuration

The launcher exposes animal selections, split files, seeds, epoch budgets,
interpolation orders, selected ablations, source-fold choice, task switches,
prediction saving and the exact LR/CSV callbacks. CV uses LR reduction driven by
inner validation, without early stopping. Interpolation pilots use a fixed LR;
release refits use a fixed duration and no adaptive callbacks. The full task
defaults retain the original filter widths, v1 MouseFull loss, RIA reference,
linear MRI interpolation and nearest-neighbour masks. Full-volume evaluation
uses original labels and the configured historical postprocessing, with centroid
ventricle separation for D1/rats.

`RevisionSuiteConfig` also exposes learning rate, patch sizes, spacing, batch
sizes, full/draft filter widths and uncertainty settings. Do not change these
after examining outer-test or D4 scores. Existing historical test-guided choices
must still be described honestly; a new runner cannot erase their history.

## Shared splits and initialization

Use canonical animal IDs such as `D1/SUBJ_0001`. A selected rat includes all its
registered scans. `D1_IDS`, `SHAM_IDS`, `RAT_IDS` and `D4_IDS` default to all animals
in each cohort. `ANIMAL_IDS` links only expert-confirmed biological identities;
reused unqualified names are never assumed to identify the same animal.

With no supplied split file:

1. D1 and D3 are each balanced by complete versus partial anatomical annotation,
   excluding lesion. Small strata are distributed rather than duplicated.
2. Every animal appears in one outer holdout. Each fold has separate fitting
   and inner-validation animals.
3. The MouseFull plan combines those cohort assignments. Its D1 projection is
   exactly the TBI pretraining plan. The nine complete TBI animals are therefore
   spread across folds, rather than relying on complete SHAM cases for balance.
4. Rat folds leave one animal out, reserve inner validation from the remainder,
   and retain all timepoints together.

`FOLDS_FILE` accepts an existing repository D1 or joint D1+D3 CV `splits.json`:

```json
[
  {"train": ["D1/animal_a"], "validation": ["D1/animal_b"], "test": ["D1/animal_c"]}
]
```

The illustrative fragment needs the remaining folds and actual canonical IDs.
Every selected animal must be an outer test animal exactly once. Two-role
historical validation-only folds, overlap, unknown IDs, missing coverage or a
projection with empty D1 fitting/validation/test roles are rejected. A supplied
D1 plan remains exact; the suite adds compatible D3 folds. A supplied joint plan
remains exact and its D1 projection supplies pretraining. Its fold count takes
precedence over generation settings.

Alternatively, `ANCHOR_SPLIT_FILE` accepts an existing fixed
`{"train": [...], "validation": [...], "test": [...]}` D1 or joint split. Its
memberships become fold zero exactly; the suite creates the remaining outer
holdouts from animals outside that first test set. Later folds have their own
inner validation. Such anchored folds can have unequal sizes. You cannot supply
both split sources. `RAT_FOLDS_FILE` optionally supplies an independent rat plan.

Each MouseFull fold loads the same-index D1 checkpoint. The normal recursive
pretraining audit rejects source fitting on target inner-validation animals and
any source fitting/selection on target holdouts. Shared source/target inner
validation is allowed and recorded. Source graphs are checked before transfer.
Architecture ablations are on D1 and start from scratch, so they do not require
another family of MouseFull source checkpoints.

These checks establish declared identity/role separation. They cannot detect
undocumented animals appearing under unrelated names. Resolve those identities
before starting. D4 IDs and image paths must be separate from development.

## Sequence and release model

The default one-seed, five-fold suite runs:

1. Four fixed-split interpolation pilots, anchored to D1 fold zero. Linear remains
   the frozen main-study setting; pilot scores never configure subsequent runs.
2. The five D1 reference folds, reused as main results and MouseFull sources.
3. MouseFull reference and its three classifier/masking controls: 20 fits.
4. Four D1 component removals: 20 fits, reusing the reference from step 2.
5. Five rat fine-tuning fits, all using the predefined mouse source fold.
6. Two deployment refits: all selected D1, then all selected D1+D3.
7. One D4 evaluation using the refitted MouseFull checkpoint, not an ensemble.

That is 56 fits including pilots and release refits. Optional switches and seeds
change the displayed count. `experiment_plan.csv` gives the exact plan before
execution. All variants share saved deterministic evaluation patches within
their task. Random training augmentations are not guaranteed bitwise identical.

Release durations are the ceiling of the median **inner-validation selected
epoch counts** from the corresponding reference CV runs, for `RELEASE_SEED`.
The all-development refits have no validation/test animals, no early stopping
and no LR scheduler. The shared runner selects the minimum training-loss
checkpoint within that fixed duration and exports an inference model. Their
training scores are not independent performance evidence. The refit contract is
explicitly recorded and recognized by initialization audits; ordinary source
runs still require their selection-validation provenance.

`RAT_SOURCE_FOLD` and `RELEASE_SEED` are predefined configuration values, never
chosen from rat/D4 or outer-test performance. D4 evaluation audits the refit's
complete initialization ancestry. Interrupted D4 evaluations are preserved under
an `interrupted` name before a fresh evaluation is attempted; completed fitting
does not need to be repeated.

## Draft mode

Without supplied split files, drafts select six D1, six SHAM, three rats and two
D4 animals by default. All registered longitudinal scans of each selected animal
are retained. Explicit split files keep their animals and folds, including in
draft mode, so the draft really checks that supplied plan.

The original input inventory is validated first. Small central physical fields
are then copied into the suite's `draft_data` directory using linear MRI and
nearest-neighbour label interpolation. Originals are never edited. The copied
third-axis spacing differs slightly from training spacing to exercise resampling.
Drafts use 16-cube inputs, small filters, batch size one, one epoch, one fitting
step and one deterministic evaluation patch per scan. They still execute actual
training, transfer, recovery, full-volume prediction, metric computation, refits
and D4 evaluation on those small copies. Compilation can dominate this runtime.

One-step predictions need not contain a usable third ventricle, so drafts disable
ventricle separation and use zero brain-morphology size thresholds. Their D1/rat
outputs retain merged training IDs; MouseFull outputs retain original IDs. The
production splitting operation remains covered by existing postprocessing tests.
This is an execution check, not a convergence, geometry-sensitivity or biological
validation experiment. Draft tables contain a `DRAFT=True` column and warning.

## Results: only four paper tables

```text
results/revision_suite/full/revision_v1/
  suite.json                 # frozen settings, source fingerprints and jobs
  splits.json                # master, source and rat plans
  split_membership.csv       # animal + scan roles for collaborator handoff
  experiment_plan.csv
  state.json
  experiments/<variant>/seed_42/  # ordinary checkpoints and complete run exports
  external/D4/
  analysis/                  # optional source volumes and reader agreement
  summary/<snapshot>/
    paper/
      paper_tables.xlsx      # at most the four sheets below
      main_results.csv
      ablations.csv
      per_region.csv
      validation.csv
      README.txt
    audit/                   # detailed measurements, intervals, coverage, sources
    provenance.json
```

The four sheets are:

- **Main results:** reference and supplied comparator scores, with brain, lesion
  and anatomy-only Dice, lesion HD95, valid complete-anatomy counts and recorded
  costs where available.
- **Ablations:** the same compact scores for component/supervision controls and
  interpolation. Lesion comparisons retain paired differences, intervals,
  contributing animals and Holm-adjusted p-values. Detailed per-region tests
  remain in the audit. Interpolation reporting uses validation, explicitly.
- **Per region:** reference-model region/brain scores, physical boundary/volume
  metrics, annotation denominators and animal Dice intervals.
- **Validation:** animal classifier rates/confusion information, SHAM lesion
  false positives, ventricle symmetry, D4 corrected/uncorrected agreement and
  configured inter-rater checks.

Datasets, SHAM/TBI conditions, seeds and evaluation roles remain separate. Main
scores average scans within animals, animals within folds, then equally across
folds. Animal confidence intervals are a separate estimate conditional on the
trained models. Missing annotations and undefined distances remain missing.
Complete-anatomy averages require every anatomy target; per-region scores retain
all available annotations. Native patch measurements remain in the audit/run
exports because their cohort-pooled definition differs from animal metrics.

Every export creates a new summary snapshot. An interrupted suite can be
summarized without fitting; inspect its audit coverage before claiming complete
CV. Raw details are retained for inspection, not proposed as hundreds of paper
tables. No absent experiment or confidence estimate is fabricated.

## Analyses that need additional inputs

`INCLUDE_ANNOTATED_VOLUMES` reports D3/D4 source annotations. Unknown NIfTI units
are recorded as `needs_verified_units` in `analysis/status.csv`; set
`ANNOTATION_SPATIAL_UNIT` only after verifying their physical scale. Symmetry near
one is descriptive, not proof of individual symmetry. D4 correction keeps
negative values; they are flagged rather than clipped. The existing agreement
routine rejects negative physical volumes, so those corrected comparisons stay
explicitly unavailable rather than silently dropping animals.

`INTER_RATER_MANIFEST` uses the format in
[`check_inter_rater.ipynb`](../notebook/check_inter_rater.ipynb). Reader independence,
experience and blinding still require study documentation. The suite cannot
invent independent masks. Without a manifest that analysis is marked unconfigured.

`EXTERNAL_RESULTS`, for example `{"d1/nnunet": Path("saved_result_folder")}`, reads
collaborator full-volume CSVs in the shared result schema. Animal, seed, fold,
role, region names and annotation availability must be compatible. It does not
convert arbitrary nnU-Net outputs or quietly manufacture matching folds. Coordinate
partial-label handling, checkpoint selection, modality and postprocessing first.

This runner does not implement TOST, treatment-effect preservation inference,
parameter-matched architectures, a full training-patch balance audit, scanner-shift
validation or missing biological annotations. It automates the agreed experiments
and supported analyses, not every outstanding manuscript request.
