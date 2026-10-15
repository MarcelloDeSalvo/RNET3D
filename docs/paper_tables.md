# Manuscript tables from saved evaluations

The final section of notebooks 01–09, including 07f, reads saved measurements
without training or repeating inference. MouseFull's v1 loss remains unchanged.
The same code handles a completed K-Fold run, an ablation/comparison study, or
one saved checkpoint evaluation.

Set `PAPER_RESULTS_DIR` to the folder containing the saved results. Leave it
`None` to use the current notebook's study/run, then its checkpoint evaluation.
The cell prints the selected folder and displays fold coverage. It never searches
for the latest run. After a kernel restart, run the import/setup cell and this
section with an explicit folder; fitting cells are not required.

```python
from pathlib import Path
from src.evaluation.paper_tables import PaperTableConfig, paper_tables_from_folder

settings = PaperTableConfig(roles=("holdout", "test"), bootstrap_samples=2000)
tables = paper_tables_from_folder(Path("../results/my_model/my_run"), settings=settings)
display(tables.segmentation)
display(tables.region_sets)
```

## Main table and averaging

`paper_table` / `tables.segmentation` contains one row per variant, seed, role,
dataset, SHAM/TBI condition, output head and region. Its columns include Dice,
IoU, HD95 and ASSD (mm), absolute and relative volume error (mm³ and %),
sensitivity, precision, and reference/predicted volume (mm³), when recorded.
Brain-mask and ROI results remain identifiable by `Head`.

Each metric is averaged in this order:

1. Average a longitudinal animal's available scans within each fold.
2. Average the contributing animals within that fold.
3. Average the fold means with equal fold weights; report their sample SD.

Thus `mean ± SD` describes variation between folds. A single evaluation reports
only its mean, because one fold has no estimable fold SD. `Animals` counts unique
Dice contributors, `BoundaryAnimals` counts animals with defined boundary
measurements, and `DiceFolds` identifies folds contributing to Dice. Missing
annotations and undefined measurements are never converted to zero.
`fold_metrics` gives each metric's per-fold counts; `statistics` gives its
full-precision aggregate and denominator. Both are exported.

`paper_animal_intervals` / `animal_intervals` reports a **separate** animal mean
and bootstrap confidence interval. Repeated models are averaged per scan first,
then scans per animal, before bootstrapping unique animals. This estimate can
differ from the equal-fold mean when fold sizes differ. These intervals are
conditional on the supplied trained models; they do not estimate uncertainty
from retraining. Repeated test predictions do not increase the animal count.
An animal appearing in multiple outer holdouts for the same variant/seed is
rejected. Seeds, evaluation roles and cohorts are never silently pooled.

## Region sets and additional tables

`paper_region_table` / `region_sets` gives explicitly named averages: all ROIs,
anatomy only, all ROIs plus brain, and anatomy plus brain. Anatomy excludes lesion;
all sets exclude background. By default, a scan contributes only if every region
in the set has an available annotation. Individual ROI rows still retain every
valid partial annotation. `COMPLETE_REGION_SETS_ONLY=False` permits averages over
available regions and must be described as such in the manuscript. Undefined
boundary measurements remain excluded even for annotated regions; inspect their
denominators. SHAM's empty-lesion Dice convention is preserved, so anatomy-only
scores are the appropriate companion to any lesion-inclusive average.

Other dataframes are kept separate because their averaging differs:

| Table | Contents |
| --- | --- |
| `patches` | Native cohort-pooled patch losses/metrics, averaged across folds with fold SD; no animal confidence interval. |
| `classification`, `confusion` | Animal classification rates, exact binomial intervals per model or disjoint pooled outer holdouts, plus descriptive equal-fold rate means and SD. |
| `agreement` | Animal mean-volume agreement: bias, limits of agreement, MAE, ICC and concordance, using the existing statistics implementation. Fewer than three animals is explicit. |
| `healthy_lesion` | SHAM lesion false positives and corresponding animal-level intervals. |
| `comparisons` | Paired animal-level differences against `PAPER_REFERENCE_VARIANT`, confidence intervals and Wilcoxon tests; Holm correction over the returned comparison family. |
| `costs` | Available recorded training/inference timings and model costs, grouped by workload and provenance. No new benchmark is performed. |
| `fold_coverage` | Observed versus expected folds, where the saved split plan supplies an expectation. Inspect before reporting a completed study. |

The standard roles are `("holdout", "test")`. Notebook 04 explicitly uses its
historical `"validation"` role for the held-out rat. New notebook-04 runs use
training-only checkpoint selection and fixed learning rate; this table cannot
change the selection protocol of an older run. For external evaluation, include
`"external"` explicitly. Ablations default to `"reference"` as their paired
comparison variant; notebook 08 uses `"3drnet"`.

## Export and provenance

Enable `EXPORT_PAPER_TABLES` or call `tables.export(fresh_output_folder)`. The
export contains `paper_tables.xlsx`, one CSV per dataframe, and
`paper_analysis.json` with aggregation settings and SHA-256 input hashes.
Formatted main cells are for copying into manuscript tables; `statistics` and
`fold_metrics` retain numerical precision. Exports require a fresh folder and
verify that inputs have not changed since analysis.

The reader prefers `summary/full_volume_metrics.csv` over root-level and
`regions/` measurements, avoiding double counting. It reads matching patch and
classifier files, joins `subjects.csv` metadata if present, and uses saved
splits/evaluation provenance for fold coverage. No raw annotations, predictions,
model weights or existing workbooks are modified. Missing experiments, missing
folds and the deferred independent D4 manual-volume analysis are not supplied by
these descriptive summaries.
