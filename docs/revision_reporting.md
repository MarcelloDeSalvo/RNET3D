# Revision reporting: step 7

Notebook [10_revision_results.ipynb](../notebook/10_revision_results.ipynb)
collects saved evidence without training or repeating inference. The additional
baseline experiments from step 6 and the separate D4 manual-volume analysis are
deferred at the user's request. No missing experimental result is fabricated.

## Select inputs explicitly

`METRIC_SOURCES` selects full-volume CSVs from a run, study or checkpoint
evaluation. Supply variant, seed, fold and role when those columns are absent.
Keep `name=None` to retain the variant names already present in a study table.
Do not load a study table and its constituent run tables together.

`PATCH_SOURCES` optionally selects native `patch_metrics.csv` files. These
must declare `EvaluationStage=patch` and `Aggregation=cohort_pooled`. They
retain their compiled loss/metric definitions, original counts and roles,
including validation. They are exported separately and never enter the
full-volume animal bootstrap. A combined `evaluation_metrics.csv` can be
selected for full-volume analysis: its patch rows are filtered out first.

`CLASSIFIER_FILES` remains the separate animal-level classifier workflow. Its
exports and exact binomial intervals are described in
[classifier_evaluation.md](classifier_evaluation.md). The revision report's
healthy-lesion diagnostic uses segmentation volumes independently of classifier
decisions.

## Recorded computational costs

`COST_SOURCES` accepts `CostSource` entries for:

- Run `summary/fold_timings.csv`: training/evaluation seconds, completed epochs,
  parameter count and the stored best-epoch index.
- Study `timings.csv`: the same fold records with variant and seed columns.
- Evaluation `inference_timings.csv`: preprocessing, inference/reconstruction
  and postprocessing seconds for each scan and model.

```python
from pathlib import Path
from src.evaluation.revision import CostSource

run_dir = Path("results/d1_3drnet/RUN_...")
cost_source = CostSource(
    path=run_dir / "summary/fold_timings.csv",
    name="3drnet", seed=42,
    metadata_file=run_dir / "metadata.json",
)
```

Single-run files need an explicit variant and seed; inference files also need
a fold. Set `role` and `dataset` when inference files omit those workload
columns. These fields remain separate in summaries. Without them, the workload
is `unknown`; do not interpret it as an identified test/external benchmark.

`metadata_file` selects actual runtime provenance. For inference performed
within training, use that run's `metadata.json`. For a later checkpoint
evaluation, use its own `evaluation.json`: the hardware/precision may differ
from training. Study-wide timings can be summarized without runtime metadata,
but then device and precision remain `unknown`; select individual run timing
files with their metadata for hardware-specific summaries.

Device fields describe recorded available physical devices, not measured GPU
utilization or hardware model specifications. Match hardware, precision,
cohort, preprocessing, patch size, stride and scan workload before interpreting
speed differences. The report does not measure peak memory or turn missing
hardware information into an assumption. A legacy zero parameter count means
not recorded, and remains missing. Negative/nonfinite durations and duplicate
cost observations are rejected.

Cost means, medians and ranges describe fold/scan-model records. They are not
animal confidence intervals, estimates of seed uncertainty, or isolated GPU
kernel benchmarks. The inference evaluator already excludes warm-up. No new
benchmark is launched. Existing timers and epoch indexing are unchanged.

## Coverage and failure review

`metric_coverage` reports observed and finite animal/scan counts, raw row
counts, unavailable annotations and undefined measurements. Repeated seeds
do not increase animal counts. A scan absent from every selected input cannot
be discovered from these metric files alone: compare with the intended split
manifest before claiming complete evaluation coverage. Brain-head rows remain
in the full-volume tables, alongside the region measurements.

`failure_review` retains original source, subject, scan, seed, fold, region,
metric and status. It selects descriptive review candidates:

- Dice below `RevisionConfig.failure_dice_threshold`, default 0.5.
- Empty predictions for a nonempty reference.
- Nonempty predictions for an empty reference.
- Unexpected nonfinite values whose status was recorded as `ok`.

Unavailable annotations and the documented both-empty metric convention are
not diagnosed as model failures. The review threshold never removes an animal,
changes a metric, selects a checkpoint or defines a new training threshold.
Repeated metric rows for a candidate scan remain visible for audit.

Use notebook 09 to inspect the saved prediction/GT for a candidate's exact
variant, seed, fold and scan. Confirm the geometry and annotation availability,
record the visual finding, and retain the source of each manuscript example.
No automatic anatomical diagnosis or run-failure rate is inferred from a
low score; failed jobs with no metric artifacts are outside this table.

## Exports

Enable `RUN_ANALYSIS` after selecting files. Review the displayed tables, then
set `EXPORT_TABLES=True` and run the export cell. The fresh output directory
contains full-precision CSVs and `revision_tables.xlsx`, with separate sheets
for pooled patch measurements, animal results and uncertainty, metric coverage
and statuses, volume pairs/agreement, comparisons, healthy-lesion diagnostics,
costs, review candidates and source files.

`source_files.csv` records SHA256 hashes and sizes of selected metric, patch,
timing and runtime metadata files. Export rejects source files changed since
analysis. `analysis.json` records the analysis settings, aggregation, hashes
and reporting scope. `reporting_notes.md` records the practical limitations.
Undefined/unavailable values remain blank. Existing output folders are rejected.

Existing reference-volume rows still support the established agreement section.
They do not replace the deferred D4 independent manual-volume analysis. No
training, inference, interpolation sensitivity analysis, baseline result, peak
memory measurement or manuscript performance claim is generated by this step.

The separate full-volume evaluator's restriction on nonorthogonal grids for
physical distance metrics is unchanged. This reporting step consumes already
saved measurements and does not repair geometry or recompute masks.
