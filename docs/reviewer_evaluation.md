# Reviewer evaluation additions

Implemented from `to_be_added.md` in the existing three requested modules.
Training metric factories and their historical formulas are unchanged. The new
full-volume helpers are offline calculations, not Keras patch metrics.

## Full-volume segmentation (`metrics.py`)

`binary_segmentation_metrics(reference, prediction, spacing_mm)` accepts aligned
binary 3D masks. `segmentation_metrics_table` accepts integer label maps plus an
explicit `{label_value: anatomical_name}` mapping and returns long-form rows:

`Subject, Scan, Dataset, Region, Label, Metric, Value, Unit, Status`.

Available measurements are Dice, IoU, HD95, ASSD, absolute volume error (AVE),
signed relative volume error (RVE), sensitivity, precision, and the reference
and prediction volumes. Distances are in mm, volumes in mm3, RVE in percent,
and other scores in fractions. AVE is `abs(predicted volume - reference volume)`;
RVE is `100 * (predicted volume - reference volume) / reference volume`.

- Supply voxel spacing in array-axis order, already converted to millimetres.
  Equal array shapes alone do not establish equal physical geometry. Reorient
  and resample to the same orthogonal grid before evaluation; spacing cannot
  describe shear. No label remapping or probability threshold is guessed.
- Surfaces are 6-connected boundary voxel centres, including image borders.
  HD95 is the maximum of the two directed 95th percentiles. ASSD is the equally
  weighted mean of the two directed means. This is a discrete surface metric,
  not a surface-area-weighted mesh measurement.
- If either mask is empty, HD95 and ASSD are NaN. Both-empty Dice/IoU equal one;
  one-empty overlap is zero. Sensitivity, precision and RVE are undefined when
  their denominators are zero. AVE and volumes remain defined.
- Unavailable labels are supplied explicitly with `available_labels`; every
  metric for those structures is NaN with `Status=unavailable`. An annotated
  structure with no foreground is different from unavailable annotation.
- Status is `ok`, `both_empty`, `reference_empty`, `prediction_empty`, or
  `unavailable`. CSV NaNs are blank fields: report valid counts and empty-mask
  counts alongside averages instead of silently discarding these cases.

```python
from src.evaluation.metrics import segmentation_metrics_table

regions = {1: "Lesion", 3: "Contra ventricle", 13: "Ipsi ventricle"}
rows = segmentation_metrics_table(
    reference_labels, predicted_labels, spacing_mm, regions,
    subject="D2/SUBJ_0113", scan="SUBJ_0113_7days", dataset="D2",
    available_labels=[1, 3, 13],
)
```

## Fold integration (`k_fold.py`)

`SegmentationVolume` carries reconstructed label maps, spacing, scan identity,
dataset and available labels. `SegmentationEvaluator` calls an experiment's
prediction callback for each canonical subject and calculates the metrics:

```python
from src.evaluation.k_fold import SegmentationEvaluator, SegmentationVolume


def predict_subject(model, subject_id):
    for scan_key in subjects.scan_keys([subject_id]):
        scan = loader.get_subject(scan_key)
        # Experiment-owned reconstruction: return aligned label maps and mm spacing.
        reference, prediction, spacing = reconstruct_and_predict(model, scan)
        yield SegmentationVolume(
            reference=reference, prediction=prediction, spacing_mm=spacing,
            scan=scan.id, dataset=scan.subject_group[0],
            available_labels=available_labels_for_scan(scan),
        )


def make_evaluator(model, subject_ids, options):
    return SegmentationEvaluator(
        model, subject_ids, options,
        predict_subject=predict_subject, regions=regions,
    )

# Pass fve_callbacks={"anatomy": make_evaluator} to k_fold_loop.
```

`reconstruct_and_predict` and `available_labels_for_scan` stand for the existing
experiment's inference/annotation policy. The runner does not invent those
choices or convert logits implicitly. Prediction saving also belongs to that
callback/factory; `options['evaluation_dir']` and `options['save_predictions']`
are available to the factory for a closure if needed.

The runner adds `Fold, DatasetRole, Evaluator, ModelVariant, ExperimentName,
DatasetName, RunID` to exported rows. `Dataset` identifies a scan's source cohort;
`DatasetName` is the experiment-level description from `ExperimentMetadata`.
Long-form tables containing Subject/Metric/Value must cover every requested
subject. Missing full-volume predictions and duplicate scan IDs within an animal
raise errors. Repeated timepoints and external-cohort fold predictions stay as
separate rows; no implicit independence or averaging is introduced.

The brief mentions `compare_experiments.py`, but that file is not present in this
repository. CSV outputs can be consumed by that future/external comparison tool.

## Agreement and paired tests (`statistics.py`)

`calculate_agreement`, `agreement_table`, and `export_agreement` retain
Bland-Altman, bias, MAE, ICC(A,1), CCC and subject-level bootstrap intervals.
They now additionally report:

- Two-sided Wilcoxon signed-rank statistic/p-value. Differences are computed
  once, exact zeros are discarded, and SciPy `method='auto'` handles the test.
  The all-zero case explicitly returns statistic 0 and p=1. Ties/zero handling
  follows the installed SciPy version; this is not always an exact p-value.
- Median-centred Levene statistic/p-value, labelled as assuming independent
  samples. It does not account for within-animal manual/automated pairing.
- Descriptive reference/prediction SDs and percentage SD reduction:
  `100 * (1 - prediction_sd/reference_sd)`. Positive values mean lower automated
  SD; this is different from percentage variance reduction. Reference SD zero
  makes the percentage undefined. The companion percentile interval resamples
  **paired animals together**, preserving within-animal dependence. This is a
  paired-data sensitivity analysis, not a corrected Levene p-value or a robust
  estimator against outliers. Original LoA endpoints keep their existing paired
  bootstrap uncertainty intervals; no small-sample precision is guaranteed.
- OLS slope of difference versus pair mean, paired-bootstrap slope interval,
  intercept and diagnostic slope p-value. Shared measurement terms and unequal
  measurement variances can induce trends: a slope is not proof of a mechanism.
- Shapiro-Wilk difference-distribution diagnostics for nonconstant differences
  and 3–5000 animals, plus histogram and normal Q-Q plots in exported reports.
  Normality is not established by a nonsignificant test, especially with n=16.
- Descriptive Pearson correlation and RMSE for continuity. They do not replace
  agreement statistics.

Optional `AgreementConfig(include_paired_t=True)` or CLI `--paired-t` adds the
paired t-test. This is an explicit analysis choice, not automatically selected
using Shapiro's p-value. Degenerate diagnostics remain NaN (JSON null), and
undefined intervals remain None. Diagnostic p-values are unadjusted: define the
confirmatory questions and multiplicity policy before interpreting several tests.
A nonsignificant paired location test does not establish agreement.

## Binary classification helper (`statistics.py`)

`calculate_classification` is separate from volume-agreement analysis. Supply
one independent animal per row with binary `Reference`/`Prediction` decisions:

```python
from src.evaluation.statistics import calculate_classification

classification = calculate_classification(subject_decisions, confidence=0.95)
print(classification.confusion_matrix)  # [[TN, FP], [FN, TP]]
print(classification.accuracy, classification.ci["accuracy"])
```

The positive class is label 1. Apply a prespecified probability threshold and
any scan-level aggregation before calling. The helper returns accuracy,
sensitivity, specificity and precision, with exact Clopper-Pearson binomial
intervals using each rate's denominator. For 16/16 correct independent animals,
the two-sided 95% accuracy interval is approximately [0.794, 1]. Missing-class
rates or no predicted positives give undefined rates/intervals rather than a
perfect score. Duplicate subjects are rejected. Select a single intended model
and cohort before using repeated external-fold predictions; do not pool them as
independent subjects. The helper does not rerun classification inference.

## Verification and references

Tests use anisotropic masks with known distances/volumes, empty/unavailable
structures, full-image boundaries, paired scaling with a known 19% SD reduction,
SciPy reference test results, constant measurements, duplicate IDs and 16/16
classification uncertainty. Existing numerical training-metric regressions are
also retained. No dataset performance claim or study training is generated.

Implementation references: [SciPy Wilcoxon](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wilcoxon.html),
[SciPy Levene](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.levene.html),
[exact binomial intervals](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats._result_classes.BinomTestResult.proportion_ci.html),
and [physical distance transforms](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.distance_transform_edt.html).
