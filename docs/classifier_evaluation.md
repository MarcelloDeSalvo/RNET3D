# Auxiliary classifier and healthy-lesion evaluation

## Preserved prediction rule

With `ModelConfig.classifier=True`, full-volume evaluation retrieves the classifier
from the same patch inference calls used for regions/brain. No additional pass is
made. Architecture, classifier loss, weights and dropout are unchanged.

SHAM is reference 0 and TBI is reference 1, matching the training generator.
Each patch votes TBI only when its classifier output exceeds 0.5. A volume is TBI
only when more than half its patch votes are positive; ties remain SHAM. Repeated
clamped edge patches retain their historical weight. `PositivePatchFraction` is a
fraction of binary decisions, **not** a calibrated probability.

`classifier_predictions.csv` records animal/scan/cohort, reference/prediction,
positive patch fraction, patch count, threshold/rule, model, seed, fold and role.
Runs, studies and checkpoint evaluation expose `result.classifier_predictions`.
Models without the head supply no classifier decisions; they are not assumed to
predict SHAM for every animal. Completed folds reload committed records. Older
evaluations without classifier records need fresh checkpoint evaluation, not fitting.

## Independent animal reporting

`classification_report` combines scan decisions by majority within an animal,
again with ties SHAM. Each animal must have a consistent reference within the
selected cohort. Single-scan MouseFull cases retain their scan decision exactly.

Accuracy, TBI sensitivity, SHAM specificity and TBI precision use the existing
`calculate_classification` helper and exact Clopper–Pearson intervals. Undefined
rates remain missing. A perfect 16/16 classification has a two-sided 95% accuracy
interval of approximately [0.794, 1]; 32 scans from those same 16 animals still
contribute only 16 independent decisions.

Variant/seed/fold/cohort estimates remain separate. Disjoint outer holdouts also
receive `Scope=pooled_holdout, Fold=-1`, with every animal contributing once.
Duplicate animal membership across those holdouts is rejected. Repeated test or
external predictions across folds are separate model estimates; no ensemble or
independent-sample pooling is invented. Seeds are not extra animals. Intervals
condition on supplied models and do not quantify training-seed variability or
dependence from overlapping fold training sets. Validation remains selection
performance, not an independent test.

## SHAM lesion false positives

Classifier decisions never gate, erase or relabel segmentations. Full-volume
metrics save the verified condition even without a classifier. `healthy_lesion_report`
uses SHAM lesion prediction volumes in mm3 and requires available empty GT. Missing
condition/annotations are not interpreted as healthy; an empty TBI lesion is not SHAM.

Per animal it reports mean/max scan lesion volume and whether any scan contains
any nonzero predicted lesion. Summaries give mean animal volume, maximum volume,
false-positive animal count/rate and an exact rate interval. A correct SHAM
classifier can coexist with a lesion false positive. These diagnostics also cover
no-classifier ablations. No volume tolerance or classifier gating is tuned here.

## Files and notebooks

Run summaries, studies and checkpoint-evaluation folders save:

- `classifier_predictions.csv`: scan decisions and provenance.
- `classifier_animals.csv`: one decision per animal/model/cohort.
- `classifier_summary.csv`: rates, intervals and sample counts.
- `classifier_confusion.csv`: counts, reference rows and prediction columns.
- `sham_lesion_animals.csv` and `sham_lesion_summary.csv`: separate lesion diagnostics.

Nonempty tables are sheets in `evaluation_metrics.xlsx`. Individual fold folders
save classifier CSV/JSON records for recovery. Training/study result sections,
best-checkpoint cells and notebook 09 display classifier and SHAM summaries.

Notebook 10 exposes `CLASSIFIER_FILES`, roles through `analysis.roles`,
`RUN_CLASSIFICATION` and `EXPORT_CLASSIFICATION`. It displays rate/animal tables
and a Plotly confusion matrix selected by variant/seed/role/scope/fold. Classifier
analysis can run independently of segmentation analysis. Segmentation reports
also export SHAM diagnostics when input metrics contain the condition; older CSVs
lacking it are not guessed to represent healthy animals.

Tests cover threshold/tie preservation, inference-call counts, independent animal
intervals, absent-class rates, duplicate records, pooled holdouts, classifier/lesion
independence, tiny fits, recovery and checkpoint evaluation. No real classifier
performance is claimed and no dataset training is launched here.

The broader regression run also exposed three existing shear-handling failures:
`test_experiment_handles_shear_before_affine_augmentation` expects orthogonalization,
while the current shared resampler uses `orthogonal=False`. Preprocessing was left
unchanged in this classifier step. A transient backbone-loading memory failure
passed when rerun independently.
