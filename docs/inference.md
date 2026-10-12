# Inference and full-volume evaluation

`src/evaluation/inference.py` retains `RandomCroppingPrediction` and
`FullVolumeEvaluation`, including their existing public helper methods and
callback arguments. The refactor adds types/docstrings and removes duplicate
padding, accumulation, normalization and report-generation code.

## Prediction behavior

- Input is a single-channel 3D array or nibabel spatial image, already prepared
  by the experiment's preprocessing pipeline. Padding is zero-valued at the
  upper edges; masks are cropped back to the original shape.
- Heads must return probabilities. Single ROI outputs, the existing positional
  `[regions, brain_mask, classifier]` heads, and dictionaries using those names
  are supported. A requested missing head or inconsistent output shape fails
  with a clear error. Models returning logits need an explicit inference view.
- ROI and brain probabilities accumulate in float64, in the historical patch
  order. ROI multiclass decisions use argmax; binary ROI and brain decisions
  use `>= threshold`.
- Classifier output remains a majority vote of patch decisions at `>0.5`.
  A final tie is false. It is not the mean of patch probabilities and does not
  use the segmentation threshold.
- The existing grid can repeat clamped edge patches, even when the image fits
  exactly into a patch. This weighting is deliberately retained. Removing these
  repeated predictions would be a separate algorithm change.
- Stride must be a positive integer no larger than the smallest patch dimension,
  preventing uncovered regions. Image/patch dimensions and threshold are
  validated before inference.

All-channel array accumulation replaces the per-class Python loop. ROI and
brain outputs share one normalization helper. Existing public allocation helpers
retain their array shapes, including per-class ROI count maps.

## Evaluation and reports

`FullVolumeEvaluation` accepts legacy and normalized loader scan keys. `Modality`
in reports retains the loader collection name, such as `D1_T1w`; `Subject` is the
scan ID. Map these to canonical animal IDs through `SubjectIndex` for fold
statistics; repeated timepoints remain separate observations until explicitly
aggregated. This class still uses the legacy overlap metric factories; physical
surface/volume metrics are available through `SegmentationEvaluator` and
`segmentation_metrics_table`.

Evaluation works with `path=None`, producing a DataFrame without disk reports.
When a path is supplied, the directory is created, reports/plots are written,
and the log is closed even if evaluation raises. Repeated evaluate calls work
without changing the requested scan keys or the model's ROI channel count.
Plots use copies of their source tables and close figures after use.

Predictions are initialized from the requested model output even when no
postprocessing callback is configured. For ROI callbacks, the historical default
is five postprocessed classes; pass the keyword-only
`postprocessed_num_classes` for a different output-label contract. Brain metric
encoding always uses one class without modifying the model's ROI channel count.
One-hot encoding preserves singleton spatial axes. Returned report columns
include per-class values and the historical Mean over all class columns,
including background. That mean does not replace structure-specific reports.

## Verification

22 focused tests cover head formats, padding, repeated edge patches, classifier
votes/ties, invalid strides, missing outputs, path-optional/repeated evaluation,
brain-to-ROI evaluation, report exports, table immutability, singleton spatial
axes and a real Keras single-output model. Direct comparison against the previous
committed implementation passed exactly for 36 synthetic configurations,
including ROI/brain/classifier outputs and prediction-call counts. The file is
about 20% shorter. The legacy notebook import (`evaluation.inference` with src
on sys.path) and repository import (`src.evaluation.inference`) both work.
