# Cross-validation and ablation runs

`src/evaluation/k_fold.py` retains the factory-based runner: every fold gets a
fresh compiled model, fresh datasets/callbacks, and a managed best-weights
checkpoint. Architecture and loss choices stay in the model factory. Reuse
`splits.json` across ablations to compare the same validation animals.

## Connect the loader and generators

`SubjectIndex` is the small bridge between animal-level splitting and the
`(group, scan_id)` keys expected by `MultiTaskGenerator` and
`DomainAdaptationGenerator`. It supports both loader layouts. Normalized scans
are grouped by `Rodent.subject_group`, so all timepoints of an animal stay in
one fold. IDs look like `D2/SUBJ_0113`; components are URL-escaped to prevent
separator collisions. The runner rejects raw scan tuples rather than silently
turning them into unusable strings.

Build the index after registering the required datasets:

```python
from pathlib import Path
from src.utils.loader import RodentDatasets
from src.evaluation.k_fold import SubjectIndex

root = Path(r"D:\Other files\TESI_CNN\MANUSCRIPT_DATASET\normalized")
loader = RodentDatasets(labels=labels)
loader.add_normalized_dataset(root, "D1")
loader.add_normalized_dataset(root, "D2")
subjects = SubjectIndex(loader)
cv_ids = subjects.subject_ids
```

`labels`, the generator settings, preprocessing and model compilation remain
experiment-specific. Normalized storage does not imply that mask values have
been remapped or that missing-label policy can be inferred. For partial-label
training, register unavailable labels explicitly and use the existing
`DomainAdaptationGenerator` and appropriate model outputs.

A factory expands fold subjects only when constructing the generator:

```python
from src.preprocessing.generator import MultiTaskGenerator
from src.evaluation.k_fold import TrainingConfig, ExperimentMetadata, k_fold_loop


def make_training_data(subject_ids: list[str]) -> MultiTaskGenerator:
    scan_keys = subjects.scan_keys(subject_ids)
    return MultiTaskGenerator(
        scan_keys, loader, generator_config,
        preprocessor=train_preprocessor, shuffle=True,
    )


def make_validation_data(subject_ids: list[str]) -> MultiTaskGenerator:
    scan_keys = subjects.scan_keys(subject_ids)
    return MultiTaskGenerator(
        scan_keys, loader, generator_config, batch_size=1,
        preprocessor=validation_preprocessor, shuffle=False,
    )


settings = TrainingConfig(model_name="rnet", k=5, epochs=100, output_dir="results")
metadata = ExperimentMetadata(model_variant="baseline", dataset_name="D1+D2")
result = k_fold_loop(
    train_gen=make_training_data,
    val_gen=make_validation_data,
    model_callback=build_compiled_model,
    config=settings,
    train_and_test_ids=cv_ids,
    subject_index=subjects,
    metadata=metadata,
)
```

The example assumes configured `labels`, `generator_config`, preprocessors and
`build_compiled_model`. Use deterministic validation preprocessing. The generators
currently drop incomplete batches; validation batch size 1 includes every scan.
Leave `infer_steps_from_ids=False`: animal counts differ from scan/patch counts.
The index integration rejects that option; use dataset cardinality or explicitly
configured steps instead. Empty Keras Sequence generators fail before model
construction. No preprocessing or model defaults are changed by the runner.

## Identities and independent cohorts

The normalized dataset was checked read-only: D1 has 112 subjects/112 scans,
D2 has 5 subjects/20 scans, D3 has 86 subjects/86 scans, and D4 has 16 subjects/16
scans. These are supplied normalized identities, not independently verified
biological identities. In particular, repeated original ID strings in D3 are
not automatically merged. Legacy layout grouping falls back to modality/scan ID
and cannot identify repeated animals without explicit information.

If confirmed scans across groups identify the same animal, pass
`subject_overrides={scan_key: "confirmed-animal-id", ...}` to `SubjectIndex`.
All scans of an existing subject group must receive the same override. Do not
mix original and normalized copies of the same data without reconciling their
identities. Original IDs and timepoints are preserved in the manifest.

Register independent cohorts in the same loader before building the index, then
select disjoint canonical IDs for CV, `EvaluationConfig.test_ids` and
`EvaluationConfig.external_ids`. Merely registering a dataset does not hold it
out: explicitly exclude its IDs from `train_and_test_ids`. The runner verifies
all selected IDs exist and are mutually disjoint, and exports `subjects.csv`
with scan keys, source paths, original IDs, timepoints and cohort roles.

## Evaluation and statistics

Evaluator factories still receive `(model, subject_ids, options)`; expand IDs
with `subjects.scan_keys(subject_ids)` when calling repository inference code.
`options["evaluation_dir"]` is a fresh directory for that fold/cohort/evaluator,
which can be passed as the existing `FullVolumeEvaluation` constructor's `path`.
Capture inference settings, loader and preprocessing in the factory: the runner's
training config is not a substitute for the inference config.

The existing `FullVolumeEvaluation` is not a drop-in factory: it needs those
additional arguments and reports scan identities. An evaluator wrapper should
return `FullVolumeEvaluation.evaluate(...).summary` to satisfy the DataFrame
contract now that inference returns typed results. It should also
map each `(Modality, Subject)` scan key through `subjects.subject_for_scan(key)`
into canonical `Subject` values, retaining separate scan/timepoint columns.
The runner accepts either a `Subject` column or a named `Subject` index and
rejects missing/unrequested subject IDs when that field is present. Legacy
aggregate-only evaluator tables remain supported, but cannot establish subject
coverage. It does not silently aggregate timepoints, regions or inference strides.

For the gradient-fixed loss, the training view produces ROI logits. Supply a
probability-producing inference view to existing evaluators; the runner does
not guess output semantics or apply another softmax automatically.

Validation scores are out-of-fold model assessments; independent cohorts are
predicted by every selected fold model. Those repeated predictions are not
independent animals. Before using `statistics.py`, define the subject-level
measurement/aggregation and select the intended metric, region and cohort.
Do not pass repeated fold/timepoint rows directly as independent observations.

## Reusing folds

Pass `precomputed_splits=previous_result.run_dir / "splits.json"` on later
ablations, using the same canonical subject IDs and the same identity mapping.
A changed registration order does not affect the index's sorted IDs. Saved
splits are validated for coverage, overlap and exactly one validation appearance
per animal. Custom splitters must return nonnegative integer indices. Stratified
or grouped custom splitting can still be supplied through `splitter`; the runner
does not infer stratification from modality, species or condition.

Validation includes synthetic legacy/normalized scans, grouping and invalid
split cases, JSON split reuse, and a two-fold toy Keras training run exercising
checkpoint restoration and evaluation exports. No study model was trained.

`SegmentationEvaluator` now provides an opt-in full-volume metric adapter, and
evaluation rows retain experiment/model-variant identifiers. See
[Reviewer evaluation additions](reviewer_evaluation.md) for its prediction
callback contract, physical units, empty-mask policy and statistical helpers.
