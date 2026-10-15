# Cohort, anatomy and annotation reports

These additions use saved full-volume scores or explicit annotation masks.
Training defaults, masking objectives, architecture, rat selection protocol,
callbacks and computational timing definitions are unchanged.

## Native annotated volumes and SHAM ventricle symmetry

```python
from pathlib import Path
from src.evaluation.volumes import dataset_volume_report

dataset_dir = Path("D:/Other files/TESI_CNN/MANUSCRIPT_DATASET/normalized/D3")
report = dataset_volume_report(dataset_dir)
display(report.animals)
display(report.summary)
```

Pass one normalized Dx directory. The function includes every registered scan
and subject, resolves annotation availability from metadata, and measures original
IDs 1/13/3 for lesion/ipsilateral/contralateral ventricles. `VolumeLabels` exposes
the IDs. Unknown regions remain missing even if the mask contains that label.
Known SHAM lesion absence is available zero; contradictory positive lesion
annotations raise an error.

Physical volume is voxel count times the absolute determinant of the full
affine, including shear. NIfTI units are converted to mm. Unknown units require
an explicit `spatial_unit` override; no physical scale is guessed. This report
does not resample or reorient the source annotations.

Scan rows include both ventricle volumes, ipsilateral/contralateral ratio,
uncorrected lesion volume, and
`LesionCorrectedMm3 = lesion + ipsilateral - contralateral`. Negative corrections
remain negative and flagged. A zero contralateral volume makes the ratio
undefined; missing annotation is a separate status. No clipping or correction
of the annotation itself occurs.

Animal rows average paired scan measurements and scan ratios within each animal,
with contributing counts for every quantity. Condition-specific distributions
report finite/observed animals, mean, sample SD, median and range. SHAM and TBI
are never pooled for the symmetry interpretation. A mean near one is descriptive
and does not establish individual symmetry or interchangeability of volumes.

Use [check_dataset_volumes.ipynb](../notebook/check_dataset_volumes.ipynb) for the
short editable workflow and native Plotly distribution. `report.export(path)`
writes CSVs and `dataset_volumes.xlsx` into a fresh folder. Exporting corrected
and uncorrected measurements does not execute a D4 agreement or sensitivity study.

Read-only verification on the current D3 annotations found 86 scans/86 SHAM
animals, all with defined ratios: mean 0.986413, SD 0.227344, median 0.984278,
range 0.512459–1.671610. These are annotation measurements, not model results.
No dataset files or existing results were changed.

## Shared full-volume evaluation reports

All new run, K-Fold, study and checkpoint workbooks include additional tables
from `evaluation.reporting.evaluation_report`. Notebook result sections display
the shared anatomy/cohort tables. Original losses, metric values and native
pooled patch measurements retain their existing definitions.

The four explicit region sets are:

| Region set | Lesion | Anatomical ROIs | Brain head | Background |
| --- | --- | --- | --- | --- |
| `all_roi` | Included | Included | Excluded | Excluded |
| `anatomy_only` | Excluded | Included | Excluded | Excluded |
| `all_roi_and_brain` | Included | Included | Included | Excluded |
| `anatomy_and_brain` | Excluded | Included | Included | Excluded |

`region_set_scans` reports macro scores over finite available regions, expected/
annotated/finite counts, the exact contributing names, and `CompleteRegionSet`.
An empty contributing set stays NaN. A partial TBI anatomy mean describes only
its measured regions, never accuracy for the missing anatomical targets.
`region_set_summary` averages repeated models per scan, then scans within
animals; SHAM/TBI and datasets remain separate. These added descriptive means/SDs
do not estimate seed uncertainty. They are separate from notebook 10's existing
per-region animal-bootstrap summaries.

`region_coverage` retains unavailable/empty statuses by condition and region.
`subgroup_summary` reports separate descriptive strata for condition, modality,
timepoint, subgroup, strain and expert-provided `FusedBoundary` when available.
Strain is not guessed from folder names. A `FusedBoundary` flag must be supplied
explicitly by an expert; the report never diagnoses fusion from predictions.

New scan manifests/full-volume rows preserve known acquisition and condition
metadata. To enrich an older CSV in notebook 10, set `MetricSource.manifest` to
an explicit CSV containing `Subject, Scan, Dataset` and the desired metadata
columns. A joined scan must have one consistent metadata record; known conflicts
raise errors. Missing metadata remains unknown. Source hashes include the
selected metadata file.

`participant_flow` describes the supplied inventory, `split_flow` describes saved
train/validation/holdout/test memberships, and `evaluation_flow` describes the
observed metric scans. All distinguish animal and scan counts. Dataset totals
and acquisition/timepoint strata are separate `Scope` rows: adding stratum animal
counts can count an animal repeatedly. Model seeds/folds remain explicit for
evaluation counts. No absent prediction is inferred from metric rows alone.

Native patch losses remain cohort-pooled. Condition-specific patch scores cannot
be reconstructed from those aggregate records; the new SHAM/TBI separation uses
full-volume scan measurements. No extra patch evaluation pass is introduced.

## Structure size and boundary metrics

`structure_size_metrics` pairs scan/model Dice, HD95, ASSD, sensitivity and
precision with GT/predicted volume in mm3. `MissedStructure` identifies a nonempty
reference with empty prediction; unknown measurements remain missing. Notebook
10's editable `SIZE_REGION`/`SIZE_METRIC` cell plots performance against actual
reference volume, retaining animal/scan/fold/seed identity on hover. No volume
cutoff is tuned and no minimum reliable size is asserted.

The physical metric functions now optionally accept `affine_mm`. Spacing-only
calls retain the original orthogonal-grid calculation. For a shared grid with
shear, surface voxel centres are compared in physical space using nearest-neighbor
point queries, and volume uses the affine determinant. The full-volume evaluator
supplies its grid affine and no longer rejects a scan solely for shear. Image
alignment is still checked; no new input interpolation or mask shift is introduced.

## Independent-reader workflow

Use [check_inter_rater.ipynb](../notebook/check_inter_rater.ipynb) and
`evaluation.inter_rater`. Its CSV manifest has one row per paired scan:

```text
Subject,Scan,Dataset,RaterA,RaterB
animal1,scan1,D1,reader_a/scan1.nii.gz,reader_b/scan1.nii.gz
```

Relative paths resolve beside the manifest. Both masks must share a physical
grid and label IDs. The workflow does not automatically interpolate, swap labels
or adjudicate disagreement. Select common foreground IDs through `REGIONS`.
Optional `AvailableA`/`AvailableB` cells are JSON ID lists; availability is
intersected, and annotated empty regions remain valid. The same file cannot be
accidentally supplied twice as independent readers.

Reports include per-scan overlap/boundaries, per-animal scan means, and volume
ICC(A,1), CCC, bias and existing agreement diagnostics/intervals. Volume agreement
uses paired animal mean volumes, not independent longitudinal scans. Fewer than
three paired animals stay marked `too_few_animals`. Dice/distances are symmetric;
precision/sensitivity and signed bias retain reader-A-to-reader-B direction.
Undefined empty-mask distances remain NaN. The interactive comparison reorients
both labels together to the common RIA reference without changing measurements.

Complete the notebook's reader experience, blinding, boundary instructions and
disagreement-resolution fields for the manuscript. Code cannot establish reader
independence. Exports retain mask paths/hashes and reject masks changed after
measurement. Actual inter-rater results require the independent-reader masks.

## Decisions retained from the discussion

No new masking experiment is enabled. V1 versus V3 compares objective versions
that both use masking; it does not answer whether partial data/masking helps
against ignoring partial scans or naively treating unknown anatomy as background.
Those would be explicit experimental controls, not production annotation policy.
With the current ten-foreground-region vocabulary and verified empty SHAM lesion
targets, metadata identifies 9 complete TBI animals and 79 complete SHAM animals;
103 TBI and 7 SHAM animals remain partial. A complete-only experiment therefore
changes sample size and condition balance as well as data use. It is not a pure
comparison of loss formulas. A naive-background control measures the consequence
of wrong supervision; it is not an expectation that wrong targets should win.

No patch-balance audit, parameter-capacity matching or additional hardware
benchmark is introduced. Symmetry addresses the correction assumption, not voxel
class balance. Existing loss weights and timing records can be reported directly.

Notebook 04 keeps every held-out rat scan out of fitting. New runs have no
ReduceLROnPlateau/early stopping and select the lowest training-loss checkpoint
with `monitor="loss"`. Its held-out curves are diagnostic, with the legacy
`validation` role retained for saved splits/viewers. They must not be used to
choose settings after viewing results. Older runs used held-out `val_loss` for
scheduling/selection; changing the notebook cannot repair those measurements
retroactively. The other standard CV workflows use separate inner validation
and outer holdouts; their validation-driven callbacks remain appropriate.
