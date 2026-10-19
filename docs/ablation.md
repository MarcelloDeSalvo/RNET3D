# Matched component and loss ablations

Notebook 07 uses D1 and the historical four-class task. Notebook 07f uses
MouseFull D1+D3, eleven classes, corrected v1 loss and scratch initialization.
Both keep `RUN_STUDY=False` until explicitly enabled. They expose complete
configurations, the experiment matrix and actual Keras callbacks.

Both ablation notebooks default to `K_FOLD_COMPATIBLE=False` and reuse fold zero
of a saved three-role plan. Notebook 07 selects `D1_splits_final.json`;
07f selects `MouseFull_5fold_balanced_splits.json`. The
[reference split guide](reference_splits.md) describes their exact roles,
preserved D1 outer tests and separate balanced MouseFull plan.
Change `REFERENCE_FOLD` to select another predefined fold, or enable K-Fold to
repeat every variant across the full reference plan. Explicit saved study splits
and resumed memberships retain precedence over the default reference.

## Default matrix

Every component row starts from the full reference and changes one setting.
The explicitly named combined supervision control in 07f changes both masking
and classifier presence, as shown in the plan.

| Row | Change | Retained behavior |
|---|---|---|
| `reference` | None | Original 3DR-Net settings and filter widths |
| `no_residual` | `residual=False` | Encoder-decoder skips, attention and head fusion |
| `no_attention` | `attention=False` | Residual blocks and skips |
| `no_multiscale` | `multiscale=False` | Encoder/decoder and their skip connections |
| `no_deep_supervision` | `deep_supervision=False` | Finest-level projection and final output layer in both heads |
| `strided_downsampling` | `max_pooling=False` | Resolution levels; pooling is replaced by strided convolution |
| `no_classifier` (07f) | `classifier=False` | Region and brain heads, partial-label supervision |
| `missing_as_background_no_classifier` (07f) | No masking and `classifier=False` | Same cohort, region/brain heads, v1 loss formula, preprocessing and evaluation |

The three multiscale pooled-input branches and their encoder concatenations are
absent in `no_multiscale`. The head-fusion ablation removes the three coarse
decoder projections and their sum from each segmentation head. The original
finest projection remains. The original builder's deep supervision consists of
fused multiresolution logits, not separately supervised auxiliary outputs.

Both new flags default to `True`; existing full-model graphs, outputs, weight
ordering, filter widths and dropout settings remain unchanged. Older saved
configurations without these fields load with both enabled. Conventional U-Net
uses its existing builder and does not use these flags.

Notebook 07 has six default variants; 07f has eight. The default fixed-split mode
requires six or eight fits per seed. Optional five-fold mode requires 30 or 40
fitted models, respectively. Optional rows add to
that count. The displayed plan and model count should be reviewed before execution.

## Optional loss rows

The visible `loss_variants` list is empty by default. Uncomment selected rows:

- D1: change `regions` from `dice_ce` to `dice_ce_gradient_fix`. The notebook
  explicitly sets the otherwise inactive `gradient_fix_version=1`, so this row
  selects corrected v1 while keeping all other loss settings unchanged.
- MouseFull: change `gradient_fix_version` from v1 to v3. An optional v2 row is
  retained for diagnosis after the reported poor convergence; it is not a default.

These rows retain the complete reference architecture, preprocessing, task
weights and SHAM known-empty lesion supervision. Changing an inactive version
while using the legacy probability loss is rejected. Legacy `dice_ce` on a
partial-label prepared dataset is rejected before any variant starts fitting;
switching objectives alone must not erase missing-label semantics. The explicit
missing-as-background control below is the deliberate exception.

## Missing labels as background: reviewer control

07f enables `INCLUDE_MISSING_AS_BACKGROUND_CONTROL=True` by default. Its
`missing_as_background_no_classifier` row sets
`loss.annotation_policy="missing_as_background"` and removes the auxiliary
classifier. This is an intentionally naive comparator, separate from all normal
training defaults. Setting the notebook switch to `False` omits it.

No ROI channel contains the `-1` ignore sentinel in this control. Existing
zero-valued voxels become ordinary background targets, and unavailable classes
are supervised as absent throughout the scan. If an unavailable ROI nevertheless
appears in the source file, its voxels are also projected to background. Verified
empty SHAM lesions still receive normal negative supervision. Brain targets are
unchanged. The v1 formula on fully available one-hot targets normalizes over all
classes, so unavailable-class logits receive gradients rather than being masked.
The control requires a background-inclusive loss (v1 or ordinary `dice_ce`);
foreground-only versions cannot silently stand in for this experiment.

Compare this row against `no_classifier` to isolate masking with the auxiliary
head absent in both models. Against the full `reference`, the comparison changes
two factors. An optional commented `missing_as_background` row retains the
classifier and isolates masking against `reference`. The plan displays the actual
annotation policy and classifier setting for every row.

True annotation availability remains in the loader, manifests and saved data
contract. Full-volume evaluation still ignores missing GT anatomy for every
variant, giving a fair comparison on the same annotated structures. Native patch
losses/metrics instead describe each training objective: control patch scores
include its assumed background/negative targets and must not be interpreted as
validation of unannotated anatomy. New controls require a fresh study folder;
this change does not run training or overwrite previous results.

Validation loss selects checkpoints separately within each objective. Loss
magnitudes across objectives are not comparable performance measurements. Use
the shared segmentation metrics for those comparisons; native patch losses and
metrics remain available with their own definitions.

## Initialization and matching

Every reference, component and masking-control model in 07/07f starts from
fresh random weights and a fresh optimizer. No TBI checkpoint, architecture-matched
source family or per-fold initialization map is required. Shared animal roles
come from the reference file, not from weights trained by notebook 01 or 02.

Notebook 05 alone retains optional TBI initialization for a separate fixed-mode
final-model experiment. It is disabled by default and is not used by the reviewer
ablation matrix. See [pretraining policy](pretraining_splits.md).

The runner reuses identical animal splits, saved evaluation patches, seeds,
callback settings and maximum epochs. Random training augmentations are not
guaranteed to be identical across different graphs. Early stopping may yield
different actual epoch counts. Altering a study matrix requires a new study
folder; resume validates the original configuration and matrix.

## Results and validation

Every variant/seed retains checkpoints, complete configuration, recovery state
and best-weight evaluations. The study exports native patch and postprocessed
full-volume region/brain tables to CSV/Excel. Classifier summaries apply to rows
with that head; SHAM lesion false-positive reports also apply to `no_classifier`.
Unavailable region annotations remain explicit rather than being scored as zero.
Use notebook 10 for animal-aware comparisons and uncertainty.

Tests cover original-network parameter counts, all four multiscale/fusion
combinations, valid probability outputs, complete feature transfer across fusion
removal, rejection of mismatched sources, partial-label objective validation,
miniature component/loss fits, native patch evaluation and completed-study resume.
No real dataset training or interpolation sensitivity analysis is launched.

Full-volume physical metrics now use the complete shared affine for nonorthogonal
grids, without resampling predictions or GT. Spacing-only metric calls retain
their orthogonal-grid convention. See [cohort reports](cohort_annotation_reports.md).
