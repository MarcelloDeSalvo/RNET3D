# Matched component and loss ablations

Notebook 07 uses D1 and the historical four-class task. Notebook 07f uses
MouseFull D1+D3, eleven classes, corrected v1 loss and TBI feature initialization.
Both keep `RUN_STUDY=False` until explicitly enabled. They expose complete
configurations, the experiment matrix and actual Keras callbacks.

## Default matrix

Every row starts from the full reference and changes one setting. Disabling
several components sequentially would answer a different question.

| Row | Change | Retained behavior |
|---|---|---|
| `reference` | None | Original 3DR-Net settings and filter widths |
| `no_residual` | `residual=False` | Encoder-decoder skips, attention and head fusion |
| `no_attention` | `attention=False` | Residual blocks and skips |
| `no_multiscale` | `multiscale=False` | Encoder/decoder and their skip connections |
| `no_deep_supervision` | `deep_supervision=False` | Finest-level projection and final output layer in both heads |
| `strided_downsampling` | `max_pooling=False` | Resolution levels; pooling is replaced by strided convolution |
| `no_classifier` (07f) | `classifier=False` | Region and brain heads, partial-label supervision |

The three multiscale pooled-input branches and their encoder concatenations are
absent in `no_multiscale`. The head-fusion ablation removes the three coarse
decoder projections and their sum from each segmentation head. The original
finest projection remains. The original builder's deep supervision consists of
fused multiresolution logits, not separately supervised auxiliary outputs.

Both new flags default to `True`; existing full-model graphs, outputs, weight
ordering, filter widths and dropout settings remain unchanged. Older saved
configurations without these fields load with both enabled. Conventional U-Net
uses its existing builder and does not use these flags.

Notebook 07 has six default variants; 07f has seven. With five outer folds and
one seed this requires 30 or 35 fitted models, respectively. Optional rows add to
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
switching objectives must not erase missing-label semantics.

Validation loss selects checkpoints separately within each objective. Loss
magnitudes across objectives are not comparable performance measurements. Use
the shared segmentation metrics for those comparisons; native patch losses and
metrics remain available with their own definitions.

## Initialization and matching

07f retains `USE_TBI_PRETRAINING=True`. Every selected variant/fold needs an
auditable source excluding target holdout/test animals and never fitting target
validation animals. Source selection-validation may be shared across stages.
Residual, attention, downsampling and multiscale changes require matching TBI
backbones. Configure `TBI_VARIANT_RUNS` for a fixed split or
`TBI_VARIANT_FOLD_SOURCES` for per-fold sources.

Head-fusion, classifier and loss changes can reuse the reference TBI backbone.
Complete encoder/decoder weights transfer through the finest decoder boundary;
region, brain and classifier heads remain fresh. Source graphs and weight shapes
are validated before any weights are copied. Sources with a different backbone
are rejected rather than partially copied. Set `USE_TBI_PRETRAINING=False` only
for an explicitly chosen scratch study.

Fresh study splits inherit memberships from every selected source, rejecting
conflicting assignments. Matching source CV folds are reused before splitting
additional animals. See [pretraining split selection](pretraining_splits.md).

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

Full-volume physical distance metrics still require an orthogonal measurement
grid; accepting nonorthogonal input affines for augmentation does not remove
that separate metric requirement. A dataset hitting that check needs a geometry
evaluation fix before its complete full-volume study can run.
