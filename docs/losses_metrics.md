# Losses, metrics and partial-label gradient review

`src/evaluation/losses.py` and `metrics.py` now have typed factories/callbacks,
short docstrings, named intermediate values and blank lines between logical
sections. Legacy factory signatures, returned callback names, smoothing placement,
thresholds, class/background selection and reduction formulas are preserved.

Combined losses construct their component callbacks once. Metrics reuse the
loss module's masking function, which remains importable through either module.
The repeated clip/round/sum calculation has one private metric helper.

## Loop execution fixes without formula changes

Class iteration now uses integer TensorFlow ranges; division retains the float32
class count. The original float-valued Python ranges failed in eager execution.
Batch Tversky additionally had an integer loop accumulator incompatible with
floating scores under AutoGraph, and per-class Python-list indexing could fail
with tensor indexes. A floating accumulator and tensor alpha lookup make these
paths executable while retaining their per-class equations and averaging.

## Numerical verification

The original files were saved before editing. `tests/fixtures/objective_baselines.json`
records reference inputs, values, callback names and loss gradients:

- 85 configurations were compared directly with the original executable
  callbacks, including macro/micro metrics, background exclusion, partial-label
  masking, Dice modes, focal options and combined losses.
- Four batch-Tversky configurations could not execute through the original outer
  loop. Their references were obtained from the original per-class score helper,
  accumulated in Python, with the original CE/focal formula where applicable.
  Those records explicitly identify this reference method.
- Refactored callbacks match these references in eager and graph execution to
  float32 tolerance (`rtol=1e-6`, `atol=1e-7`).

`python -m pytest tests/test_losses_metrics.py -q` passes 185 checks, including
dynamic class counts, existing empty-class behavior, the current model output
activation, and the gradient sanity check below.

## Reviewer 2: confirmed softmax coupling

The `regions` head in `src/models/networks.py` uses softmax. The serialized
`regions` layer in `models/rnet_da.h5` also records `activation: softmax`.

`DA_3DRNET_Training.ipynb` configures the ROI objective as
`diceCELoss(smooth=1e-5, batch_wise=True, gamma=0.8, domain_adaptation=True)`.
The generator encodes unavailable target channels as -1. The loss subsequently
zeros their target values and **already-activated probabilities**. It does not
remove their logits before softmax.

The sanity check used this exact objective with one voxel and eleven channels.
It used unavailable channel indexes `[0, 2, 4, 5, 6, 8, 9]`, corresponding to
the partial-label configuration in that notebook. Channel 1 was the positive
target; other available channels were zero.

| Quantity | All logits zero | Only unavailable logit 2 changed to 2 |
|---|---:|---:|
| Probability of valid target class 1 | 0.0909091 | 0.0575074 |
| Loss | 0.1168043 | 0.1511553 |
| Gradient with respect to unavailable probability 2 | 0 | 0 |
| Gradient with respect to unavailable logit 2 | 0.0068630 | 0.0316902 |

Full values and gradients are saved in [partial_label_gradient_check.json](partial_label_gradient_check.json).
The gradient regression test also agrees with a finite-difference derivative.
The same coupling was observed using the original loss before refactoring.

For softmax probabilities `p = softmax(z)`, a masked channel `j` can have
`dL/dp_j = 0` while `dL/dz_j = -p_j * sum_i(p_i * dL/dp_i)` remains nonzero.
Its logit still changes the normalization of valid probabilities.

**Conclusion:** the legacy implementation does not guarantee zero ROI-loss
gradient for unannotated logits. This is a property of the implemented loss and
activation, not a numerical artifact introduced by the refactor. The check does
not measure segmentation performance or independently establish the provenance
of every historical training run.

If MouseFull results must satisfy that zero-gradient requirement, the objective
needs correction and the affected model must be trained/evaluated under the
corrected protocol. Check the actual run configuration before deciding which
reported results require replacement. The separate correction below changes the
training objective while retaining the inference softmax and architecture.

## Other existing behaviors retained for later decisions

- Dice loss uses `1 - 2 * (intersection + smooth) / (union + smooth)`.
  Empty target and prediction yield -1, rather than zero. Metric Dice has a
  different smoothing placement and yields 1 for the same empty inputs.
- Zeroed unavailable classes still participate in class averaging. They are
  not excluded from the denominator. Fully ignored channels contribute the
  existing empty-class constants; metrics such as Dice/IoU can reward them
  as perfect empty matches.
- CE/focal losses average over every element, including the class dimension,
  instead of summing the class contributions before averaging voxels.
- Macro metrics pool the batch within each class, then average class scores.
  They are not means of full-volume subject-level metrics.
- Binary overlap metrics use a per-channel `>0.5` threshold, not multiclass
  argmax. Confusion-based metrics retain backend rounding after clipping.
- `class_index` is applied after optional background removal, so its meaning
  depends on `exclude_background`.

The original refactor did not add a new objective or launch training. The
subsequent opt-in correction is described below; the original loss and metrics
remain available with their existing computations.

## Version-2 foreground partial-label gradient correction

`diceCELoss_gradient_fix` in `src/evaluation/losses.py` consumes **raw logits**.
It replaces unavailable logits with negative infinity before softmax, removing
them from the normalization. `masked_softmax_gradient_fix` exposes the same
conditional probabilities for inspection. All-ignored vectors are handled
explicitly without producing NaNs. Half precision is promoted to float32.

The target is a channels-last 3D batch `(N, X, Y, Z, C)`: `-1` means unavailable,
and available entries lie in `[0, 1]`. The sentinel can be configured with
`ignore_value`. **Channel zero must remain available wherever foreground ROIs
are available.** It is a competing output, not an explicit Dice or CE target.
Removing it would leave no trainable background competitor when only one ROI
is annotated. An entirely unavailable vector is still permitted.

An available foreground ROI means complete annotation throughout the volume:
its positive voxels and its negative entries both supply Dice supervision.
Zero-valued source voxels need not identify anatomical background; they may be
unselected anatomy or unavailable regions. Those voxels are nevertheless known
negatives for a completely annotated selected ROI. This assumption does not
support scribbles, limited annotated slices or incomplete ROI contours without
an additional spatial validity mask.

Foreground Dice uses `(2 * intersection + smooth) / (prediction + truth + smooth)`
and averages only available foreground sample/class groups (classes when
`batch_wise=True`). Annotated empty groups remain in the mean and retain
negative-only supervision. Perfect empty reference/prediction gives zero Dice
loss, rather than the legacy doubled-smoothing constant. Empty-ROI gradients
can be small when prediction mass greatly exceeds smoothing; gradient checks
do not establish practical convergence.

CE/focal uses only voxels with a positive available foreground target. There is
no background-target CE. It retains class-mean scaling: sum foreground CE terms
and divide by the number of available foreground entries at those positive
voxels. For one positive voxel and two available foreground classes, the CE is
`-log(p_target) / 2`. Batches with no foreground positives have zero CE but still
receive Dice supervision if an ROI is annotated. No foreground annotations at
all gives zero combined loss and gradient. Background receives indirect gradients
through normalization; unavailable anatomical logits receive exactly zero ROI
loss gradient. Zeroing only post-softmax terms cannot provide that guarantee.

This objective intentionally differs from both the version-1 correction and the
legacy loss, including on complete targets. Gamma/focal options remain available,
but changed supervision and reductions mean historical loss magnitudes are not
directly comparable. Legacy loss factories retain their original computations.

Use [check_partial_label_loss_gradient.ipynb](../notebook/check_partial_label_loss_gradient.ipynb)
for probability/gradient tables, native Plotly comparisons, finite differences,
extreme-logit checks, foreground/background cases, false-positive penalties,
mixed annotation patterns, and an example training view of the real network.
The training view shares
weights and exposes the tensor before the existing `regions` softmax through a
linear float32 output. Original builders and inference outputs are unchanged.
Do not attach the new loss directly to an existing softmax output, or attach
probability-based ROI metrics unchanged to the logits output. Saving the training
view produces a logits-output model; save/use the original inference view when
probabilities are required.

`PartialLabelDice` supplies epoch-pooled macro soft Dice for logits-based training.
It uses full softmax probabilities, then masks unavailable target entries,
retaining available ROI negatives even without a positive target.
Background and classes with no
supervision are excluded from the macro mean. Its accumulators pool batches
within each class, avoiding dilution by entirely unavailable classes/samples.
The experiment builder logs `regions_logits_mean_dice` and
`val_regions_logits_mean_dice`. Metric inputs are detached; ROI loss values,
gradients, network weights and inference outputs remain unchanged. This is a
patch-level training monitor, not a substitute for full-volume animal Dice.

New experiments record `LossConfig.gradient_fix_version=2`. Older saved
`dice_ce_gradient_fix` configurations without this field are identified as
version 1: offline evaluation and backbone initialization remain available,
but fitting/resuming under the revised objective requires a new run/study.
No old checkpoint is retroactively corrected by the new loss. Legacy
`dice_ce` configurations still load and resume under their existing objective.

The zero-gradient contract applies to the ROI loss derivative with respect to
unavailable ROI logits. Auxiliary tasks and fully annotated samples can still
update shared weights. These checks do not establish model performance; no
retraining or sensitivity study has been performed.
