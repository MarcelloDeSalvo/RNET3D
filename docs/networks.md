# Network blocks and ablation preparation

`src/models/networks.py` exposes `mt_r_net_3d` and `standard_unet_3d`.
The unused double-decoder builder has been removed. Function signatures, output
order, head names and float32 output activations remain compatible with the
training notebooks.

Outputs are `[regions, brain_mask]`, with `classifier` appended when requested.
Concrete input dimensions should be divisible by 16, with channels last.

## Corrected behavior

1. **Convolution activation:** each `conv_block` repetition applies convolution,
   optional batch normalization and the configured activation. Strings such as
   `"relu"` and `"sigmoid"` now work. `True` selects ReLU; `False` and `None`
   disable activation. Projections that explicitly pass `None` remain linear.
2. **Bottleneck and decoder dropout:** a nonzero `dropout` rate adds
   `SpatialDropout3D` after feature processing. It drops complete channels during
   training and is an identity during inference. Zero adds no layer. Existing
   builders still pass zero for these blocks; encoder and classifier dropout
   settings are unchanged.
3. **Complete residual toggle:** `mt_r_net_3d(residual=False)` now selects two
   plain convolutions in each multiscale branch, as it already does in the
   encoder, bottleneck and decoder. The flag removes residual shortcuts while
   retaining U-Net skip concatenations, attention additions and head fusion.

These are intentional changes to the networks built from code. The previous
readability-only refactor preserved the old graph, but its numerical-equivalence
claim does not apply after these fixes. Existing serialized model files are not
modified, and historical training/evaluation metrics have not been recomputed.

## Component controls

| Component | Location and scope |
|---|---|
| Multiscale inputs | `multiscale=True`: three pooled-input branches and encoder concatenations |
| Residual connections | Builder flag controls multiscale, encoder, bottleneck and decoder |
| Attention | `attention_block`, controlled by the existing decoder flag |
| Downsampling | `encoder_block`: max pooling or strided convolution |
| Upsampling | `decoder_block`: transpose convolution or upsampling plus convolution |
| Spatial dropout | Encoder after downsampling; bottleneck/decoder after feature processing |
| Deep supervision | `deep_supervision=True`: sum four projected decoder levels in each segmentation head |
| Condition classifier | `classifier_block`, controlled by `with_classifier` |

`_segmentation_head` keeps the final unregularized 1x1x1 projection and named
activation in one place. Deep supervision currently means fused multiresolution
logits; the builder does not expose four separately supervised outputs or
auxiliary losses. `deep_supervision=False` removes the three coarse projections
and their sum, retaining the original finest projection and final output layer
for both regions and brain. `multiscale=False` removes all three input branches
and their encoder concatenations. These switches are independent and default
to `True`, preserving the graph and weights of the existing full model.
They apply only to the 3DR-Net builder; conventional U-Net has neither component.

Saved configurations without the new fields load with both flags enabled.
Backbone transfer uses the common finest decoder boundary, whose ancestors
include the complete encoder/decoder. Head-fusion and classifier removals can
reuse reference TBI features while keeping output heads fresh. Multiscale removal
changes backbone weights and requires a matching source; silent partial copying
is rejected. See [the matched ablation workflow](ablation.md).

## Validation

Run `python -m pytest tests/test_networks.py -q` in the `rnet` environment.
Tests cover all residual/attention/pooling combinations for both builders,
classifier output, actual convolution activation values, convolution/normalization/
activation order, connected dropout and its training/inference behavior, multiscale
shortcut removal, and three-head forward passes with every combination of the
multiscale and head-fusion switches. Study tests cover fitting, native patch
evaluation and completed-run resumption with the new components and loss rows.

`tests/fixtures/network_graphs.json` records historical graphs from before these
fixes. Their graph hashes are historical evidence, not expectations for the
corrected models. Tests still use their parameter counts: activations and dropout
add no weights; disabling multiscale residual shortcuts removes their projection
weights. For one input channel and test widths `[2, 4, 8, 16, 32]`, that removes
56 parameters from the nonresidual multitask model.

Validation uses miniature synthetic fits; no dataset training or sensitivity
analysis was performed.
