# Generator behavior and review

`src/preprocessing/generator.py` keeps the existing constructor arguments and
Keras `Sequence` interface. `DomainAdaptationGenerator` now inherits batch
loading, preprocessing, allocation and shuffling from `MultiTaskGenerator`.
Its only customization is preparing missing-label channels and condition targets.
Functions have types and docstrings, computed arguments have names, and blank
lines separate initialization, validation, processing and returns.

## Preserved behavior

- Outputs remain `(images, [regions, brain_mask])` for multitask batches and
  `(images, [regions, brain_mask, condition])` for domain adaptation.
- Output arrays remain float64. One image is repeated across input channels;
  configuring several channels does not load several MRI modalities.
- Only complete batches are returned. The final remainder is dropped, and a
  collection smaller than `batch_size` has zero batches.
- Shuffling uses NumPy's global random state. `__getcase__` returns an ID in
  original list order, independently of shuffled batch indexes.
- Domain-adaptation condition targets remain 0 for SHAM and 1 otherwise.
- Unavailable source labels are matched to positions in `config['labels']`;
  the corresponding complete output channels receive the sentinel -1.

## Small fixes

- `preprocessor=None` now supports already prepared inputs.
- Invalid batch indexes raise `IndexError` instead of producing empty or
  partly zero-filled batches beyond the advertised sequence length.
- Batch size, channel count, class count and spatial dimensions must be
  positive integers; the configured input shape must have three axes.
- Missing brain masks, unexpected shapes and nonfinite arrays produce explicit
  errors identifying the subject.
- ROI values must be integer class indexes in `[0, num_classes)` before
  one-hot encoding. Negative indexes and fractional values cannot silently
  select the wrong class. Use `MapLabels` to convert original mask IDs.
- One-hot encoding retains singleton spatial axes by flattening and explicitly
  restoring the configured shape.
- Subject IDs are copied so later changes to the caller's list cannot invalidate
  the generator's indexes.

## Remaining issues and decisions

- Dropping incomplete batches can omit validation/test subjects. Changing this
  policy requires an explicit batching decision; this refactor preserves it.
- Missing-label encoding assumes source-label order corresponds to output
  channel order. A many-to-one or reordered label mapping needs a separate
  availability policy. An ignored channel beyond `num_classes` now raises a
  clear error, but valid channel bounds do not establish correct label semantics.
- The normalized loader does not infer unavailable labels from metadata.
  The generator uses only explicitly supplied `unavailable_labels`. The -1
  encoding alone does not verify that training losses ignore those channels.
- Shape checks do not establish physical alignment. Spatial preprocessing
  checks image/mask affines; callers bypassing it must supply aligned volumes.
- Float64 output allocation is retained for compatibility. Switching to float32
  could reduce batch memory and should be considered separately.

## Validation

Run `python -m pytest tests/test_generator.py tests/test_loader.py -q` in the
`rnet` environment. The 39 tests include both dataset layouts, both generators,
singleton-axis encoding, repeated channels, ignored channels, SHAM/TBI flags,
batch bounds, invalid labels, missing masks and shuffle/list ownership.

No model training, loss changes or notebook edits were made for this refactor.
