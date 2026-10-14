# MouseFull ventricle mapping audit

Checked on 2026-10-14 against normalized D1/D3 and the saved configuration for
`mouse_full_3drnet/RUN_20261013_175412_728737`.

| Region | Original mask ID | Training/logit channel | Exported mask ID |
| --- | --- | --- | --- |
| Contralateral ventricle | 3 | 3 | 3 |
| Ipsilateral ventricle | 13 | 7 | 13 |

The complete channel order is `[0, 1, 2, 3, 5, 6, 12, 13, 15, 16, 21]`, matching
the original `DA_3DRNET_Training.ipynb`. Dataset metadata has a different
descriptive `network_label` order; the experiment intentionally uses the original
`mask_value` IDs and its explicit `LabelConfig.mapping`.

- Audited all 112 D1 and 86 D3 source masks and annotation metadata. Both ventricle
  labels are available in all 198 scans. Label 3 is nonempty in every scan.
  Label 13 is empty in one D1 scan; annotated empty masks remain valid supervision.
- Metadata names consistently associate contralateral ventricles with mask value
  3 and ipsilateral ventricles with value 13.
- Real preprocessed/encoded target checks agree voxel-for-voxel: the first D1
  validation patch has 1,031 channel-3 and 184 channel-7 positive voxels; the first
  D3 patch has 1,827 and 2,057. Neither channel is ignored. These counts depend on
  the saved geometry and single center-crop preview, not the new multi-patch policy.
- `MapLabels` maps original data into a fresh output array, preventing collisions
  between source IDs and channel indexes. Generator availability masking uses
  explicit source-to-channel mappings. Inference applies softmax/argmax without
  permuting channels. MouseFull postprocessing leaves laterality separate, then
  maps channels 3 and 7 back to IDs 3 and 13.
- The saved initialization report confirms backbone transfer and reinitialized
  output heads, so the earlier four-class TBI output ordering was not transferred.

The saved evaluation's contralateral ventricle is overpredicted: mean Dice is
0.001904, sensitivity 0.932664, predicted volume 2,482.296 mm³ and reference volume
2.533 mm³. In D1/SUBJ_0008, 87.712% of label-3 predictions are outside its GT brain
mask. This supports investigating false-positive/background supervision rather
than a missing label/channel. These checks do not establish the anatomical
correctness of source annotations or prove that changing the loss fixes convergence.

`tests/test_mouse_full_label_mapping.py` verifies normalized metadata, ROI mapping,
one-hot encoding, unavailable-class masking, argmax and original-ID export together
for both complete and partial annotation patterns.
