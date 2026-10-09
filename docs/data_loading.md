# Loading rodent datasets

`src/utils/loader.py` supports the historical folders and the normalized dataset
through the same `RodentDatasets` collection and `Rodent` scan records. No
TensorFlow dependency is needed for loading. Python 3.10 or newer is required.

## Historical layout

```text
root/
  FLASH/
    mouse_id/
      Anat/
        mouse_id_N4.nii.gz
        mouse_id_Labels.nii.gz
        mouse_id_brain_mask.nii.gz
```

From the repository root:

```python
from src.utils.loader import RodentDatasets

data = RodentDatasets(labels=[0, 1, 3, 13, 21])
data.add_dataset("example", "FLASH", sub_folder="Anat")
scan = data.get_subject(data.get_subjects_list()[0])
image, labels, brain_mask = scan.get_images()
```

Existing notebook imports (`from utils.loader import ...`, with `src` on the
Python path) and positional arguments continue to work. Custom suffixes,
`gz=False`, explicit `sham` and `unavailable_labels` remain supported.
Use `brain_mask_suffix=None` if the mask is not available.

## Normalized layout

```text
normalized/
  mapping.xlsx
  D1/SUBJ_0001/
    metadata.json
    N4.nii.gz
    Labels.nii.gz
    brain_mask.nii.gz
  D2/SUBJ_0113/
    metadata.json
    7days/{N4,Labels,brain_mask}.nii.gz
    14days/{N4,Labels,brain_mask}.nii.gz
```

```python
from pathlib import Path
from src.utils.loader import RodentDatasets

root = Path(r"D:\Other files\TESI_CNN\MANUSCRIPT_DATASET\normalized")
data = RodentDatasets(labels=[0, 1, 3, 13, 21])
data.add_normalized_dataset(root, "D1")
data.add_normalized_dataset(root, "D2")
# Optional selection: modality="T1w" or modality="T2w".

scan = data.get_subject(("D2_T1w", "SUBJ_0113_7days"))
print(scan.subject_id)     # SUBJ_0113
print(scan.original_id)    # RAT-1
print(scan.timepoint)      # 7days
print(scan.subject_group)  # ('D2', 'SUBJ_0113')
print(scan.metadata["annotated_regions"])
```

Choose datasets explicitly; the loader does not automatically include D4 or
the separate `nnUNet_raw` exports. The latter are not an input layout supported
by this loader. Use `image_name="anat_orig"` to select the original image,
`brain_mask_name=None` to omit brain masks, or `gz=False` for uncompressed files.

Each normalized subject must have `metadata.json` containing `normalized_id`,
`original_id`, `dataset`, `modality`, and `condition` (`TBI` or `SHAM`). An absent
or empty `timepoints` list denotes one scan directly in the subject folder;
otherwise each listed timepoint becomes a separate scan. Metadata IDs must
match the dataset and subject folders. The loader uses these JSON records
directly; it does not require Excel libraries or reconcile `mapping.xlsx` on
every load. Original IDs and all annotation metadata remain accessible.

## Shared behavior and compatibility

- Generators still use `get_subjects_list()`, `get_subject(key)` and
  `get_images()`. Both layouts can coexist in one collection.
- Legacy keys remain `(modality, scan_id)`. Normalized keys are
  `(dataset_modality, scan_id)`, e.g. `('D1_T1w', 'SUBJ_0001')`.
  `get_modalities()` retains its historical name but returns these collection
  group names. A scan's `get_modality()` returns the actual acquisition modality.
- A longitudinal scan ID includes its timepoint; `subject_group` excludes it.
  `SubjectIndex` in `evaluation.k_fold` uses this group when expanding
  subject-level folds to generator scan keys. See
  [cross-validation integration](cross_validation.md). Existing notebook
  splitting behavior is unchanged.
- D3 contains repeated original ID strings across subgroups. The loader
  preserves the supplied normalized identities and subgroup metadata without
  merging them or asserting that they identify different animals.
- Files are opened as NIfTI proxies. Voxel arrays are read when requested.
  Intensities, labels, geometry, and files on disk are not transformed.
- SHAM status comes from normalized metadata. Missing-label masks are **not**
  inferred from label counts or absent voxel values. The optional
  `unavailable_labels` argument applies to every scan in that registration;
  mixed annotation configurations need a separate, explicit training policy.
  The `labels` constructor argument stores a vocabulary; it does not remap data.
- Subject folders are sorted for stable discovery. This differs from the old
  filesystem-dependent ordering and can change a newly generated random split
  even with the same seed. Historical split IDs should be reused explicitly.
- Registrations merge distinct keys and reject duplicate keys instead of
  replacing a modality's previous contents. Missing required files or invalid
  metadata abort that registration without adding any of its scans.

The inspection helper also supports normalized names:

```python
from src.utils.loader import load_data

images, arrays, paths = load_data(
    root / "D2" / "SUBJ_0113" / "7days",
    "SUBJ_0113",
    ["N4", "Labels", "brain_mask"],
    normalized=True,
)
```

Here the subject ID is retained for API compatibility and is not used in
normalized filenames. As before, this helper prints and skips missing files;
collection loading requires all requested files to exist.

## Verification

Using a Python environment with nibabel, NumPy and pytest installed:

```text
python -m pytest tests/test_loader.py -q
```

Tests use temporary synthetic NIfTI files. They do not alter the real dataset.
Four generator compatibility cases also run when TensorFlow is installed;
otherwise pytest reports those cases as skipped.
