# Exploring encoder features

Open [check_feature_space.ipynb](../notebook/check_feature_space.ipynb) with the
`rnet` kernel. It loads one selected checkpoint, extracts deterministic center-patch
features and plots an interactive projection with native Plotly. It never fits,
compiles or initializes a training model. Classifier-free models are supported.

1. Choose `RUN_DIR` and `FOLD`, or an explicit weights/configuration pair. For a
   complete legacy export, use `RUN_DIR=None`, `CHECKPOINT_KIND="model"` and
   `MODEL_FILE`. The run loader follows its saved checkpoint selection.
2. Choose normalized `DATASETS` or `CSV_MANIFEST`. A CSV requires `Subject`, `Scan`,
   `Dataset` and `ImagePath`; image paths can be relative to the CSV. CSV inputs
   need no annotation masks. Animal IDs must be unique across datasets, and all
   timepoints of selected animals remain present. Reproducible sampling defaults
   to at most 12 animals per dataset; use `MAX_ANIMALS_PER_DATASET=None` for all.
3. Review scan/animal counts and the visible preprocessing pipeline. The shared
   RIA reference, saved spacing, unit correction and normalization are used with
   linear MRI resampling and a deterministic center crop/pad. Legacy exports
   have explicit fallbacks. There is no augmentation.
4. Set `RUN_FEATURE_EXTRACTION=True`. `FEATURE_LAYER=None` uses the encoder
   bottleneck before classifier pooling, or the first decoder up-convolution
   when there is no classifier. The table lists other encoder features.
5. Project and select `COLOR_BY`/`SYMBOL_BY` to explore condition, dataset,
   modality, timepoint or saved split role. Changing colors does not refit the
   projection. Hover identifies each animal and its scans.

`extract_scan_features` takes the spatial mean of each feature channel, one
forward pass per scan with `training=False`. Only one scan patch is held at a
time. These features describe a **center patch**, not the entire brain. They
contain no annotation targets, classifier probabilities or metadata encodings.
Do not combine features from independently trained models/folds in one matrix:
their learned channel coordinates need not correspond.

`AGGREGATION="animal"` averages all selected scan vectors within each animal
before projection. This gives each animal one point regardless of longitudinal
scan count. Varying timepoints/modalities are listed together in hover; conflicting
condition, dataset or split roles raise errors. `"scan"` retains separate points for
timepoint exploration, with repeated animals explicitly identifiable. Plot titles
record both animal and scan counts.

`PROJECTION_METHOD="pca"` uses deterministic full SVD and reports explained
variance. `STANDARDIZE=True` scales feature channels before reduction; it is
explicit because [PCA centers but does not scale inputs](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html).
Scaling follows [StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html).
At least three selected points and varying feature vectors are required.

Optional `"umap"` uses an already installed compatible `umap-learn`, with visible
seed, neighbor and minimum-distance settings. No package is installed or upgraded.
At least four points are required, and the effective neighbor count is recorded.
Both reducers receive **only feature vectors**; metadata is attached afterwards.
UMAP can introduce apparent clusters or distort density: see its
[clustering guidance](https://umap-learn.readthedocs.io/en/latest/clustering.html).

Saved `splits.json` labels membership for the selected run/fold. Unlisted scans
are labeled `not_in_saved_split`, which does not audit source pretraining or
certify independence. `ROLE_FILTER` can restrict the view. Apparent separation
does not demonstrate classification accuracy or anatomical reasoning; cohort,
condition and acquisition may be confounded. These are exploratory projections,
not a replacement for held-out evaluation.

Export is optional: set `EXPORT_FEATURE_SPACE=True` and a fresh `FEATURE_OUTPUT`.
It saves scan vectors, their manifest, projection coordinates and provenance with
checkpoint/reference SHA-256 hashes and preprocessing/reducer settings. Existing
folders are rejected. Nothing is written by default.

Tests cover exact channel pooling, unchanged weights, animal averaging, repeated
scan counts, metadata-independent PCA, optional UMAP configuration, and both
normalized/CSV notebook flows on synthetic saved models. No training is needed.
