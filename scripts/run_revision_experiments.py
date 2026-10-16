"""Edit the variables below, inspect the plan, then launch a draft or full suite."""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path


REPOSITORY: Path = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY))

from src.experiments.revision_config import DraftConfig, RevisionSuiteConfig
from src.experiments.revision_suite import run_revision_suite


# Inputs and output identity. Original D4 remains independent of development.
DATA_ROOT: Path = Path("D:/Other files/TESI_CNN/MANUSCRIPT_DATASET/normalized")
REFERENCE: Path = REPOSITORY / "example/RARE/TBI_fm_19_49/Anat/TBI_fm_19_49_N4.nii.gz"
OUTPUT_ROOT: Path = REPOSITORY / "results/revision_suite"
RUN_NAME: str = "revision_v1"

# None selects all animals in that cohort. Use canonical animal IDs, not scans.
D1_IDS: tuple[str, ...] | None = None
SHAM_IDS: tuple[str, ...] | None = None
RAT_IDS: tuple[str, ...] | None = None
D4_IDS: tuple[str, ...] | None = None
ANIMAL_IDS: dict[str, str] = {}  # Only expert-confirmed cross-name animal links.

# Supply one existing D1 or joint MouseFull CV plan, or anchor fold 0 to a fixed
# split. The suite generates the missing cohort assignments around that plan.
FOLDS_FILE: Path | None = None
ANCHOR_SPLIT_FILE: Path | None = None
RAT_FOLDS_FILE: Path | None = None
FOLDS: int = 5
SPLIT_SEED: int = 42
TRAINING_SEEDS: tuple[int, ...] = (42,)
INNER_VALIDATION_FRACTION: float = 0.2

# Full-study budgets. Validation/test always cover every selected sample.
D1_EPOCHS: int = 100
MOUSE_FULL_EPOCHS: int = 300
RAT_EPOCHS: int = 100
INTERPOLATION_EPOCHS: int = 10
INTERPOLATION_STEPS: int | None = 4
INTERPOLATION_ORDERS: tuple[int, ...] = (0, 1, 2, 3)
DATA_MULTIPLIER: int = 3

ARCHITECTURE_ABLATIONS: tuple[str, ...] = (
    "no_attention", "no_residual", "no_multiscale", "no_deep_supervision",
)
MOUSE_FULL_CONTROLS: tuple[str, ...] = (
    "no_classifier", "missing_as_background", "missing_as_background_no_classifier",
)
INCLUDE_INTERPOLATION: bool = True
INCLUDE_RATS: bool = True
INCLUDE_RELEASE: bool = True
INCLUDE_D4: bool = True
INCLUDE_ANNOTATED_VOLUMES: bool = True
INTER_RATER_MANIFEST: Path | None = None
ANNOTATION_SPATIAL_UNIT: str | None = None  # Explicit verified override for unknown NIfTI units.

# Optional collaborator results, already in the shared full-volume CSV format.
# Keys identify the task/variant; files retain real seed/fold/animal identities.
EXTERNAL_RESULTS: dict[str, Path] = {}  # Example: {"d1/nnunet": Path("...")}

# No ensemble. Refit durations use median inner-validation best epochs, rounded
# up; neither D4 nor outer-test metrics select the release model or its duration.
RELEASE_SEED: int = 42
RAT_SOURCE_FOLD: int = 0
MIXED_PRECISION: bool = False
SAVE_PREDICTIONS: bool = True

# Tiny real-data central fields and small networks. Drafts have their own root.
DRAFT = DraftConfig(
    folds=2, d1_animals=6, sham_animals=6, rat_animals=3, external_animals=2,
    epochs=1, steps_per_epoch=1, patch_size=(16, 16, 16), filters=(2, 4, 8, 16, 32),
)


def configuration() -> RevisionSuiteConfig:
    """Collect the visible settings without fitting or changing any input data."""
    return RevisionSuiteConfig(
        data_root=DATA_ROOT, reference=REFERENCE, output_root=OUTPUT_ROOT, run_name=RUN_NAME,
        d1_ids=D1_IDS, sham_ids=SHAM_IDS, rat_ids=RAT_IDS, external_ids=D4_IDS,
        animal_ids=ANIMAL_IDS, folds_file=FOLDS_FILE, anchor_split_file=ANCHOR_SPLIT_FILE,
        rat_folds_file=RAT_FOLDS_FILE, folds=FOLDS, split_seed=SPLIT_SEED,
        seeds=TRAINING_SEEDS, validation_fraction=INNER_VALIDATION_FRACTION,
        d1_epochs=D1_EPOCHS, mouse_full_epochs=MOUSE_FULL_EPOCHS, rat_epochs=RAT_EPOCHS,
        interpolation_epochs=INTERPOLATION_EPOCHS, interpolation_steps=INTERPOLATION_STEPS,
        interpolation_orders=INTERPOLATION_ORDERS, data_multiplier=DATA_MULTIPLIER,
        architecture_ablations=ARCHITECTURE_ABLATIONS, supervision_controls=MOUSE_FULL_CONTROLS,
        include_interpolation=INCLUDE_INTERPOLATION, include_rats=INCLUDE_RATS,
        include_release=INCLUDE_RELEASE, include_external=INCLUDE_D4,
        include_annotation_volumes=INCLUDE_ANNOTATED_VOLUMES,
        inter_rater_manifest=INTER_RATER_MANIFEST, annotation_spatial_unit=ANNOTATION_SPATIAL_UNIT,
        external_results=EXTERNAL_RESULTS, release_seed=RELEASE_SEED, rat_source_fold=RAT_SOURCE_FOLD,
        mixed_precision=MIXED_PRECISION, save_predictions=SAVE_PREDICTIONS, draft=DRAFT,
    )


def main() -> None:
    """Default to planning; explicit --run or --quick starts sequential fitting."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="Run the full study; otherwise only save its plan")
    parser.add_argument("--quick", action="store_true", help="Run the complete draft sequence with tiny real-data copies")
    parser.add_argument("--plan-only", action="store_true", help="Save memberships and the experiment plan without fitting")
    parser.add_argument("--resume", action="store_true", help="Resume the exact saved suite, including after planning")
    parser.add_argument("--summarize-only", action="store_true", help="Read available results and create a fresh compact summary")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--run-name")
    parser.add_argument("--folds-file", type=Path)
    parser.add_argument("--anchor-split-file", type=Path)
    args = parser.parse_args()
    if args.run and args.quick:
        parser.error("Choose --run or --quick")
    settings = configuration()
    overrides = {}
    for name in ("data_root", "output_root", "reference", "run_name", "folds_file", "anchor_split_file"):
        value = getattr(args, name)
        if value is not None:
            overrides[name] = value
    settings = replace(settings, **overrides)
    mode = "full"
    if args.quick:
        mode = "draft"
    if args.summarize_only:
        from src.experiments.revision_summary import export_suite_summary

        folder = settings.output_root / mode / settings.run_name
        samples = settings.bootstrap_samples
        if mode == "draft":
            samples = 100
        summary = export_suite_summary(folder, bootstrap_samples=samples)
        print("Compact paper tables:", summary)
        return

    # These callbacks are visible and editable. Only inner-validation loss drives
    # LR changes. Interpolation pilots and deployment refits use no LR callback.
    import tensorflow as tf

    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=10, min_lr=1e-7, verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(filename="training.csv", append=True),
    ]
    plan_only: bool = args.plan_only or (not args.run and not args.quick)
    folder = run_revision_suite(
        settings, mode=mode, callbacks=callbacks, resume=args.resume, plan_only=plan_only,
    )
    print("Results:", folder)


if __name__ == "__main__":
    main()
