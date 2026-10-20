"""Preview proportional mouse splits; --apply backs up and replaces the references."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Sequence
from zoneinfo import ZoneInfo

import pandas as pd


REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY))

from src.evaluation.k_fold import FoldSplit
from src.experiments.data import PreparedData, prepare_data
from src.experiments.revision_config import RevisionSuiteConfig, experiment_configuration
from src.experiments.revision_splits import make_suite_splits, split_membership
from src.experiments.splits import select_animals, validate_folds


DATA_ROOT = Path("D:/Other files/TESI_CNN/MANUSCRIPT_DATASET/normalized")
REFERENCE = REPOSITORY / "example/RARE/TBI_fm_19_49/Anat/TBI_fm_19_49_N4.nii.gz"


def json_bytes(payload: object) -> bytes:
    """Use one stable encoding for digests, staged files and final references."""
    text = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    return text.encode("utf-8")


def reference_files(
    data: PreparedData, folds: Sequence[FoldSplit], name: str, *, seed: int,
    validation_fraction: float, created: str,
) -> tuple[dict[str, bytes], pd.DataFrame]:
    """Serialize a CV plan, its fixed fold zero, and scan-level balance audits."""
    validate_folds(folds, data.subjects.subject_ids)
    coverage = data.annotation_groups(include_lesion=False)
    membership = split_membership(data, folds, name)
    membership["AnatomyCoverage"] = membership["Subject"].map(coverage)
    keys = ["Fold", "DatasetRole", "Dataset", "Modality", "AnatomyCoverage"]
    balance = membership.groupby(keys, dropna=False).agg(
        Animals=("Subject", "nunique"), Scans=("Scan", "size"),
    ).reset_index()

    cv_name = "D1_splits_final"
    fixed_name = "D1_fixed_split_from_kfold_fold_0"
    if name == "MouseFull":
        cv_name = "MouseFull_5fold_balanced_splits"
        fixed_name = "MouseFull_fixed_split_from_kfold_fold_0"
    payload = [fold.to_dict() for fold in folds]
    cv_bytes = json_bytes(payload)
    fixed_bytes = json_bytes(payload[0])
    metadata = {
        "created": created, "generator": "scripts/regenerate_reference_splits.py",
        "seed": seed, "folds": len(folds), "validation_fraction": validation_fraction,
        "stratification": ["dataset", "anatomy_coverage", "modality"],
        "balance_policy": "Preserve cohort proportions; retain every animal; no oversampling",
        "include_lesion_in_completeness": False,
        "outer_test_policy": "Each animal is tested exactly once",
        "inner_validation_policy": "Proportional allocation within the remaining outer training pool",
        "d1_compatibility": "D1 roles are identical in the D1 and MouseFull plans",
        "complete_d1_animals": sorted(
            animal for animal, group in coverage.items()
            if animal.startswith("D1/") and group == "complete"
        ),
        "counts": balance.to_dict(orient="records"),
        "sha256": hashlib.sha256(cv_bytes).hexdigest(),
    }
    fixed_metadata = {
        "created": created, "generator": metadata["generator"],
        "source": cv_name + ".json", "source_sha256": metadata["sha256"],
        "reference_fold": 0, "seed": seed, "validation_fraction": validation_fraction,
        "stratification": metadata["stratification"],
        "sha256": hashlib.sha256(fixed_bytes).hexdigest(),
    }
    files = {
        cv_name + ".json": cv_bytes,
        cv_name + ".metadata.json": json_bytes(metadata),
        cv_name + ".balance.csv": balance.to_csv(index=False).encode("utf-8"),
        cv_name + ".membership.csv": membership.to_csv(index=False).encode("utf-8"),
        fixed_name + ".json": fixed_bytes,
        fixed_name + ".metadata.json": json_bytes(fixed_metadata),
    }
    return files, balance


def replace_references(folder: Path, files: dict[str, bytes], created: str) -> Path:
    """Verify byte-for-byte backups before replacing any reference in this folder.

    Staged files are replaced atomically one at a time. If replacement fails,
    already replaced references are restored from the verified backups.
    """
    folder = folder.resolve(strict=True)
    if any(Path(name).name != name for name in files):
        raise ValueError("Reference filenames must stay directly inside the split folder")
    targets = {name: (folder / name).resolve() for name in files}
    if any(target.parent != folder for target in targets.values()):
        raise ValueError("Reference targets must stay inside the split folder")

    stamp = datetime.fromisoformat(created).strftime("%Y%m%d_%H%M%S_%f")
    backup = folder / "split_backups" / f"modality_balanced_{stamp}"
    backup.mkdir(parents=True, exist_ok=False)
    previous: dict[str, str] = {}
    for name, target in targets.items():
        if target.exists():
            digest = hashlib.sha256(target.read_bytes()).hexdigest()
            copied = backup / name
            shutil.copy2(target, copied)
            if hashlib.sha256(copied.read_bytes()).hexdigest() != digest:
                raise OSError(f"Backup verification failed: {target}")
            previous[name] = digest
    manifest = {"created": created, "source_folder": str(folder), "sha256": previous}
    (backup / "backup_manifest.json").write_bytes(json_bytes(manifest))

    staging = backup / "new_references"
    staging.mkdir()
    for name, content in files.items():
        if name.endswith(".metadata.json"):
            metadata = json.loads(content)
            metadata["backup_folder"] = str(backup)
            content = json_bytes(metadata)
        (staging / name).write_bytes(content)

    replaced: list[str] = []
    try:
        for name, target in targets.items():
            (staging / name).replace(target)
            replaced.append(name)
    except OSError:
        for name in replaced:
            if name in previous:
                shutil.copy2(backup / name, targets[name])
            else:
                targets[name].unlink()
        raise
    return backup


def main() -> None:
    """Preview without fitting; write references only when --apply is supplied."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--reference", type=Path, default=REFERENCE)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    folder = args.data_root.resolve(strict=True).parent
    settings = RevisionSuiteConfig(
        data_root=args.data_root, reference=args.reference,
        split_seed=args.seed, validation_fraction=args.validation_fraction,
    )
    config = experiment_configuration(settings, "mouse_full", output_dir=folder)
    mouse_full = prepare_data(config)
    d1_ids = [animal for animal in mouse_full.subjects.subject_ids if animal.startswith("D1/")]
    d1 = select_animals(mouse_full, d1_ids)
    splits = make_suite_splits(
        d1, mouse_full, None, k=5, seed=args.seed,
        validation_fraction=args.validation_fraction,
    )
    created = datetime.now(ZoneInfo("Europe/Rome")).isoformat()
    files: dict[str, bytes] = {}
    for data, folds, name in ((d1, splits.d1, "D1"), (mouse_full, splits.mouse_full, "MouseFull")):
        generated, balance = reference_files(
            data, folds, name, seed=args.seed,
            validation_fraction=args.validation_fraction, created=created,
        )
        files.update(generated)
        counts = balance.groupby(["Fold", "DatasetRole", "Modality"])["Animals"].sum().unstack(fill_value=0)
        print(f"\n{name}:\n{counts.to_string()}")
    if args.apply:
        backup = replace_references(folder, files, created)
        print(f"\nUpdated {len(files)} reference/audit files in {folder}\nBackup: {backup}")
    else:
        print("\nPreview only. Use --apply to back up and replace the reference files.")


if __name__ == "__main__":
    main()
