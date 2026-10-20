"""Restore original mouse CV and derive final-model splits; --apply backs up files."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime
from itertools import permutations
from pathlib import Path
from typing import Sequence
from zoneinfo import ZoneInfo

import pandas as pd


REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY))

from src.evaluation.k_fold import FoldSplit
from src.experiments.data import PreparedData, SubjectSplit, prepare_data
from src.experiments.revision_config import RevisionSuiteConfig, experiment_configuration
from src.experiments.revision_splits import split_membership
from src.experiments.splits import (
    balanced_folds, merge_folds, read_selection, reorder_folds, select_animals,
    stratified_validation_ids, validate_folds,
)


DATA_ROOT = Path("D:/Other files/TESI_CNN/MANUSCRIPT_DATASET/normalized")
REFERENCE = REPOSITORY / "example/RARE/TBI_fm_19_49/Anat/TBI_fm_19_49_N4.nii.gz"
ORIGINAL_SPLITS = Path("C:/Users/MDesalvo/Downloads/D1_splits_final_original.json")
REFERENCE_FOLD: int = 4  # Original outer holdout with eight T2w animals.
FIXED_TEST_T2W_MINIMUM: int = 5


def json_bytes(payload: object) -> bytes:
    """Use one stable encoding for digests, staged files and final references."""
    text = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    return text.encode("utf-8")


def reference_files(
    data: PreparedData, folds: Sequence[FoldSplit], name: str, *, seed: int,
    validation_fraction: float, created: str, generation_fold_order: Sequence[int] | None = None,
    fixed_split: SubjectSplit | None = None, fixed_reference_fold: int = 0,
) -> tuple[dict[str, bytes], pd.DataFrame]:
    """Serialize CV roles, a fixed final-model plan and scan-level balance audits."""
    validate_folds(folds, data.subjects.subject_ids)
    coverage = data.annotation_groups(include_lesion=False)
    membership = split_membership(data, folds, name)
    membership["AnatomyCoverage"] = membership["Subject"].map(coverage)
    keys = ["Fold", "DatasetRole", "Dataset", "Modality", "AnatomyCoverage"]
    balance = membership.groupby(keys, dropna=False).agg(
        Animals=("Subject", "nunique"), Scans=("Scan", "size"),
    ).reset_index()

    cv_name = "D1_splits_final"
    fixed_name = f"D1_fixed_split_from_kfold_fold_{fixed_reference_fold}"
    if name == "MouseFull":
        cv_name = "MouseFull_5fold_balanced_splits"
        fixed_name = f"MouseFull_fixed_split_from_kfold_fold_{fixed_reference_fold}"
    payload = [fold.to_dict() for fold in folds]
    use_validation = any(fold.val_ids for fold in folds)
    if not use_validation:
        payload = [{"train": fold.train_ids, "test": fold.test_ids} for fold in folds]
    if generation_fold_order is None:
        generation_fold_order = list(range(len(folds)))
    reorder_folds(folds, generation_fold_order)
    cv_bytes = json_bytes(payload)
    fixed_bytes = json_bytes(payload[0])
    if fixed_split is not None:
        fixed_split.validate(data.subjects)
        fixed_payload = {
            "train": fixed_split.train, "validation": fixed_split.validation, "test": fixed_split.test,
        }
        fixed_bytes = json_bytes(fixed_payload)
    metadata = {
        "created": created, "generator": "scripts/regenerate_reference_splits.py",
        "seed": seed, "folds": len(folds), "validation_fraction": validation_fraction,
        "generation_fold_order": list(generation_fold_order),
        "use_validation_for_selection": use_validation,
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
        "reference_fold": fixed_reference_fold, "seed": seed, "validation_fraction": validation_fraction,
        "stratification": metadata["stratification"],
        "generation_fold_order": metadata["generation_fold_order"],
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
    if not use_validation:
        metadata["validation_fraction"] = 0.0
        metadata["optional_inner_validation_fraction"] = validation_fraction
        metadata["inner_validation_policy"] = "No validation pass; predetermined schedule and final-epoch weights"
        files[cv_name + ".metadata.json"] = json_bytes(metadata)
    if fixed_split is not None:
        fixed_fold = FoldSplit(
            fixed_reference_fold, list(fixed_split.train), list(fixed_split.validation), list(fixed_split.test),
        )
        fixed_membership = split_membership(data, [fixed_fold], name)
        fixed_membership["AnatomyCoverage"] = fixed_membership["Subject"].map(coverage)
        fixed_balance = fixed_membership.groupby(keys, dropna=False).agg(
            Animals=("Subject", "nunique"), Scans=("Scan", "size"),
        ).reset_index()
        fixed_metadata["counts"] = fixed_balance.to_dict(orient="records")
        fixed_metadata["target_fractions"] = {"train": 0.7, "validation": 0.2, "test": 0.1}
        fixed_metadata["use_validation_for_selection"] = True
        files[fixed_name + ".metadata.json"] = json_bytes(fixed_metadata)
        files[fixed_name + ".balance.csv"] = fixed_balance.to_csv(index=False).encode("utf-8")
        files[fixed_name + ".membership.csv"] = fixed_membership.to_csv(index=False).encode("utf-8")
    return files, balance


def balanced_extension_order(
    source: Sequence[FoldSplit], extra: Sequence[FoldSplit], data: PreparedData,
) -> list[int]:
    """Balance combined complete-anatomy counts and fold sizes without moving D1."""
    coverage = data.annotation_groups(include_lesion=False)

    def score(order: tuple[int, ...]) -> tuple[int, int, int, int]:
        complete: list[int] = []
        sizes: list[int] = []
        for left, index in zip(source, order):
            test = left.test_ids + extra[index].test_ids
            complete.append(sum(coverage[animal] == "complete" for animal in test))
            sizes.append(len(test))
        return (
            max(complete) - min(complete), max(sizes) - min(sizes),
            sum(count * count for count in complete), sum(size * size for size in sizes),
        )

    order = min(permutations(range(len(extra))), key=score)
    return list(order)


def final_fixed_split(
    data: PreparedData, reference: FoldSplit, *, seed: int,
    validation_fraction: float = 0.2, test_fraction: float = 0.1,
    minimum_test_t2w: int = FIXED_TEST_T2W_MINIMUM,
) -> SubjectSplit:
    """Use a test subset of one original fold, then stratify the development pool.

    Cohorts are sampled separately so D1 roles are identical for single-task and
    MouseFull training. Test fractions are rounded to whole animals. D1 retains
    at least five T2w test animals when the reference permits it.
    """
    coverage = data.annotation_groups(include_lesion=False)
    manifest = data.subjects.manifest(data.subjects.subject_ids).drop_duplicates("Subject")
    metadata = manifest.set_index("Subject")
    strata = {
        animal: f"{coverage[animal]}|{metadata.loc[animal, 'Modality']}"
        for animal in data.subjects.subject_ids
    }

    def subset(pool: list[str], count: int, group_seed: int) -> list[str]:
        if count == 0:
            return []
        if count == len(pool):
            return sorted(pool)
        return stratified_validation_ids(pool, strata, count, group_seed, retain_singletons=True)

    training: list[str] = []
    validation: list[str] = []
    test: list[str] = []
    for dataset, scans in manifest.groupby("Dataset", sort=True):
        animals = sorted(scans["Subject"])
        candidates = sorted(set(animals).intersection(reference.test_ids))
        test_count = max(1, round(len(animals) * test_fraction))
        validation_count = max(1, round(len(animals) * validation_fraction))
        if test_count > len(candidates):
            raise ValueError("Reference holdout is too small for the requested fixed test subset")
        t2 = [animal for animal in candidates if metadata.loc[animal, "Modality"] == "T2w"]
        other = sorted(set(candidates) - set(t2))
        if dataset == "D1" and t2:
            desired_t2 = round(test_count * len(t2) / len(candidates))
            desired_t2 = max(desired_t2, minimum_test_t2w)
            t2_count = min(desired_t2, len(t2), test_count - int(bool(other)))
            selected = subset(t2, t2_count, seed)
            selected += subset(other, test_count - t2_count, seed)
        else:
            selected = subset(candidates, test_count, seed)
        pool = sorted(set(animals) - set(selected))
        validation_ids = stratified_validation_ids(pool, strata, validation_count, seed + 1)
        training.extend(sorted(set(pool) - set(validation_ids)))
        validation.extend(validation_ids)
        test.extend(selected)
    result = SubjectSplit(tuple(sorted(training)), tuple(sorted(validation)), tuple(sorted(test)))
    result.validate(data.subjects)
    return result


def replace_references(
    folder: Path, files: dict[str, bytes], created: str, *, obsolete: Sequence[str] = (),
) -> Path:
    """Verify byte-for-byte backups before replacing any reference in this folder.

    Staged files are replaced atomically one at a time. If replacement fails,
    already replaced references are restored from the verified backups.
    """
    folder = folder.resolve(strict=True)
    names = list(files) + list(obsolete)
    if len(names) != len(set(names)):
        raise ValueError("Updated and obsolete reference filenames must be distinct")
    if any(Path(name).name != name for name in names):
        raise ValueError("Reference filenames must stay directly inside the split folder")
    targets = {name: (folder / name).resolve() for name in names}
    if any(target.parent != folder for target in targets.values()):
        raise ValueError("Reference targets must stay inside the split folder")

    stamp = datetime.fromisoformat(created).strftime("%Y%m%d_%H%M%S_%f")
    backup = folder / "split_backups" / f"original_protocol_{stamp}"
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
        for name in files:
            target = targets[name]
            (staging / name).replace(target)
            replaced.append(name)
        for name in obsolete:
            if targets[name].exists():
                targets[name].unlink()
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
    parser.add_argument("--original-splits", type=Path, default=ORIGINAL_SPLITS)
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
    d1_folds = read_selection(args.original_splits, use_validation=False)
    if not isinstance(d1_folds, list):
        raise ValueError("The original reference must contain a complete CV plan")
    validate_folds(d1_folds, d1_ids)
    sham_ids = sorted(set(mouse_full.subjects.subject_ids) - set(d1_ids))
    sham = select_animals(mouse_full, sham_ids)
    extra = balanced_folds(
        sham, k=len(d1_folds), seed=args.seed, validation_fraction=args.validation_fraction,
        use_validation=False,
    )
    extension_order = balanced_extension_order(d1_folds, extra, mouse_full)
    extra = reorder_folds(extra, extension_order)
    full_folds = merge_folds(d1_folds, extra)
    created = datetime.now(ZoneInfo("Europe/Rome")).isoformat()
    files: dict[str, bytes] = {}
    for data, folds, name in ((d1, d1_folds, "D1"), (mouse_full, full_folds, "MouseFull")):
        fixed = final_fixed_split(data, folds[REFERENCE_FOLD], seed=args.seed)
        generated, balance = reference_files(
            data, folds, name, seed=args.seed,
            validation_fraction=args.validation_fraction, created=created,
            fixed_split=fixed, fixed_reference_fold=REFERENCE_FOLD,
        )
        for filename, content in list(generated.items()):
            if filename.endswith(".metadata.json"):
                metadata = json.loads(content)
                metadata["original_source"] = str(args.original_splits.resolve())
                metadata["original_source_sha256"] = hashlib.sha256(args.original_splits.read_bytes()).hexdigest()
                metadata["extension_fold_order"] = extension_order
                if "fixed_split" in filename:
                    metadata["test_subset_of_reference"] = True
                    metadata["minimum_test_t2w"] = FIXED_TEST_T2W_MINIMUM
                elif name == "D1":
                    metadata["stratification"] = []
                    metadata["balance_policy"] = "Exact original D1 assignments; no D1 rebalancing"
                else:
                    metadata["stratification"] = ["D3_anatomy_coverage"]
                    metadata["balance_policy"] = "Exact original D1 assignments; balanced D3 coverage and combined fold sizes"
                generated[filename] = json_bytes(metadata)
        files.update(generated)
        counts = balance.groupby(["Fold", "DatasetRole", "Modality"])["Animals"].sum().unstack(fill_value=0)
        print(f"\n{name}:\n{counts.to_string()}")
        print("Final fixed split:", len(fixed.train), len(fixed.validation), len(fixed.test))
    files["D1_splits_final_original.json"] = args.original_splits.read_bytes()
    if args.apply:
        obsolete = (
            "D1_fixed_split_from_kfold_fold_0.json", "D1_fixed_split_from_kfold_fold_0.metadata.json",
            "MouseFull_fixed_split_from_kfold_fold_0.json", "MouseFull_fixed_split_from_kfold_fold_0.metadata.json",
        )
        backup = replace_references(folder, files, created, obsolete=obsolete)
        print(f"\nUpdated {len(files)} reference/audit files in {folder}\nBackup: {backup}")
    else:
        print("\nPreview only. Use --apply to back up and replace the reference files.")


if __name__ == "__main__":
    main()
