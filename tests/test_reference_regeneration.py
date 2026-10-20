"""Reference regeneration preserves role compatibility and recoverable originals."""

import hashlib
import json
from pathlib import Path

import pytest

from test_revision_suite import revision_settings
from scripts.regenerate_reference_splits import reference_files, replace_references
from src.experiments.data import prepare_data
from src.experiments.revision_config import experiment_configuration
from src.experiments.revision_splits import make_suite_splits
from src.experiments.splits import reference_folds, reorder_folds, select_animals


@pytest.mark.parametrize("order", [(0, 1), (1, 0)])
def test_fixed_exports_and_extended_reference_match_joint_folds(revision_settings, tmp_path, order):
    config = experiment_configuration(revision_settings, "mouse_full", output_dir=tmp_path)
    mouse_full = prepare_data(config)
    ids = [animal for animal in mouse_full.subjects.subject_ids if animal.startswith("D1/")]
    d1 = select_animals(mouse_full, ids)
    plan = make_suite_splits(d1, mouse_full, None, k=2, seed=42, validation_fraction=0.2)
    source = reorder_folds(plan.d1, order)
    joint = reorder_folds(plan.mouse_full, order)
    for data, folds, name in ((d1, source, "D1"), (mouse_full, joint, "MouseFull")):
        files, balance = reference_files(
            data, folds, name, seed=42, validation_fraction=0.2, created="2026-10-20T10:00:00+02:00",
            generation_fold_order=order,
        )
        fixed_name = f"{name}_fixed_split_from_kfold_fold_0.json"
        assert json.loads(files[fixed_name]) == folds[0].to_dict()
        assert balance["Animals"].sum() == len(folds) * len(data.subjects.subject_ids)
        if name == "D1":
            path = tmp_path / "D1_splits_final.json"
            path.write_bytes(files[path.name])
            metadata = path.with_suffix(".metadata.json")
            metadata.write_bytes(files[metadata.name])
            assert reference_folds(mouse_full, path) == joint
            resolved = make_suite_splits(
                d1, mouse_full, None, k=2, seed=42, validation_fraction=0.2, folds_file=path,
            )
            assert resolved.mouse_full == joint
            assert resolved.d1 == source


def test_backups_keep_original_bytes_and_hashes(tmp_path):
    old = {"D1_splits_final.json": b"original\r\n", "D1_splits_final.metadata.json": b"old metadata"}
    for name, content in old.items():
        (tmp_path / name).write_bytes(content)
    updated = {"D1_splits_final.json": b"new", "D1_splits_final.metadata.json": b"{}"}
    backup = replace_references(tmp_path, updated, "2026-10-20T10:00:00+02:00")
    manifest = json.loads((backup / "backup_manifest.json").read_bytes())
    for name, content in old.items():
        assert (backup / name).read_bytes() == content
        assert manifest["sha256"][name] == hashlib.sha256(content).hexdigest()
    assert (tmp_path / "D1_splits_final.json").read_bytes() == b"new"
    metadata = json.loads((tmp_path / "D1_splits_final.metadata.json").read_bytes())
    assert metadata["backup_folder"] == str(backup)


def test_failed_replacement_restores_originals(tmp_path, monkeypatch):
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_bytes(b"old first")
    second.write_bytes(b"old second")
    replace = Path.replace

    def fail_second(source, target):
        if source.name == "second.json":
            raise OSError("Simulated replacement failure")
        return replace(source, target)

    monkeypatch.setattr(Path, "replace", fail_second)
    with pytest.raises(OSError, match="Simulated"):
        replace_references(
            tmp_path, {"first.json": b"new first", "second.json": b"new second"},
            "2026-10-20T10:00:00+02:00",
        )
    assert first.read_bytes() == b"old first"
    assert second.read_bytes() == b"old second"
