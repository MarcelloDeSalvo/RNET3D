"""Preflight fitting/selection ancestry and exact inventory identities before jobs."""

import json
from dataclasses import replace

import pandas as pd
import pytest

from test_experiments import experiment
from src.experiments.config import DatasetSpec, InitializationConfig, ExperimentConfig
from src.experiments.data import SubjectSplit, prepare_data, validate_inventory
from src.experiments.initialization import pretraining_audit
from src.experiments import training
from src.evaluation import k_fold
from src.evaluation.k_fold import FoldSplit


def source_run(data, folder, train_ids, validation_ids, *, parent=None):
    """Write source provenance only; no model construction or training needed."""
    fold = folder / 'fold_0'
    fold.mkdir(parents=True)
    checkpoint = fold / 'best.weights.h5'
    checkpoint.write_bytes(b'weights not loaded by the provenance audit')
    config = data.config
    if parent is not None:
        config = replace(config, initialization=parent)
    (folder / 'experiment.json').write_text(json.dumps(config.to_dict(), default=str))
    selection = {'train': train_ids, 'validation': validation_ids}
    (fold / 'split.json').write_text(json.dumps(selection))
    known = [animal for animal in train_ids + validation_ids if animal in data.subjects.subject_ids]
    manifest = data.subjects.manifest(known)
    for animal in set(train_ids + validation_ids) - set(known):
        external = {'Subject': animal, 'Dataset': 'independent_source', 'OriginalID': animal,
                    'ImagePath': str(folder / (animal + '.nii.gz')),
                    'Subgroup': '', 'Group': '', 'Scan': animal}
        manifest = pd.concat([manifest, pd.DataFrame([external])], ignore_index=True)
    manifest.to_csv(folder / 'subjects.csv', index=False)
    return InitializationConfig(mode='weights', checkpoint=checkpoint)


@pytest.mark.parametrize('source_role', ['train', 'validation'])
def test_source_fitting_and_selection_animals_cannot_be_target_holdouts(experiment, tmp_path, source_role):
    data = prepare_data(experiment)
    roles = {'train': ['independent'], 'validation': ['other-independent']}
    roles[source_role] = ['D1/SUBJ_0002']
    settings = source_run(data, tmp_path / 'source', roles['train'], roles['validation'])
    with pytest.raises(ValueError, match='Pretraining leakage.*D1/SUBJ_0002'):
        pretraining_audit(data, settings, ['D1/SUBJ_0002'])


def test_original_identity_catches_renumbering_without_guessing_cross_cohort_names(experiment, tmp_path):
    data = prepare_data(experiment)
    settings = source_run(data, tmp_path / 'source', ['independent'], ['other-independent'])
    path = tmp_path / 'source' / 'subjects.csv'
    manifest = pd.read_csv(path).fillna('')
    manifest.loc[0, ['Dataset', 'OriginalID']] = ['D1', 'animal-2']
    manifest.to_csv(path, index=False)
    with pytest.raises(ValueError, match='Pretraining leakage'):
        pretraining_audit(data, settings, ['D1/SUBJ_0002'])
    manifest.loc[0, 'Dataset'] = 'D3'
    manifest.to_csv(path, index=False)
    audit = pretraining_audit(data, settings, ['D1/SUBJ_0002'])
    assert audit['sha256']


def test_ancestor_initialization_cannot_hide_target_animal_leakage(experiment, tmp_path):
    data = prepare_data(experiment)
    ancestor = source_run(data, tmp_path / 'ancestor', ['D1/SUBJ_0002'], ['independent'])
    settings = source_run(data, tmp_path / 'source', ['D1/SUBJ_0000'], ['other-independent'], parent=ancestor)
    with pytest.raises(ValueError, match='Pretraining leakage'):
        pretraining_audit(data, settings, ['D1/SUBJ_0002'])


def test_reused_names_are_qualified_by_subgroup_including_older_manifests(experiment, tmp_path):
    path = experiment.data.root / 'D1' / 'SUBJ_0002' / 'metadata.json'
    metadata = json.loads(path.read_text())
    metadata.update(original_id='reused-name', subgroup='independent-group')
    path.write_text(json.dumps(metadata))
    data = prepare_data(experiment)
    settings = source_run(data, tmp_path / 'source', ['independent'], ['other-independent'])
    manifest_file = tmp_path / 'source' / 'subjects.csv'
    manifest = pd.read_csv(manifest_file, dtype=str).fillna('')
    manifest.loc[0, ['Dataset', 'OriginalID', 'Subgroup', 'Group', 'Scan']] = [
        'D1', 'reused-name', 'source-group', 'D1_T1w', 'renumbered',
    ]
    manifest.to_csv(manifest_file, index=False)
    pretraining_audit(data, settings, ['D1/SUBJ_0002'])

    manifest.drop(columns='Subgroup').to_csv(manifest_file, index=False)
    contract_file = tmp_path / 'source' / 'data_contract.json'
    contract = [{'scan': ['D1_T1w', 'renumbered'], 'metadata': {'subgroup': 'source-group'}}]
    contract_file.write_text(json.dumps(contract))
    pretraining_audit(data, settings, ['D1/SUBJ_0002'])
    contract[0]['metadata']['subgroup'] = 'independent-group'
    contract_file.write_text(json.dumps(contract))
    with pytest.raises(ValueError, match='Pretraining leakage'):
        pretraining_audit(data, settings, ['D1/SUBJ_0002'])


def test_missing_provenance_is_rejected_before_training(experiment, tmp_path, monkeypatch):
    checkpoint = tmp_path / 'standalone.weights.h5'
    checkpoint.write_bytes(b'unknown provenance')
    config = replace(experiment, initialization=InitializationConfig(mode='weights', checkpoint=checkpoint))
    data = prepare_data(config)
    split = SubjectSplit(train=('D1/SUBJ_0000', 'D1/SUBJ_0001'),
                         validation=('D1/SUBJ_0002',), test=('D1/SUBJ_0003',))
    monkeypatch.setattr(training, 'build_model', lambda *_: pytest.fail('Must reject before building'))
    with pytest.raises(ValueError, match='saved source run'):
        training.run_training(data, split)
    assert not (tmp_path / 'runs').exists()


def test_legacy_source_requires_confirmed_identity_links_for_normalized_targets(experiment, tmp_path):
    data = prepare_data(experiment)
    settings = source_run(data, tmp_path / 'source', ['independent'], ['other-independent'])
    manifest_file = tmp_path / 'source' / 'subjects.csv'
    manifest = pd.read_csv(manifest_file, dtype=str).fillna('')
    manifest['Dataset'] = 'FLASH'
    manifest.to_csv(manifest_file, index=False)
    legacy = DatasetSpec(layout='legacy', modality='FLASH', available_labels=(0, 1, 2))
    source_config = replace(experiment, data=replace(experiment.data, datasets=(legacy,)))
    path = tmp_path / 'source' / 'experiment.json'
    path.write_text(json.dumps(source_config.to_dict()))
    with pytest.raises(ValueError, match='explicit animal_ids'):
        pretraining_audit(data, settings, ['D1/SUBJ_0002'])
    identities = {'FLASH/original-a': 'independent', 'FLASH/original-b': 'other-independent'}
    source_config = replace(source_config, data=replace(source_config.data, animal_ids=identities))
    path.write_text(json.dumps(source_config.to_dict()))
    pretraining_audit(data, settings, ['D1/SUBJ_0002'])


def test_all_cv_sources_are_checked_before_any_fold_starts(experiment, tmp_path, monkeypatch):
    config = replace(experiment, training=replace(experiment.training, k=2))
    data = prepare_data(config)
    folds = [FoldSplit(index=0, train_ids=['D1/SUBJ_0000'], val_ids=['D1/SUBJ_0001'],
                       test_ids=['D1/SUBJ_0002', 'D1/SUBJ_0003']),
             FoldSplit(index=1, train_ids=['D1/SUBJ_0002'], val_ids=['D1/SUBJ_0003'],
                       test_ids=['D1/SUBJ_0000', 'D1/SUBJ_0001'])]
    safe = source_run(data, tmp_path / 'safe', ['D1/SUBJ_0000'], ['independent'])
    data.config = replace(config, initialization=safe)
    monkeypatch.setattr(k_fold, 'train_fold', lambda **_: pytest.fail('Must reject all folds before fitting'))
    with pytest.raises(ValueError, match='Pretraining leakage'):
        training.run_cross_validation(data, folds=folds)
    assert not (tmp_path / 'runs').exists()

    other = source_run(replace(data, config=config), tmp_path / 'other', ['D1/SUBJ_0002'], ['independent'])
    data.config = replace(config, initialization_by_fold={0: safe, 1: other})
    received = []
    monkeypatch.setattr(training, 'build_model', lambda cfg: received.append(cfg.initialization.checkpoint))
    monkeypatch.setattr(k_fold, 'train_fold', lambda **options: options['model_factory']())
    result = training.run_cross_validation(data, folds=folds)
    assert received == [safe.checkpoint, other.checkpoint]
    audits = json.loads((result.run_dir / 'pretraining_audit.json').read_text())
    assert set(audits) == {'0', '1'}
    assert ExperimentConfig.from_dict(data.config.to_dict()).initialization_by_fold == {0: safe, 1: other}


def test_inventory_crosschecks_exact_original_ids_and_longitudinal_scans(experiment):
    data = prepare_data(experiment)
    manifest = data.subjects.manifest(data.subjects.subject_ids)
    inventory = manifest.rename(columns={'Dataset': 'dataset', 'SubjectID': 'anonymized_subject_id',
                                         'Timepoint': 'timepoint_or_context', 'OriginalID': 'original_subject_id'})
    path = experiment.data.root / 'mapping.xlsx'
    inventory.to_excel(path, sheet_name='Mapping', index=False)
    assert len(validate_inventory(data)) == 5
    inventory.loc[0, 'original_subject_id'] = 'incorrect animal'
    inventory.to_excel(path, sheet_name='Mapping', index=False)
    with pytest.raises(ValueError, match='identity differs'):
        prepare_data(experiment)
