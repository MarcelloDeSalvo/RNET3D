"""Subject grouping and runner integration, using small on-disk NIfTI scans."""

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

from src.evaluation import k_fold as cv
from src.utils.loader import RodentDatasets


@pytest.fixture
def indexed_data(tmp_path):
    """Four normalized animals, including repeated timepoints, plus legacy data."""
    loader = RodentDatasets()
    for number in range(4):
        subject = f'SUBJ_{number:04d}'
        folder = tmp_path / 'D2' / subject
        folder.mkdir(parents=True)
        metadata = dict(dataset='D2', normalized_id=subject, original_id=f'rat-{number}',
                        condition='TBI', modality='T1w', timepoints=['7days', '14days'])
        (folder / 'metadata.json').write_text(json.dumps(metadata), encoding='utf-8')
        for timepoint in metadata['timepoints']:
            scan_folder = folder / timepoint
            scan_folder.mkdir()
            image = nib.Nifti1Image(np.ones((2, 2, 2), dtype=np.int16), np.eye(4))
            for name in ('N4', 'Labels', 'brain_mask'):
                nib.save(image, scan_folder / f'{name}.nii.gz')
    loader.add_normalized_dataset(tmp_path, 'D2')
    legacy_folder = tmp_path / 'FLASH' / 'mouse'
    legacy_folder.mkdir(parents=True)
    for name in ('N4', 'Labels', 'brain_mask'):
        nib.save(image, legacy_folder / f'mouse_{name}.nii.gz')
    loader.add_dataset(tmp_path, 'FLASH')
    return loader, cv.SubjectIndex(loader)


def test_subject_index_groups_timepoints_and_preserves_loader_keys(indexed_data):
    loader, index = indexed_data
    assert len(index.subject_ids) == 5
    assert len(index.scan_keys(index.subject_ids)) == 9
    keys = index.scan_keys(['D2/SUBJ_0000'])
    assert len(keys) == 2
    assert {loader.get_subject(key).subject_id for key in keys} == {'SUBJ_0000'}
    assert index.scan_keys(['FLASH/mouse']) == [('FLASH', 'mouse')]
    assert index.subject_for_scan(keys[0]) == 'D2/SUBJ_0000'
    assert index.manifest(['D2/SUBJ_0000'])['OriginalID'].tolist() == ['rat-0', 'rat-0']

    config = cv.TrainingConfig(model_name='grouped', k=2)
    folds = cv._splits(index.subject_ids, config, None)
    for fold in folds:
        train_groups = {loader.get_subject(key).subject_group for key in index.scan_keys(fold.train_ids)}
        val_groups = {loader.get_subject(key).subject_group for key in index.scan_keys(fold.val_ids)}
        assert train_groups.isdisjoint(val_groups)


def test_explicit_identity_overrides_cannot_separate_timepoints(indexed_data):
    loader, index = indexed_data
    keys = index.scan_keys(['D2/SUBJ_0000'])
    with pytest.raises(ValueError, match='keep every scan'):
        cv.SubjectIndex(loader, subject_overrides={keys[0]: 'animal'})
    overrides = {key: 'animal' for key in keys}
    overrides[('FLASH', 'mouse')] = 'animal'
    merged = cv.SubjectIndex(loader, subject_overrides=overrides)
    assert len(merged.scan_keys(['animal'])) == 3
    assert len(merged.subject_ids) == 4
    with pytest.raises(ValueError, match='Unknown subject'):
        merged.scan_keys(['missing'])
    with pytest.raises(ValueError, match='duplicate'):
        merged.scan_keys(['animal', 'animal'])


@pytest.mark.parametrize('indices', [[-1], [0.1], [True], [2], [np.bool_(False)]])
def test_invalid_split_indices_are_rejected(indices):
    with pytest.raises(ValueError):
        cv._subjects_at_indices(['a', 'b'], indices)


def test_splits_round_trip_and_coverage(tmp_path):
    ids = ['a', 'b', 'c', 'd']
    config = cv.TrainingConfig(model_name='test', k=2)
    folds = cv._splits(ids, config, None)
    payload = [{'train': fold.train_ids, 'validation': fold.val_ids} for fold in folds]
    path = tmp_path / 'splits.json'
    path.write_text(json.dumps(payload))
    assert cv._load_splits(path, list(reversed(ids)), 2) == folds
    payload[1] = payload[0]
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match='once'):
        cv._load_splits(path, ids, 2)


@pytest.mark.parametrize('payload', [[None, {}], [{'train': [[1]], 'validation': ['a']}, {}]])
def test_malformed_split_files_fail_clearly(tmp_path, payload):
    path = tmp_path / 'bad.json'
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        cv._load_splits(path, ['a', 'b'], 2)


def test_cohort_leakage_and_scan_tuples_fail_before_creating_run(tmp_path, indexed_data):
    _, index = indexed_data
    config = cv.TrainingConfig(model_name='test', k=2, output_dir=str(tmp_path / 'runs'))
    noop = lambda *args: None
    with pytest.raises(ValueError, match='subject strings'):
        cv.k_fold_loop(noop, noop, noop, config, [('FLASH', 'mouse'), ('FLASH', 'other')])
    evaluation = cv.EvaluationConfig(test_ids=(index.subject_ids[0],))
    with pytest.raises(ValueError, match='disjoint'):
        cv.k_fold_loop(noop, noop, noop, config, index.subject_ids, evaluation=evaluation, subject_index=index)
    assert not (tmp_path / 'runs').exists()


def test_evaluation_accepts_subject_index_but_rejects_missing_or_unknown_ids(tmp_path):
    result = cv.TrainingResult(tmp_path)
    table = pd.DataFrame({'Subject': ['a'], 'Value': [0.9]}).set_index('Subject')
    cv._record_evaluation(result, 'regions', 'validation', 0, table, ['a'])
    assert result.evaluations['regions'][0]['Subject'].tolist() == ['a']
    for subject in (None, 'wrong'):
        invalid = pd.DataFrame({'Subject': [subject]})
        with pytest.raises(ValueError):
            cv._record_evaluation(result, 'regions', 'validation', 0, invalid, ['a'])


def test_small_keras_run_exports_manifest_and_reusable_splits(tmp_path, indexed_data):
    """Exercise real fit, checkpoint restore, evaluation and saved fold provenance."""
    tf = pytest.importorskip('tensorflow')
    loader, index = indexed_data
    calls = []

    def dataset_factory(subject_ids):
        keys = index.scan_keys(subject_ids)
        calls.append(tuple(subject_ids))
        inputs = np.ones((len(keys), 1), dtype=np.float32)
        targets = np.zeros_like(inputs)
        dataset = tf.data.Dataset.from_tensor_slices((inputs, targets)).batch(1)
        options = tf.data.Options()
        options.threading.private_threadpool_size = 1
        return dataset.with_options(options)

    def model_factory():
        model = tf.keras.Sequential([tf.keras.layers.Input((1,)), tf.keras.layers.Dense(1)])
        model.compile(optimizer='sgd', loss='mse')
        return model

    class Evaluator:
        def __init__(self, model, subject_ids, options):
            self.ids = subject_ids
            assert Path(options['evaluation_dir']).is_dir()
        def evaluate(self):
            return pd.DataFrame({'Subject': self.ids, 'Metric': 'example', 'Value': 1.0})

    config = cv.TrainingConfig(model_name='tiny', k=2, epochs=1, output_dir=str(tmp_path / 'runs'),
                               default_callbacks=False, save_model=False, fit_kwargs={'verbose': 0})
    cv_ids = [subject for subject in index.subject_ids if subject.startswith('D2/')]
    evaluation = cv.EvaluationConfig(external_ids=('FLASH/mouse',))
    result = cv.k_fold_loop(dataset_factory, dataset_factory, model_factory, config, cv_ids,
                            fve_callbacks={'regions': Evaluator}, evaluation=evaluation, subject_index=index)
    assert len(result.histories) == 2
    assert len(calls) == 4
    manifest = pd.read_csv(result.run_dir / 'subjects.csv')
    assert len(manifest) == 9
    assert set(manifest['DatasetRole']) == {'cv', 'external'}
    restored = cv._load_splits(result.run_dir / 'splits.json', cv_ids, 2)
    assert len(restored) == 2
    table = pd.read_csv(result.run_dir / 'summary' / 'all_evaluations.csv')
    assert len(table[table['DatasetRole'] == 'validation']) == 4
    assert len(table[table['DatasetRole'] == 'external']) == 2
    for fold in range(2):
        assert (result.run_dir / f'fold_{fold}' / 'best.weights.h5').is_file()


def test_subject_index_rejects_subject_based_steps_before_run(tmp_path, indexed_data):
    _, index = indexed_data
    config = cv.TrainingConfig(model_name='test', k=2, output_dir=str(tmp_path / 'runs'),
                               infer_steps_from_ids=True)
    noop = lambda *args: None
    with pytest.raises(ValueError, match='cardinality'):
        cv.k_fold_loop(noop, noop, noop, config, index.subject_ids, subject_index=index)
    assert not (tmp_path / 'runs').exists()


def test_empty_generator_is_rejected_before_building_model(tmp_path):
    tf = pytest.importorskip('tensorflow')

    class EmptySequence(tf.keras.utils.Sequence):
        def __len__(self):
            return 0
        def __getitem__(self, index):
            raise IndexError(index)

    def model_factory():
        pytest.fail('An empty generator must be rejected before model construction')

    config = cv.TrainingConfig(model_name='empty', k=2, output_dir=str(tmp_path / 'runs'))
    factory = lambda ids: EmptySequence()
    with pytest.raises(ValueError, match='no complete batches'):
        cv.k_fold_loop(factory, factory, model_factory, config, ['a', 'b'])


@pytest.mark.parametrize('name', ['summary', 'fold_0'])
def test_evaluator_names_cannot_collide_with_runner_folders(name):
    with pytest.raises(ValueError, match='collides'):
        cv._named_evaluators({name: lambda *args: None})


def test_standard_metric_rows_require_subject_coverage_and_include_variant(tmp_path):
    metadata = cv.ExperimentMetadata(model_variant='no_attention', dataset_name='D1+D2', experiment_name='ablation')
    result = cv.TrainingResult(tmp_path / 'run_a', metadata=metadata)
    table = pd.DataFrame({'Subject': ['a'], 'Metric': ['Dice'], 'Value': [0.9], 'Dataset': ['D2']})
    with pytest.raises(ValueError, match='omitted'):
        cv._record_evaluation(result, 'regions', 'validation', 0, table, ['a', 'b'])
    cv._record_evaluation(result, 'regions', 'validation', 0, table, ['a'])
    row = result.evaluations['regions'][0].iloc[0]
    assert row['ModelVariant'] == 'no_attention'
    assert row['Dataset'] == 'D2'
    assert row['DatasetName'] == 'D1+D2'
    assert row['RunID'] == 'run_a'
