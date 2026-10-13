"""Check saved multi-patch coverage, pairing and independence from fitting seeds."""

import json
from dataclasses import replace

import nibabel as nib
import numpy as np
import pytest

from test_experiments import experiment
from src.experiments.config import EvaluationPatchConfig, ExperimentConfig
from src.experiments.data import prepare_data


@pytest.fixture
def patch_data(experiment):
    """Use asymmetric, larger scans so distinct crops and LR flips are observable."""
    shape = (32, 32, 32)
    ramp = np.indices(shape)[0].astype(np.float32)
    regions = np.zeros(shape, dtype=np.int16)
    regions[3:8, 5:20, 5:20] = 1
    brain = (ramp > 4).astype(np.int16)
    for path in experiment.data.root.rglob('*.nii.gz'):
        image = nib.load(path)
        arrays = {'N4.nii.gz': ramp, 'Labels.nii.gz': regions, 'brain_mask.nii.gz': brain}
        nib.save(nib.Nifti1Image(arrays[path.name], image.affine, image.header), path)

    settings = EvaluationPatchConfig(patches_per_scan=3, seed=17)
    return prepare_data(replace(experiment, evaluation_patches=settings))


def test_multiple_patches_are_stable_across_epochs_and_retain_tail(patch_data):
    generator = patch_data.make_generator(patch_data.subjects.subject_ids, training=False, batch_size=8)
    assert len(generator.ids) == 15  # All five longitudinal scans have three crops.
    assert len(generator) == 2
    assert generator[1][0].shape[0] == 7
    before = generator[0]
    generator.on_epoch_end()
    after = generator[0]
    np.testing.assert_array_equal(before[0], after[0])
    for first, second in zip(before[1], after[1]):
        np.testing.assert_array_equal(first, second)
    samples = patch_data.patch_collection().samples
    assert len({sample.center for sample in samples[:3]}) > 1
    assert not generator.shuffle


def test_saved_coordinates_survive_new_objects_and_training_seeds(patch_data, tmp_path):
    path = tmp_path / 'evaluation_patches.json'
    patch_data.save_evaluation_patches(path)
    training = replace(patch_data.config.training, seed=99, data_multiplier=7)
    preprocessing = replace(patch_data.config.preprocessing, augment=True, random_crop=False)
    other = prepare_data(replace(patch_data.config, training=training, preprocessing=preprocessing))
    np.random.seed(1009)
    other.save_evaluation_patches(path, resume=True)
    assert patch_data.patch_collection().samples == other.patch_collection().samples
    ids = patch_data.subjects.subject_ids[:1]
    np.testing.assert_array_equal(
        patch_data.make_generator(ids, training=False)[0][0],
        other.make_generator(ids, training=False)[0][0],
    )


def test_lr_pairs_reflect_image_roi_and_brain_together(patch_data):
    settings = replace(patch_data.config.evaluation_patches, left_right_pairs=True)
    data = prepare_data(replace(patch_data.config, evaluation_patches=settings))
    generator = data.make_generator(data.subjects.subject_ids[:1], training=False, batch_size=2)
    images, targets = generator[0]
    np.testing.assert_array_equal(images[1], np.flip(images[0], axis=0))
    for target in targets:
        np.testing.assert_array_equal(target[1], np.flip(target[0], axis=0))
    assert len(generator.ids) == 6
    assert generator.sample_preprocessors[0].patch.center == generator.sample_preprocessors[1].patch.center


def test_manifest_rejects_changes_and_missing_resume_samples(patch_data, tmp_path):
    path = tmp_path / 'evaluation_patches.json'
    with pytest.raises(FileNotFoundError, match='saved evaluation patches'):
        patch_data.save_evaluation_patches(path, resume=True)
    patch_data.save_evaluation_patches(path)
    changed = replace(patch_data.config.evaluation_patches, seed=18)
    other = prepare_data(replace(patch_data.config, evaluation_patches=changed))
    with pytest.raises(ValueError, match='configuration or inputs differ'):
        other.save_evaluation_patches(path, resume=True)
    payload = json.loads(path.read_text())
    payload['samples'][0]['center'][0] += 1
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match='modified'):
        patch_data.save_evaluation_patches(path, resume=True)


def test_older_configs_keep_their_single_center_patch(experiment):
    values = experiment.to_dict()
    values.pop('evaluation_patches')
    restored = ExperimentConfig.from_dict(values)
    assert restored.evaluation_patches == EvaluationPatchConfig(patches_per_scan=1, sampling='center')


@pytest.mark.parametrize('settings', [
    EvaluationPatchConfig(patches_per_scan=0), EvaluationPatchConfig(patches_per_scan=True),
    EvaluationPatchConfig(seed=-1), EvaluationPatchConfig(sampling='random'),
    EvaluationPatchConfig(patches_per_scan=3, sampling='center'),
])
def test_invalid_patch_settings_are_rejected(settings):
    with pytest.raises(ValueError):
        settings.validate()
