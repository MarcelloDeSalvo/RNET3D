"""Trace sparse source IDs through normalized metadata, targets and final export."""

import json
from dataclasses import replace

import nibabel as nib
import numpy as np
import pytest

from test_experiments import experiment
from src.experiments.config import LabelConfig, PostprocessingConfig
from src.experiments.data import prepare_data
from src.evaluation.postprocessing import FullVolumePostprocessor, OutputLabels
from src.preprocessing.generator import MultiTaskGenerator
from src.preprocessing.preprocessor import MapLabels, Preprocessor


@pytest.mark.parametrize('partial', [False, True])
def test_mouse_full_ventricles_keep_their_identity_through_the_pipeline(experiment, partial):
    mapping = {0: 0, 1: 1, 2: 2, 3: 3, 5: 4, 6: 5, 12: 6, 13: 7, 15: 8, 16: 9, 21: 10}
    names = ('Background', 'Lesion', 'Contra CC', 'Contra ventricle', 'Contra hippocampus',
             'Contra cortex', 'Ipsi CC', 'Ipsi ventricle', 'Ipsi hippocampus', 'Ipsi cortex', 'Third ventricle')
    labels = LabelConfig(mapping=mapping, names=names)
    subject_folder = experiment.data.root / 'D1' / 'SUBJ_0000'
    metadata_path = subject_folder / 'metadata.json'
    metadata = json.loads(metadata_path.read_text())
    metadata['annotated_regions'] = ['Les', 'cl-V', 'il-V', 'tV']
    if not partial:
        metadata['annotated_regions'].extend(['cl-CC', 'il-CC', 'cl-Hipp', 'il-Hipp', 'cl-Ctx', 'il-Ctx'])
    # The dataset's descriptive network_label order is not the model channel order.
    metadata['label_mapping'] = [
        {'network_label': 2, 'name': 'Contra Ventricle', 'mask_value': 3},
        {'network_label': 3, 'name': 'Ipsi Ventricle', 'mask_value': 13},
    ]
    metadata_path.write_text(json.dumps(metadata))
    source = nib.load(subject_folder / 'Labels.nii.gz')
    original = np.zeros(source.shape, dtype=np.int16)
    original[2:4, 4:8, 4:8] = 3
    original[11:14, 4:8, 4:8] = 13
    nib.save(nib.Nifti1Image(original, source.affine), subject_folder / 'Labels.nii.gz')
    config = replace(experiment, labels=labels)
    data = prepare_data(config)
    key = ('D1_T1w', 'SUBJ_0000')
    scan = data.loader.get_subject(key)
    assert 3 not in scan.unavailable_labels and 13 not in scan.unavailable_labels

    settings = {'dataset_path': str(config.data.root), 'labels': list(mapping), 'label_mapping': mapping,
                'input_shape': config.preprocessing.patch_size, 'in_channels': 1, 'num_classes': 11, 'batch_size': 1}
    pipeline = Preprocessor([MapLabels(mapping=mapping)])
    generator = MultiTaskGenerator(
        [key], data.loader, settings, shuffle=False, preprocessor=pipeline,
        drop_last=False, partial_labels=True,
    )
    _, targets = generator[0]
    target = targets[0][0]
    np.testing.assert_array_equal(target[..., 3] == 1, original == 3)
    np.testing.assert_array_equal(target[..., 7] == 1, original == 13)
    if partial:
        assert np.all(target[..., 2] == -1)  # CC unavailable, ventricles unaffected.

    output = OutputLabels.resolve(labels, split_ventricles=False, original_label_ids=True)
    assert output.prediction_mapping[3] == 3
    assert output.prediction_mapping[7] == 13
    channels = np.argmax(target, axis=-1)
    processor = FullVolumePostprocessor(labels, PostprocessingConfig(), original_label_ids=True)
    processed = processor.process(channels, None)
    np.testing.assert_array_equal(processed.regions, original)
