"""Verify physical LR augmentation, relative label semantics and shared geometry."""

from dataclasses import replace
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import SimpleITK as sitk

from test_experiments import experiment
from src.experiments.data import make_preprocessor
from src.preprocessing import preprocessor as p


REFERENCE = Path(__file__).resolve().parents[1] / "example/RARE/TBI_fm_19_49/Anat/TBI_fm_19_49_N4.nii.gz"


def oriented_phantom(codes):
    """Place a lesion, ipsilateral CC and contralateral CC on opposite sides."""
    directions = {
        "R": (1, 0, 0), "L": (-1, 0, 0), "A": (0, 1, 0),
        "P": (0, -1, 0), "S": (0, 0, 1), "I": (0, 0, -1),
    }
    shape = (7, 9, 11)
    affine = np.eye(4)
    affine[:3, :3] = np.column_stack([directions[code] for code in codes]) * 0.2
    center = (np.asarray(shape) - 1) / 2
    affine[:3, 3] = -affine[:3, :3] @ center
    labels = np.zeros(shape, dtype=np.int16)
    axis = next(index for index, code in enumerate(codes) if code in ("L", "R"))
    for label, coordinate in ((1, 1), (12, 2), (2, shape[axis] - 2)):
        position = center.astype(int)
        position[axis] = coordinate
        labels[tuple(position)] = label
    image = nib.Nifti1Image(labels.astype(np.float32), affine)
    roi = nib.Nifti1Image(labels, affine)
    brain = nib.Nifti1Image((labels > 0).astype(np.int16), affine)
    return image, roi, brain, axis


@pytest.mark.parametrize("codes", [("R", "I", "A"), ("I", "P", "L"), ("A", "S", "R"), ("L", "P", "I")])
def test_lr_flip_tracks_the_affine_and_preserves_relative_label_ids(codes):
    image, roi, brain, expected_axis = oriented_phantom(codes)
    flip = p.Flip(anatomical_axis="LR", probability=1)
    assert flip.sample_parameters(image)["axis"] == expected_axis
    transformed, masks = flip.apply(image, [roi, brain])
    expected = np.flip(roi.get_fdata(), axis=expected_axis)
    np.testing.assert_array_equal(transformed.get_fdata(), expected)
    np.testing.assert_array_equal(masks[0].get_fdata(), expected)
    np.testing.assert_array_equal(masks[1].get_fdata(), expected > 0)
    np.testing.assert_array_equal(transformed.affine, image.affine)
    for mask in masks:
        np.testing.assert_array_equal(mask.affine, transformed.affine)

    def world_x(labels, label):
        position = np.argwhere(labels == label)[0]
        return nib.affines.apply_affine(image.affine, position)[0]

    lesion = world_x(expected, 1)
    ipsilateral = world_x(expected, 12)
    contralateral = world_x(expected, 2)
    assert np.sign(lesion) == np.sign(ipsilateral)
    assert np.sign(lesion) != np.sign(contralateral)
    assert np.sign(lesion) == -np.sign(world_x(roi.get_fdata(), 1))
    assert set(np.unique(masks[0].get_fdata())) == {0, 1, 2, 12}


def test_shared_reorientation_resampling_affine_crop_and_flip(monkeypatch):
    """Use nearest MRI interpolation on this label phantom to compare markers exactly."""
    image, roi, brain, _ = oriented_phantom(("I", "P", "L"))
    reference = nib.load(REFERENCE)
    affine = p.RandomAffine(
        rotation_range=(-5, 5), scale_range=(0.95, 1.05),
        probability=1, interpolation=sitk.sitkNearestNeighbor,
    )
    original_sample = affine.sample_parameters
    samples = []

    def sample(group_image):
        parameters = original_sample(group_image)
        samples.append(parameters)
        return parameters

    monkeypatch.setattr(affine, "sample_parameters", sample)
    pipeline = p.Preprocessor([
        p.Reorient(reference), p.Resample((0.2, 0.2, 0.2), interpolation=0, orthogonal=True),
        affine, p.RandomCropping((7, 7, 7), mode="center", std=None),
        p.Flip(anatomical_axis="LR", probability=1),
    ])
    np.random.seed(21)
    transformed, masks = pipeline.preprocess(image, [roi, brain])
    assert len(samples) == 1
    assert nib.aff2axcodes(transformed.affine) == ("R", "I", "A")
    assert np.any(masks[0].get_fdata() > 0)
    np.testing.assert_array_equal(transformed.get_fdata(), masks[0].get_fdata())
    np.testing.assert_array_equal(masks[0].get_fdata() > 0, masks[1].get_fdata())
    for mask in masks:
        assert mask.shape == transformed.shape == (7, 7, 7)
        np.testing.assert_array_equal(mask.affine, transformed.affine)
    assert set(np.unique(masks[0].get_fdata())) <= {0, 1, 2, 12}


def test_training_flip_is_opt_in_and_never_used_for_evaluation(experiment):
    assert experiment.preprocessing.flip_probability == 0
    settings = replace(experiment.preprocessing, reference=REFERENCE, augment=True, flip_probability=0.5)
    config = replace(experiment, preprocessing=settings)
    training = make_preprocessor(config, stage="training")
    flips = [step for step in training.steps if isinstance(step, p.Flip)]
    assert len(flips) == 1
    assert flips[0].anatomical_axis == "LR" and flips[0].axis_list is None
    for stage in ("validation", "inference"):
        pipeline = make_preprocessor(config, stage=stage)
        assert not any(isinstance(step, p.Flip) for step in pipeline.steps)
    disabled = replace(config, preprocessing=replace(settings, flip_probability=0.0))
    pipeline = make_preprocessor(disabled, stage="training")
    assert not any(isinstance(step, p.Flip) for step in pipeline.steps)


@pytest.mark.parametrize("kwargs", [{}, {"axis_list": [0], "anatomical_axis": "LR"}, {"anatomical_axis": "bad"}])
def test_invalid_anatomical_axis_selection_is_explicit(kwargs):
    with pytest.raises(ValueError):
        p.Flip(**kwargs)


def test_notebook_reference_and_cohort_flip_policy():
    root = Path(__file__).resolve().parents[1]
    probabilities = {
        "01_tbi_mouse_training.ipynb": 0.5, "02_tbi_mouse_kfold.ipynb": 0.5,
        "03_rat_finetuning.ipynb": 0.5, "04_rat_kfold.ipynb": 0.5,
        "05_mouse_full_training.ipynb": 0.0, "06_mouse_full_kfold.ipynb": 0.0,
        "07_ablation.ipynb": 0.5, "07f_ablation.ipynb": 0.0, "08_unet_comparison.ipynb": 0.5,
    }
    for filename, probability in probabilities.items():
        cells = json.loads((root / "notebook" / filename).read_text(encoding="utf-8"))["cells"]
        sources = ["".join(cell["source"]) for cell in cells if cell["cell_type"] == "code"]
        setup = next(source for source in sources if "REPO_ROOT: Path = next(" in source)
        assert '"TBI_fm_19_49_N4.nii.gz"' in setup
        config = next(source for source in sources if "preprocessing = PreprocessingConfig(" in source)
        assert f"flip_probability={probability}" in config
        assert 'REFERENCE: Path = DATA_ROOT' not in config
        checkpoint = next(source for source in sources if "checkpoint_settings = best_checkpoint_settings(" in source)
        assert "reference=REFERENCE" in checkpoint
    assert nib.aff2axcodes(nib.load(REFERENCE).affine) == ("R", "I", "A")
