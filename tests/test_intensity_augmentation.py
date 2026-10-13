"""Check native acquisition degradation, explicit units and spatial alignment."""

import ast
import json
from dataclasses import replace
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from test_experiments import experiment
from src.experiments.config import ExperimentConfig
from src.experiments.data import make_preprocessor, prepare_data
from src.preprocessing import preprocessor as p


@pytest.mark.parametrize("grid", ["native", "resampled"])
def test_intensity_steps_use_the_declared_grid_and_leave_masks_aligned(experiment, monkeypatch, grid):
    """Observe the actual grids, including anisotropic MRI and both label masks."""
    settings = replace(
        experiment.preprocessing, augment=True, augmentation_probability=1.0,
        intensity_augmentation_grid=grid, rotation_range=(0, 0), scale_range=(1, 1),
        random_crop=False, blur_sigma_range=(0.1, 0.1), noise_std_range=(0.1, 0.1),
    )
    config = replace(experiment, preprocessing=settings)
    data = prepare_data(config)
    scan = data.loader.get_subject(("D1_T1w", "SUBJ_0000"))
    affine = np.diag([0.2, 0.2, 0.3, 1.0])
    image = nib.Nifti1Image(scan.image.get_fdata(), affine)
    labels = nib.Nifti1Image(scan.labels.get_fdata(), affine)
    brain = nib.Nifti1Image(scan.brain_mask.get_fdata(), affine)
    observed = []

    for step_class in (p.GaussianBlur, p.Noise):
        original = step_class.execute

        def observe(self, image, is_mask=False, _original=original, **parameters):
            if not is_mask:
                observed.append((self.name, image.shape, nib.affines.voxel_sizes(image.affine)))
            return _original(self, image, is_mask=is_mask, **parameters)

        monkeypatch.setattr(step_class, "execute", observe)

    pipeline = make_preprocessor(config, stage="training")
    names = [step.name for step in pipeline.steps]
    if grid == "native":
        assert names.index("GaussianBlur") < names.index("Noise") < names.index("Resample") < names.index("RandomAffine")
    else:
        assert names.index("Resample") < names.index("RandomAffine") < names.index("GaussianBlur") < names.index("Noise")
    output, masks = pipeline.preprocess(image, [labels, brain])

    expected_spacing = [0.2, 0.2, 0.3] if grid == "native" else [0.1, 0.1, 0.1]
    assert [record[0] for record in observed] == ["GaussianBlur", "Noise"]
    for _, shape, spacing in observed:
        np.testing.assert_allclose(spacing, expected_spacing)
        assert shape == ((16, 16, 16) if grid == "native" else (32, 32, 48))
    assert output.shape == (16, 16, 16)
    for mask in masks:
        assert mask.shape == output.shape
        np.testing.assert_array_equal(mask.affine, output.affine)
    assert set(np.unique(masks[0].get_fdata())) <= {0, 1, 2}
    assert set(np.unique(masks[1].get_fdata())) <= {0, 1}


def test_native_noise_is_interpolated_before_normalization(experiment, monkeypatch):
    """Linear grid conversion filters an injected native high-frequency noise field."""
    def alternating_noise(mean, std, shape):
        positions = np.indices(shape)[0]
        return mean + std * (-1.0) ** positions

    monkeypatch.setattr(np.random, "normal", alternating_noise)
    observed = []
    original = p.Normalize.execute

    def capture(self, image, is_mask=False, **parameters):
        if not is_mask:
            observed.append(image.get_fdata().copy())
        return original(self, image, is_mask=is_mask, **parameters)

    monkeypatch.setattr(p.Normalize, "execute", capture)
    source = nib.Nifti1Image(np.zeros((16, 16, 16)), np.diag([0.2, 0.2, 0.2, 1.0]))
    settings = replace(
        experiment.preprocessing, augment=True, augmentation_probability=1.0,
        rotation_range=(0, 0), scale_range=(1, 1), blur_sigma_range=(0, 0),
        noise_std_range=(1, 1), random_crop=False,
    )
    for grid in ("native", "resampled"):
        configured = replace(settings, intensity_augmentation_grid=grid)
        config = replace(experiment, preprocessing=configured)
        pipeline = make_preprocessor(config, stage="training")
        pipeline.preprocess(source)

    native, resampled = observed
    assert native.shape == resampled.shape == (32, 32, 32)
    assert np.std(native) < 0.75 * np.std(resampled)
    assert not np.allclose(native, resampled)


@pytest.mark.parametrize("units", ["intensity", "image_std"])
def test_noise_unit_scale_is_explicit_and_intensity_scaling_behaves_as_declared(units):
    values = np.arange(64, dtype=float).reshape(4, 4, 4)
    source = nib.Nifti1Image(values, np.eye(4))
    scaled = nib.Nifti1Image(4 * values + 10, np.eye(4))
    noise = p.Noise((0.1, 0.1), std_units=units)

    np.random.seed(18)
    first = noise.execute(source).get_fdata() - values
    np.random.seed(18)
    second = noise.execute(scaled).get_fdata() - scaled.get_fdata()
    factor = 4 if units == "image_std" else 1
    np.testing.assert_allclose(second, factor * first, atol=1e-12)
    assert noise.execute(source, is_mask=True) is source
    np.testing.assert_array_equal(source.get_fdata(), values)


def test_relative_noise_does_not_invent_variation_in_constant_images():
    source = nib.Nifti1Image(np.ones((4, 4, 4)), np.eye(4))
    result = p.Noise((0.2, 0.2), std_units="image_std").execute(source)
    np.testing.assert_array_equal(result.get_fdata(), source.get_fdata())


def test_millimetre_blur_converts_sigma_for_anisotropic_spacing():
    values = np.zeros((17, 17, 17))
    values[8, 8, 8] = 1
    source = nib.Nifti1Image(values, np.diag([0.2, 0.4, 0.8, 1]))
    blur = p.GaussianBlur((0.4, 0.4), sigma_units="mm")
    actual = blur.execute(source)
    expected = gaussian_filter(values, sigma=(2, 1, 0.5))
    np.testing.assert_allclose(actual.get_fdata(), expected)
    np.testing.assert_array_equal(actual.affine, source.affine)
    assert blur.execute(source, is_mask=True) is source
    np.testing.assert_array_equal(source.get_fdata(), values)


@pytest.mark.parametrize("stage,augment", [("validation", True), ("inference", True), ("training", False)])
def test_intensity_augmentation_is_absent_outside_augmented_training(experiment, stage, augment):
    settings = replace(experiment.preprocessing, augment=augment)
    pipeline = make_preprocessor(replace(experiment, preprocessing=settings), stage=stage)
    assert not any(isinstance(step, (p.GaussianBlur, p.Noise, p.RandomAffine)) for step in pipeline.steps)


@pytest.mark.parametrize("field,value", [
    ("intensity_augmentation_grid", "unknown"), ("blur_sigma_units", "unknown"),
    ("noise_std_units", "unknown"), ("blur_sigma_range", (-0.1, 1)),
    ("noise_std_range", (1, 0)), ("noise_std_range", (0, float("nan"))),
    ("blur_sigma_range", (0,)),
])
def test_invalid_augmentation_units_and_bounds_fail_before_loading(experiment, field, value):
    settings = replace(experiment.preprocessing, **{field: value})
    config = replace(experiment, preprocessing=settings)
    with pytest.raises(ValueError):
        prepare_data(config)
    with pytest.raises(ValueError):
        make_preprocessor(config, stage="training")


def test_new_configuration_roundtrips_units_and_older_runs_keep_resampled_order(experiment, tmp_path):
    from src.experiments.training import save_provenance, validate_resume

    payload = json.loads(json.dumps(experiment.to_dict()))
    assert ExperimentConfig.from_dict(payload) == experiment
    fields = ("intensity_augmentation_grid", "blur_sigma_range", "blur_sigma_units", "noise_std_range", "noise_std_units")
    for field in fields:
        payload["preprocessing"].pop(field)
    restored = ExperimentConfig.from_dict(payload)
    assert restored.preprocessing.intensity_augmentation_grid == "resampled"
    restored.validate()
    data = prepare_data(restored)
    save_provenance(restored, data, tmp_path)
    (tmp_path / "experiment.json").write_text(json.dumps(payload), encoding="utf-8")
    validate_resume(data, tmp_path)
    with pytest.raises(ValueError, match="configuration differs"):
        validate_resume(prepare_data(experiment), tmp_path)


@pytest.mark.parametrize("filename", [
    "01_tbi_mouse_training.ipynb", "02_tbi_mouse_kfold.ipynb", "03_rat_finetuning.ipynb",
    "04_rat_kfold.ipynb", "05_mouse_full_training.ipynb", "06_mouse_full_kfold.ipynb",
    "07_ablation.ipynb", "07f_ablation.ipynb", "08_unet_comparison.ipynb",
])
def test_training_notebooks_show_native_grid_ranges_and_units(filename):
    path = Path(__file__).resolve().parents[1] / "notebook" / filename
    notebook = json.loads(path.read_text(encoding="utf-8"))
    source = "\n".join("".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code")
    ast.parse(source)
    assert 'intensity_augmentation_grid="native"' in source
    assert 'blur_sigma_units="voxels"' in source
    assert 'noise_std_units="intensity"' in source
