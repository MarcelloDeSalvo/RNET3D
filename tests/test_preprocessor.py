"""Regression checks for preprocessing numerics, label integrity, and physical grids."""

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import SimpleITK as sitk

from src.preprocessing import preprocessor as p


def image(data=None, shape=(8, 8, 8), affine=None):
    """Build a small anisotropic NIfTI image, preserving integer fixture types."""
    data = np.arange(np.prod(shape), dtype=float).reshape(shape) if data is None else np.asarray(data)
    affine = np.diag([0.2, 0.3, 0.4, 1.0]) if affine is None else affine
    return nib.Nifti1Image(data, affine, dtype=data.dtype)


def world(img, voxel):
    """Map voxel coordinates to physical RAS coordinates."""
    return nib.affines.apply_affine(img.affine, voxel)


@pytest.mark.parametrize("mode", ["minmax", "zscore"])
def test_constant_normalization_and_mask_passthrough(mode):
    source = image(np.ones((4, 4, 4)))
    result = p.Normalize(mode).execute(source)
    assert np.isfinite(result.get_fdata()).all()
    assert not result.get_fdata().any()
    assert p.Normalize(mode).execute(source, is_mask=True) is source
    assert np.all(source.get_fdata() == 1)


@pytest.mark.parametrize("shape", [(5, 3, 7), (4, 4, 4), (1, 1, 1), (8, 8, 8)])
def test_exact_crop_size_and_world_coordinates(shape):
    source = image()
    center = (4, 4, 4)
    output = p.RandomCropping(shape).execute(source, center=center)
    starts = np.asarray(center) - np.asarray(shape) // 2
    assert output.shape == shape
    np.testing.assert_allclose(world(output, (0, 0, 0)), world(source, starts))
    assert output.get_fdata()[0, 0, 0] == source.get_fdata()[tuple(starts)]


def test_crop_last_valid_start_is_reachable(monkeypatch):
    monkeypatch.setattr(np.random, "randint", lambda high: high - 1)
    crop = p.RandomCropping((5, 5, 5))
    source = image()
    result = crop.apply(source)
    assert result.get_fdata()[-1, -1, -1] == source.get_fdata()[-1, -1, -1]


@pytest.mark.parametrize("mode", ["random", "center"])
def test_crop_padding_preserves_pair_and_origin(mode):
    source = image()
    mask = image(np.ones(source.shape))
    original_affine = source.affine.copy()
    result, result_mask = p.RandomCropping((11, 8, 9), mode=mode, padding=True).apply(source, mask)
    assert result.shape == result_mask.shape == (11, 8, 9)
    np.testing.assert_array_equal(result.affine, result_mask.affine)
    np.testing.assert_allclose(world(result, (1, 0, 0)), world(source, (0, 0, 0)))
    np.testing.assert_array_equal(source.affine, original_affine)
    assert p.RandomCropping((10, 8, 8)).padding(source).shape == (10, 8, 8)


def test_mask_only_padding_and_center_cut():
    source = image(np.ones((8, 8, 8)))
    output = p.Padder((10, 8, 8)).apply(masks=source)
    assert output.shape == (10, 8, 8)
    assert not output.get_fdata()[0].any()
    np.testing.assert_allclose(world(output, (1, 0, 0)), world(source, (0, 0, 0)))
    cut = p.PadderCutter((4, 10, 8)).execute(source)
    np.testing.assert_allclose(world(cut, (0, 1, 0)), world(source, (2, 0, 0)))


@pytest.mark.parametrize("step", [p.Resample((0.1, 0.15, 0.2)), p.Scale(2)])
def test_linear_integer_images_and_discrete_masks(step):
    grid = np.indices((8, 8, 8))[0].astype(np.int16)
    source = image(grid)
    mask = image(np.where(grid > 3, 13, 0).astype(np.int16))
    output, output_mask = step.apply(source, mask)
    values = output.get_fdata()
    assert np.any(values != np.rint(values))  # Linear output must not be cast back to integers.
    assert set(np.unique(output_mask.get_fdata())) == {0, 13}
    np.testing.assert_array_equal(output.affine, output_mask.affine)
    center = (np.asarray(source.shape) - 1) / 2
    np.testing.assert_allclose(world(source, center), world(output, (np.asarray(output.shape) - 1) / 2))
    voxel = (5, 5, 5)
    source_coordinate = nib.affines.apply_affine(np.linalg.inv(source.affine), world(output, voxel))
    assert values[voxel] == pytest.approx(source_coordinate[0])


def test_scale_keeps_masks_discrete_with_linear_image_interpolation():
    step = p.Scale(2, interpolation=1)
    output = step.execute(image(np.pad(np.ones((2, 2, 2)), 1)), is_mask=True)
    assert set(np.unique(output.get_fdata())) == {0, 1}


def test_resample_near_spacing_is_nonmutating():
    source = image()
    original_affine = source.affine.copy()
    output = p.Resample((0.205, 0.3, 0.4)).execute(source)
    np.testing.assert_array_equal(source.affine, original_affine)
    np.testing.assert_allclose(nib.affines.voxel_sizes(output.affine), (0.205, 0.3, 0.4))
    assert output is not source


def test_resample_does_not_expand_unchanged_float32_spacing_axis():
    source = image(shape=(32, 32, 32), affine=np.diag([0.125, 0.5, float(np.float32(0.1)), 1.]))
    assert p.Resample((0.1, 0.1, 0.1)).execute(source).shape == (40, 160, 32)


@pytest.mark.parametrize("shear", [0.00015, 0.3])
@pytest.mark.parametrize("handedness", [-1, 1])
def test_orthogonal_resampling_covers_sheared_grid_and_keeps_labels(shear, handedness):
    from itertools import product

    affine = np.array([
        [handedness * 0.2, shear * 0.3, 0, 5],
        [0, 0.3, 0, -7], [0, 0, 0.4, 9], [0, 0, 0, 1],
    ])
    grid = np.indices((8, 8, 8))[0].astype(np.int16)
    source = image(grid, affine=affine)
    labels = np.zeros(grid.shape, dtype=np.int16)
    labels[2:6, 2:6, 2:6] = 13
    mask = image(labels, affine=affine)
    source_affine = source.affine.copy()
    step = p.Resample((0.1, 0.15, 0.2), orthogonal=True)
    result, result_mask = step.apply(source, mask)

    spacing = nib.affines.voxel_sizes(result.affine)
    direction = result.affine[:3, :3] / spacing
    np.testing.assert_allclose(spacing, (0.1, 0.15, 0.2))
    np.testing.assert_allclose(direction.T @ direction, np.eye(3), atol=1e-12)
    assert np.sign(np.linalg.det(result.affine[:3, :3])) == handedness
    np.testing.assert_array_equal(result.affine, result_mask.affine)
    np.testing.assert_array_equal(source.affine, source_affine)
    assert set(np.unique(result_mask.get_fdata())) == {0, 13}
    assert np.any(result.get_fdata() != np.rint(result.get_fdata()))
    source_center = (np.asarray(source.shape) - 1) / 2
    result_center = (np.asarray(result.shape) - 1) / 2
    np.testing.assert_allclose(world(source, source_center), world(result, result_center))

    limits = [(-0.5, size - 0.5) for size in source.shape]
    corners = np.asarray(list(product(*limits)))
    world_corners = nib.affines.apply_affine(source.affine, corners)
    result_corners = nib.affines.apply_affine(np.linalg.inv(result.affine), world_corners)
    assert np.all(result_corners >= -0.5 - 1e-6)
    assert np.all(result_corners <= np.asarray(result.shape) - 0.5 + 1e-6)
    p._lps_geometry(result)  # The resulting direction is accepted by physical transforms.


def test_orthogonal_resampling_does_not_skip_shear_at_matching_spacing():
    affine = np.diag([0.1, 0.1, 0.1, 1.0])
    affine[0, 1] = 0.000015
    affine[1, 1] = np.sqrt(0.1 ** 2 - affine[0, 1] ** 2)
    source = image(affine=affine)
    unchanged = p.Resample((0.1, 0.1, 0.1)).execute(source)
    np.testing.assert_array_equal(unchanged.affine, source.affine)
    result = p.Resample((0.1, 0.1, 0.1), orthogonal=True).execute(source)
    assert not np.array_equal(result.affine, source.affine)
    p._lps_geometry(result)


def test_orthogonal_option_preserves_existing_oblique_orthogonal_resampling():
    affine = np.array([[0.2, -0.3, 0, 5], [0.2, 0.3, 0, 7], [0, 0, -0.4, 9], [0, 0, 0, 1.]])
    source = image(affine=affine)
    legacy = p.Resample((0.1, 0.1, 0.1)).execute(source)
    orthogonal = p.Resample((0.1, 0.1, 0.1), orthogonal=True).execute(source)
    np.testing.assert_array_equal(orthogonal.affine, legacy.affine)
    np.testing.assert_array_equal(orthogonal.get_fdata(), legacy.get_fdata())


def test_same_oblique_grid_restoration_preserves_boundary_labels():
    affine = np.array([[0.1, -0.2, 0, 11.3], [0.2, 0.1, 0, 5.7], [0, 0, 0.3, -6.1], [0, 0, 0, 1.]])
    source = image(np.ones((8, 8, 8)), affine=affine)
    restored = p.Preprocessor([]).deprocess(source, source, {0: 0, 1: 1})
    np.testing.assert_array_equal(restored.get_fdata(), source.get_fdata())


def test_pad_crop_and_reorient_restore_exact_reference_grid():
    source = image(np.indices((8, 8, 8))[0].astype(np.int16))
    reference = image(affine=np.diag([-0.2, 0.3, 0.4, 1.0]))
    pipeline = p.Preprocessor([p.Padder((12, 12, 12)), p.Reorient(reference)])
    prediction = pipeline.preprocess(masks=source)
    restored = pipeline.deprocess(prediction, source, {i: i for i in range(8)})
    np.testing.assert_array_equal(restored.get_fdata(), source.get_fdata())
    np.testing.assert_array_equal(restored.affine, source.affine)


def test_deprocess_flip_and_save_reference(tmp_path):
    source = image(np.indices((8, 8, 8))[0].astype(np.int16))
    prediction = p.Flip([0]).execute(source, is_mask=True, axis=0)
    np.testing.assert_array_equal(prediction.affine, source.affine)  # Anatomical augmentation.
    output = p.Preprocessor([]).deprocess(prediction, source, {i: i for i in range(8)}, flip_axis=[0], save_path=str(tmp_path / "mask.nii.gz"))
    np.testing.assert_array_equal(output.get_fdata(), source.get_fdata())
    np.testing.assert_allclose(nib.load(tmp_path / "mask.nii.gz").affine, source.affine)


def test_spacing_correction_is_per_axis_and_round_trips():
    source = image(np.ones((8, 8, 8)), affine=np.diag([1.0, 0.5, 0.1, 1.0]))
    pipeline = p.Preprocessor([p.CorrectX10()])
    corrected = pipeline.preprocess(source)
    np.testing.assert_allclose(nib.affines.voxel_sizes(corrected.affine), (0.1, 0.5, 0.1))
    restored = pipeline.deprocess(corrected, source, {0: 0, 1: 1})
    np.testing.assert_array_equal(restored.get_fdata(), source.get_fdata())
    np.testing.assert_array_equal(restored.affine, source.affine)
    genuine_anisotropic = image(affine=np.diag([0.1, 0.6, 0.1, 1.0]))
    np.testing.assert_array_equal(p.CorrectX10().execute(genuine_anisotropic).affine, genuine_anisotropic.affine)


def test_unknown_labels_and_irreversible_mapping():
    source = image(np.full((2, 2, 2), 13))
    with pytest.raises(ValueError, match="Unmapped mask labels"):
        p.MapLabels(mapping={0: 0}).execute(source, is_mask=True)
    assert not p.MapLabels(mapping={0: 0}, unknown="background").execute(source, is_mask=True).get_fdata().any()
    with pytest.raises(ValueError, match="many-to-one"):
        p.MapLabels(mapping={3: 2, 13: 2}, reverse=True)
    assert np.all(p.MapLabels(mapping={13: 3}).execute(source, is_mask=True).get_fdata() == 3)


@pytest.mark.parametrize("extension", [".nii", ".nii.gz"])
def test_save_postfix_and_no_stale_path(tmp_path, extension):
    source = image()
    input_path = tmp_path / ("scan" + extension)
    saver = p.SaveNifti(postfix="_N4")
    saver.apply(source, path=str(input_path))
    assert (tmp_path / ("scan_N4" + extension)).exists()
    with pytest.raises(ValueError, match="output path"):
        saver.apply(source)
    with pytest.raises(ValueError, match="overwrite"):
        p.SaveNifti(postfix="").apply(source, path=str(input_path))
    with pytest.raises(ValueError, match="one image"):
        p.SaveNifti(path=str(input_path)).apply(source, source)


def test_save_resamples_instead_of_only_replacing_header(tmp_path):
    source = image(np.indices((8, 8, 8))[0].astype(np.int16))
    shifted = source.affine.copy()
    shifted[:3, 3] += shifted[:3, 0] * 0.5
    reference = image(affine=shifted)
    path = tmp_path / "aligned.nii.gz"
    output = p.SaveNifti(str(path), reference=reference).execute(source)
    np.testing.assert_array_equal(output.affine, shifted)
    assert output.get_fdata()[3, 3, 3] == pytest.approx(3.5)
    saved = nib.load(path)
    assert saved.get_fdata()[3, 3, 3] == pytest.approx(3.5)
    discrete = p.SaveNifti(str(tmp_path / "mask.nii.gz"), reference=reference).execute(source, is_mask=True)
    assert set(np.unique(discrete.get_fdata())).issubset(set(range(8)))


def test_n4_passes_physical_geometry_and_has_independent_defaults(monkeypatch):
    captured = {}

    def correct(self, ants_image):
        captured.update(origin=ants_image.origin, spacing=ants_image.spacing, direction=ants_image.direction)
        return ants_image

    monkeypatch.setattr(p.N4BiasFieldCorrection, "_correct_bias_field", correct)
    affine = np.array([[0, -0.3, 0, 10], [0.2, 0, 0, 20], [0, 0, 0.4, 30], [0, 0, 0, 1]])
    source = image(affine=affine)
    first, second = p.N4BiasFieldCorrection(), p.N4BiasFieldCorrection()
    first.iters[0] = 1
    first.spline_parameters[0] = 2
    assert second.iters[0] == 50 and second.spline_parameters[0] == 1
    result = second.execute(source)
    np.testing.assert_allclose(captured["origin"], (-10, -20, 30))
    np.testing.assert_allclose(captured["spacing"], (0.2, 0.3, 0.4))
    np.testing.assert_allclose(captured["direction"], [[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
    np.testing.assert_array_equal(result.get_fdata(), source.get_fdata())
    np.testing.assert_array_equal(result.affine, source.affine)


@pytest.mark.parametrize("spline", [2.5, [2.5], (2.5,)])
def test_n4_scalar_spacing_reaches_backend_without_truncation(monkeypatch, spline):
    captured = []
    get_lib_fn = p.ants.utils.get_lib_fn
    monkeypatch.setattr(p.ants.process_args, "_int_antsProcessArguments", lambda kwargs: kwargs)
    monkeypatch.setattr(p.ants.utils, "get_lib_fn", lambda name: captured.append if name == "N4BiasFieldCorrection" else get_lib_fn(name))
    source = image(np.ones((8, 8, 8), dtype=np.float32))
    correction = p.N4BiasFieldCorrection(masking=False, spline_parameters=spline)
    correction.execute(source)
    assert captured[0]["b"] == "[2.5]"


@pytest.mark.parametrize("masking", [False, True])
def test_n4_backend_receives_configured_options(monkeypatch, masking):
    captured = []
    mask_requests = []
    source = image(np.ones((8, 8, 8), dtype=np.float32))
    mask = p.ants.from_numpy(np.ones(source.shape, dtype=np.float32))

    def get_mask(ants_image):
        mask_requests.append(ants_image)
        return mask

    get_lib_fn = p.ants.utils.get_lib_fn
    monkeypatch.setattr(p.ants, "get_mask", get_mask)
    monkeypatch.setattr(p.ants.process_args, "_int_antsProcessArguments", lambda kwargs: kwargs)
    monkeypatch.setattr(p.ants.utils, "get_lib_fn", lambda name: captured.append if name == "N4BiasFieldCorrection" else get_lib_fn(name))

    correction = p.N4BiasFieldCorrection(
        iters=[2, 3], tolerance=0.001, spline_parameters=[2, 3, 4],
        spline_order=2, masking=masking, rescale_intensities=True, verbose=True,
    )
    correction.execute(source)

    arguments = captured[0]
    assert arguments["c"] == "[2x3, 0.0010000000]"
    assert arguments["b"] == "[2x3x4, 2]"
    assert arguments["s"] == "4"
    assert arguments["r"] == 1 and arguments["v"] == 1
    assert arguments["w"] is None
    assert len(mask_requests) == int(masking)
    if masking:
        assert arguments["x"] is mask
    else:
        assert arguments["x"] is None


@pytest.mark.parametrize("shear", [0.0, 0.00015, 0.3])
def test_random_affine_rotates_in_physical_coordinates_and_preserves_labels(shear):
    angle = np.radians(30)
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    affine = np.eye(4)
    affine[:3, :3] = rotation @ np.diag([0.2, 0.5, 0.8])
    affine[:3, 1] += shear * affine[:3, 0]
    affine[:3, 3] = [10, 20, 30]
    coordinates = np.indices((17, 17, 17)).transpose(1, 2, 3, 0)
    physical = nib.affines.apply_affine(affine, coordinates)
    source = image(physical @ [1., 2., 3.], affine=affine)
    transform = p.RandomAffine()
    assert transform.interpolation == sitk.sitkLinear
    output = transform.execute(source, angles=[0, 0, 20], scale_factor=1)
    np.testing.assert_array_equal(output.affine, source.affine)
    point = (9, 10, 8)
    center = world(source, (8, 8, 8))
    angle = np.radians(20)
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    expected_point = center + rotation @ (world(source, point) - center)
    assert output.get_fdata()[point] == pytest.approx(expected_point @ [1, 2, 3])
    mask = image(np.where(coordinates[..., 0] > 8, 13., 0.), affine=affine)
    discrete = transform.execute(mask, is_mask=True, angles=[0, 0, 20])
    assert set(np.unique(discrete.get_fdata())) <= {0, 13}


@pytest.mark.parametrize("shear", [0.00015, 0.3])
def test_sheared_identity_augmentation_keeps_image_and_mask_aligned(shear):
    affine = np.diag([0.2, 0.3, 0.4, 1.0])
    affine[0, 1] = shear * 0.3
    labels = np.zeros((17, 17, 17))
    labels[4:13, 4:13, 4:13] = 13
    source = image(labels, affine=affine)
    mask = image(labels, affine=affine)
    step = p.RandomAffine(
        rotation_range=(0.0, 0.0), scale_range=(1.0, 1.0), probability=1.0,
    )

    output, output_mask = step.apply(source, mask)

    np.testing.assert_allclose(output.get_fdata(), labels, atol=1e-12)
    np.testing.assert_array_equal(output_mask.get_fdata(), labels)
    np.testing.assert_array_equal(output.affine, affine)
    np.testing.assert_array_equal(output_mask.affine, affine)


def test_n4_accepts_sheared_geometry_without_resampling():
    affine = np.diag([0.2, 0.3, 0.4, 1.0])
    affine[0, 1] = 0.1
    coordinates = np.indices((32, 32, 32))
    data = 10.0 + coordinates[0] * 0.1 + coordinates[1] * 0.03
    source = image(data, affine=affine)
    step = p.N4BiasFieldCorrection(iters=[1])

    result = step.execute(source)

    assert np.all(np.isfinite(result.get_fdata()))
    assert result.shape == source.shape
    np.testing.assert_array_equal(result.affine, affine)
    np.testing.assert_array_equal(source.get_fdata(), data)


def test_group_transform_alignment_and_input_list_immutability():
    source = image(np.indices((8, 8, 8))[0].astype(float))
    masks = [source]
    output, output_masks = p.RandomCropping((5, 5, 5)).apply(source, masks)
    np.testing.assert_array_equal(output.get_fdata(), output_masks[0].get_fdata())
    np.testing.assert_array_equal(output.affine, output_masks[0].affine)
    assert masks[0] is source and output_masks is not masks
    shifted = source.affine.copy()
    shifted[0, 3] = 100
    with pytest.raises(ValueError, match="aligned"):
        p.RandomAffine().apply(source, image(affine=shifted))


def test_configuration_preserves_mapping_and_probabilities():
    pipeline = p.Preprocessor([p.MapLabels(mapping={0: 0, 13: 3}), p.Flip([0]), p.Noise([0, 1])])
    pipeline.set_global_probability(0)
    configuration = json.loads(pipeline.get_configuration())
    assert configuration[0]["mapping"] == {"0": 0, "13": 3}
    assert configuration[1]["probability"] == configuration[2]["probability"] == 0


@pytest.mark.parametrize("factory", [
    lambda: p.Normalize("typo"), lambda: p.Padder([1, 2]),
    lambda: p.RandomCropping((0, 3, 3)), lambda: p.RandomCropping((3, 3, 3), std=0),
    lambda: p.Flip([3]), lambda: p.Noise([1, 0]), lambda: p.Noise([-1, 1]),
    lambda: p.RandomAffine(scale_range=[0, 1]), lambda: p.GaussianBlur([0, 1], probability=2),
    lambda: p.Resample((0.1, 0, 0.1)), lambda: p.Scale(-1),
    lambda: p.N4BiasFieldCorrection(iters=[]), lambda: p.MapLabels(),
])
def test_invalid_configuration_fails_early(factory):
    with pytest.raises(ValueError):
        factory()
