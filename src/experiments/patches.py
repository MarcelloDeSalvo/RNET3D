"""Saved validation/test crop coordinates shared by runs, folds and studies."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import nibabel as nib
import numpy as np
from scipy.stats import truncnorm

from ..preprocessing.preprocessor import CorrectX10, Flip, Preprocessor, RandomCropping
from ..utils.loader import SubjectKey

if TYPE_CHECKING:
    from .data import PreparedData
    from ..preprocessing.preprocessor import ImageInput, PreprocessResult


@dataclass(frozen=True)
class EvaluationPatch:
    """One crop on the padded, reoriented/resampled scan grid.

    Coordinates are voxel centers before cropping. Padding is shared with the
    MRI, ROI and brain targets. Affine and shape identify the uncropped grid.
    """

    scan: SubjectKey
    index: int
    center: tuple[int, int, int]
    padding: tuple[tuple[int, int], ...]
    shape: tuple[int, int, int]
    affine: tuple[tuple[float, ...], ...]
    flipped: bool = False


class FixedPatchPreprocessor(Preprocessor):
    """Apply the deterministic full-volume pipeline, then one saved crop/pair."""

    def __init__(self, base: Preprocessor, patch: EvaluationPatch, size: tuple[int, ...]) -> None:
        """Share the base steps and keep the sample's geometry explicit."""
        super().__init__(base.steps)
        self.patch: EvaluationPatch = patch
        self.crop: RandomCropping = RandomCropping(size, padding=True)

    def preprocess(
        self, imgs: ImageInput = None, masks: ImageInput = None, path: str | None = None,
    ) -> PreprocessResult:
        """Crop aligned MRI/masks and optionally reflect their anatomical LR axis."""
        if imgs is None or isinstance(imgs, list):
            raise ValueError("Fixed evaluation patches require one MRI")

        image, processed_masks = super().preprocess(imgs, masks, path)
        patch = self.patch
        if image.shape != patch.shape or not np.allclose(image.affine, patch.affine, rtol=0, atol=1e-6):
            raise ValueError("Saved evaluation patch grid differs from the processed scan")

        crop_parameters = {"center": patch.center, "pad_values": patch.padding}
        image = self.crop.execute(image, **crop_parameters)
        processed_masks = [self.crop.execute(mask, is_mask=True, **crop_parameters) for mask in processed_masks]

        if patch.flipped:
            codes = nib.aff2axcodes(image.affine)
            axis = next(index for index, code in enumerate(codes) if code in ("L", "R"))
            reflection = Flip(anatomical_axis="LR", probability=1.0)
            image = reflection.execute(image, axis=axis)
            processed_masks = [reflection.execute(mask, is_mask=True, axis=axis) for mask in processed_masks]

        return image, processed_masks


@dataclass
class EvaluationPatches:
    """An auditable collection covering every selected scan, regardless of split."""

    contract: dict[str, Any]
    samples: list[EvaluationPatch]

    @staticmethod
    def expected_contract(data: PreparedData) -> dict[str, Any]:
        """Bind coordinates to inputs, reference, grid and sampling policy."""
        config = data.config
        preprocessing = config.preprocessing
        reference_digest = hashlib.sha256(Path(preprocessing.reference).read_bytes()).hexdigest()
        contract = {
            "version": 1, "sampling": asdict(config.evaluation_patches),
            "patch_size": preprocessing.patch_size, "spacing_mm": preprocessing.spacing_mm,
            "correct_units": preprocessing.correct_units, "reference_sha256": reference_digest,
            "data": data.contract(),
        }
        if preprocessing.correct_units:
            contract["unit_correction_version"] = CorrectX10.geometry_version
        return json.loads(json.dumps(contract))

    @classmethod
    def create(cls, data: PreparedData) -> EvaluationPatches:
        """Sample center-biased starts with a local RNG stable for each scan."""
        from .data import make_preprocessor

        config = data.config
        settings = config.evaluation_patches
        settings.validate()
        size = config.preprocessing.patch_size
        pipeline = make_preprocessor(config, stage="inference")
        samples: list[EvaluationPatch] = []

        for key in sorted(data.loader.get_subjects_list()):
            scan = data.loader.get_subject(key)
            image = pipeline.preprocess(scan.image)
            shape = tuple(image.shape)
            affine = tuple(tuple(row) for row in image.affine)
            padding: list[tuple[int, int]] = []
            for available, wanted in zip(shape, size):
                difference = max(0, wanted - available)
                before = difference // 2
                padding.append((before, difference - before))

            padded_shape = [max(available, wanted) for available, wanted in zip(shape, size)]
            seed_text = json.dumps([settings.seed, key])
            seed_digest = hashlib.sha256(seed_text.encode()).digest()
            seed = int.from_bytes(seed_digest[:8], "little")
            rng = np.random.default_rng(seed)

            for index in range(settings.patches_per_scan):
                center: list[int] = []
                for available, wanted in zip(padded_shape, size):
                    maximum = available - wanted
                    start = maximum // 2
                    if maximum > 0 and settings.sampling == "center_biased":
                        mean = maximum / 2
                        std = available / 4
                        lower = -mean / std
                        upper = (maximum - mean) / std
                        sampled = truncnorm.rvs(lower, upper, loc=mean, scale=std, random_state=rng)
                        start = int(np.rint(sampled))

                    center.append(start + wanted // 2)

                reflections = [False]
                if settings.left_right_pairs:
                    reflections.append(True)

                for flipped in reflections:
                    patch = EvaluationPatch(
                        scan=key, index=index, center=tuple(center), padding=tuple(padding),
                        shape=shape, affine=affine, flipped=flipped,
                    )
                    samples.append(patch)

        return cls(contract=cls.expected_contract(data), samples=samples)

    def save(self, path: Path) -> None:
        """Write exact coordinates and a digest to detect accidental changes."""
        payload = {"contract": self.contract, "samples": [asdict(sample) for sample in self.samples]}
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        payload["sha256"] = digest
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path, data: PreparedData) -> EvaluationPatches:
        """Reuse saved coordinates; reject altered sampling, inputs or geometry."""
        payload = json.loads(path.read_text(encoding="utf-8"))
        saved_digest = payload.pop("sha256")
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        if saved_digest != digest:
            raise ValueError("Saved evaluation patches were modified")
        expected = cls.expected_contract(data)
        if payload["contract"].get("unit_correction_version") != expected.get("unit_correction_version"):
            raise ValueError(
                "Saved evaluation patches use a different x10 unit-correction policy. "
                "Start a fresh run/study folder so training and evaluation use consistent geometry."
            )
        if payload["contract"] != expected:
            raise ValueError("Saved evaluation patch configuration or inputs differ")

        samples: list[EvaluationPatch] = []
        for values in payload["samples"]:
            values["scan"] = tuple(values["scan"])
            for name in ("center", "shape"):
                values[name] = tuple(values[name])
            for name in ("padding", "affine"):
                values[name] = tuple(tuple(row) for row in values[name])
            samples.append(EvaluationPatch(**values))

        return cls(contract=payload["contract"], samples=samples)
