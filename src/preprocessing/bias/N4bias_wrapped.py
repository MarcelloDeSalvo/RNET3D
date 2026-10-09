"""Correct NIfTI volumes in ./source using the shared preprocessing step."""

import sys
from pathlib import Path

import nibabel as nib


REPO_ROOT: Path = Path(__file__).resolve().parents[3]
repo_path: str = str(REPO_ROOT)
if repo_path not in sys.path:
    sys.path.insert(0, repo_path)

from src.preprocessing.preprocessor import N4BiasFieldCorrection, NiftiImage


def main() -> None:
    """Save corrected images under ./source/N4, preserving input files."""
    source_directory: Path = Path.cwd() / "source"
    output_directory: Path = source_directory / "N4"

    if not source_directory.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {source_directory}")

    output_directory.mkdir(exist_ok=True)
    correction = N4BiasFieldCorrection(
        masking=True,
        rescale_intensities=True,
        spline_parameters=[1, 1, 1],
        spline_order=3,
    )

    input_paths = source_directory.glob("*.nii.gz")
    image_paths: list[Path] = sorted(input_paths)
    for image_path in image_paths:
        print(image_path.name)
        subject_name: str = image_path.name.removesuffix(".nii.gz").removesuffix("_anat_orig")
        output_path: Path = output_directory / f"{subject_name}_N4.nii.gz"

        image: NiftiImage = nib.load(image_path)
        corrected: NiftiImage = correction.execute(image)

        nib.save(corrected, output_path)


if __name__ == "__main__":
    main()
