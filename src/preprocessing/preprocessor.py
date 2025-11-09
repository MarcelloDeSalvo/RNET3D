import abc
import ants
import numpy as np
import nibabel as nib
from scipy.ndimage import zoom, rotate
from scipy.stats import truncnorm
from nibabel.processing import *
from nibabel.orientations import *
from preprocessing.n4_bias import ants_n4_bias_field_wrapper
from scipy import ndimage
import SimpleITK as sitk


class PreprocessingStep(abc.ABC):
    """
    Base class for processing steps applied to images and masks.

    Attributes:
    - function: Function to apply to the image.
    - verbose: Print the information about the process.
    - probability: Probability to apply the step.
    """

    def __init__(self, verbose=False, probability=1, global_probability=False):
        self.name = self.__class__.__name__
        self.probability = probability
        self.global_probability = global_probability
        self.verbose = verbose
        self.current_path = None

    @abc.abstractmethod
    def execute(self, image, is_mask=False):
        pass

    def apply(self, imgs=None, masks=None, path=None):
        """
        Apply the function to the images and the masks based on their type.
        """
        if path is not None:
            self.current_path = path
        if np.random.rand() > self.probability:
            return self.return_processed(imgs, masks)
        return self.iterate(imgs, masks)

    def iterate(self, imgs, masks):
        if imgs is not None:
            imgs = self.iterate_list(imgs, self.execute, False)
        if masks is not None:
            masks = self.iterate_list(masks, self.execute, True)

        return self.return_processed(imgs, masks)

    def iterate_list(self, input, callback, *args):
        """
        Utils function to iterate through a list and apply a function to each element.\n
        Checks if the input is a list or a single element.\n
        Return the transformed list or the transformed element.
        """
        if isinstance(input, list):
            for i in range(len(input)):
                input[i] = callback(input[i], *args)
            return input
        else:
            return callback(input, *args)

    def return_processed(self, imgs, masks):
        if masks is not None and imgs is not None:
            return imgs, masks

        if masks is not None:
            return masks
        return imgs

    def get_global_probability(self):
        return self.global_probability

    def set_probability(self, probability):
        self.probability = probability

    def get_configuration(self):
        configuration = f"\n> Pre-processing step: {self.name}\n"
        configuration += f"\t- name: {self.name}"

        # Add the attributes of the class with their values
        for key, value in self.__dict__.items():
            if isinstance(value, (int, float, str, list, tuple)):
                configuration += f"\n\t- {key}: {value}"
            else:
                configuration += f"\n\t- {key}: {type(value)}"

        return configuration


class Preprocessor:
    """
    PreprocessorMRI class is used to preprocess MRI images in an ordered pipeline.

    Parameters:
    - steps: list, list of preprocessing steps.
    - verbose: bool, print the information about the preprocessing steps.

    Methods:
    - preprocess: Apply the preprocessing steps to the image and the mask.
    - deprocess: Deprocess the data by reversing the preprocessing steps given a prediction and the original image.
    - get_configuration: Get the configuration of the preprocessing steps.
    - save_configuration: Save the configuration of the preprocessing steps to a file.

    example:
    ```
    # The order of the steps is important!
    steps = [
        Reorient(ref_img),
        CorrectX10(),
        Resample(target_resolution),
        N4BiasFieldCorrection(),
        Normalize(mode='minmax')
    ]

    preprocessor = Preprocessor(steps, verbose=True)
    preprocessor.preprocess(image, mask)
    ```
    """

    def __init__(self, steps, verbose=False):
        self.steps = steps
        self.verbose = verbose
        self.global_probability = 0

    def preprocess(self, imgs=None, masks=None, path=None):
        if self.verbose:
            print("Preprocessing steps{\n")

        for step in self.steps:
            if self.verbose:
                print(step.get_configuration())
            if imgs is not None:
                if masks is not None:
                    imgs, masks = step.apply(imgs, masks, path)
                else:
                    imgs = step.apply(imgs=imgs, path=path)
            else:
                masks = step.apply(masks=masks, path=path)

        if self.verbose:
            print("}>\n")

        if masks is not None and imgs is not None:
            return imgs, masks

        if masks is not None:
            return masks
        return imgs

    def deprocess(
        self,
        prediction,
        original_image,
        mapping,
        save_path=None,
        verbose=False,
        reference_resolution=(0.1, 0.1, 0.1),
        flip_axis=None,
        scale_factor=None,
    ):
        """
        Reverses the preprocessing steps given the original image as reference.
        """
        if scale_factor is not None:
            prediction = Scale(scale_factor, verbose=verbose).execute(
                prediction, is_mask=True
            )

        reoriented_prediction = Reorient(original_image, verbose=verbose).execute(
            prediction
        )
        corrected_original = CorrectX10(reference_resolution, verbose=verbose).execute(
            original_image
        )
        corrected_resolution = corrected_original.header.get_zooms()
        resampled_prediction = Resample(corrected_resolution, verbose=verbose).execute(
            reoriented_prediction, is_mask=True
        )
        matched_prediction = PadderCutter(
            original_image.shape, verbose=verbose
        ).execute(resampled_prediction)
        remapped_prediction = MapLabels(mapping=mapping, verbose=verbose).execute(
            matched_prediction, is_mask=True
        )

        if flip_axis is not None:
            remapped_prediction = Flip(flip_axis, verbose=verbose).execute(
                remapped_prediction
            )

        if save_path is not None:
            SaveNifti(
                path=save_path, reference=original_image, verbose=verbose
            ).execute(remapped_prediction)

        return remapped_prediction

    def get_configuration(self):
        configuration = "Preprocessor class:"
        for step in self.steps:
            configuration += step.get_configuration()
        return configuration

    def save_configuration(self, path):
        with open(path, "w") as file:
            file.write(self.get_configuration())
        print(f"Configuration saved to {path}")

    def set_global_probability(self, probability):
        self.global_probability = probability
        for step in self.steps:
            if step.get_global_probability():
                step.set_probability(probability)


# Pre-processing steps
class Resample(PreprocessingStep):
    """
    Resample class is used to resample the image to a target resolution.

    Parameters:
    - target_resolution: tuple, target resolution of the image.
    - interpolation: int, interpolation method used for resampling.
    - verbose: bool, print the information about the resampling process.
    """

    def __init__(self, target_resolution, interpolation=1, verbose=False):
        super().__init__(verbose)
        self.target_resolution = target_resolution
        self.interpolation = interpolation

    def execute(self, image, is_mask=False):
        """
        Apply the resampling to the image based on the target resolution and the interpolation method.
        """

        affine = nib.affines.rescale_affine(
            image.affine, shape=image.shape, zooms=self.target_resolution
        )

        voxel_dims = image.header.get_zooms()
        if self.verbose:
            print("\t|- Original voxel dimensions: ", voxel_dims)

        # if they are close to the target resolution, skip resampling
        if np.allclose(voxel_dims, self.target_resolution, atol=1e-2):
            if self.verbose:
                print(
                    "\t|- Voxel dimensions are already close to the target resolution."
                )
            image.set_qform(affine, code=1)
            image.set_sform(affine, code=1)
            return image

        scale_factors = [
            current_dim / target_dim
            for current_dim, target_dim in zip(voxel_dims, self.target_resolution)
        ]
        if self.verbose:
            print("\t|- Scale factors: ", scale_factors)

        _order = 0 if is_mask else self.interpolation
        _mode = "nearest" if is_mask else "constant"
        resampled_data = zoom(
            image.get_fdata(), scale_factors, order=_order, mode=_mode
        )

        if self.verbose:
            print("\t|- Resampled image shape: ", resampled_data.shape)

        # Make qform/sform explicit & identical (newer version ITK-SNAP require this)
        resampled = nib.Nifti1Image(resampled_data, affine, None)
        resampled.set_qform(affine, code=1)
        resampled.set_sform(affine, code=1)
        return resampled


class CorrectX10(PreprocessingStep):
    """
    CorrectX10 class is used to correct the voxel dimensions that were wrongly scaled by factors of 10 based on the reference resolution.
    """

    def __init__(
        self, reference_resolution=(0.1, 0.1, 0.1), threshold=5, verbose=False
    ):
        super().__init__(verbose)
        self.reference_resolution = reference_resolution
        self.threshold = threshold

    def execute(self, image, is_mask=False):
        """
        Correct the voxel dimensions that were wrongly scaled by factors of 10 based on the reference resolution.
        """
        voxel_dims = image.header.get_zooms()
        if self.verbose:
            print("\t|- Original voxel dimensions: ", voxel_dims)

        # Detect factors of 10
        detected_factors = [
            np.round(dim / ref_dim)
            for dim, ref_dim in zip(voxel_dims, self.reference_resolution)
        ]
        if self.verbose:
            print("\t|- Detected factors: ", detected_factors)

        # Divide by a factor of 10 if one of the factors is close to a factor of 10
        if np.any(
            [np.abs(factor - 10) < self.threshold for factor in detected_factors]
        ):
            voxel_dims = [dim / 10 for dim in voxel_dims]

        if self.verbose:
            print("\t|- Corrected factors: ", voxel_dims)
        new_affine = nib.affines.rescale_affine(
            image.affine, shape=image.shape, zooms=voxel_dims
        )

        return nib.Nifti1Image(
            image.get_fdata(), affine=new_affine, header=image.header, dtype=np.float64
        )


class MapLabels(PreprocessingStep):
    """
    MapLabels class is used to map the labels to a new set of labels.
    """

    def __init__(self, labels=None, mapping=None, reverse=None, verbose=False):
        """
        Map the labels to a new set of labels by providing a mapping dictionary or a list of labels.
        If only a list of labels is provided, the mapping will be done automatically by enumerating the labels starting from 0.
        """
        super().__init__(verbose)
        self.labels = labels

        if mapping is None:
            self.mapping = {
                old_label: new_label for new_label, old_label in enumerate(labels)
            }
        elif reverse:
            if self.verbose:
                print("\t|- Reversing mapping...")
            self.mapping = {
                new_label: old_label for old_label, new_label in mapping.items()
            }
        else:
            self.mapping = mapping

        if self.verbose:
            print("\t|- Mapping labels: ", self.mapping)

    def execute(self, image, is_mask=False):
        """
        Apply the mapping to the image based on the mapping dictionary.\n
        If the image is not a mask, return the image as it is.
        """
        if not is_mask:
            return image

        data = image.get_fdata()
        if self.verbose:
            print("\t|- Unique labels: ", np.unique(data))
        if self.verbose:
            print("\t|- Data type: ", data.dtype)

        tmp_data = np.zeros(data.shape)
        for old_label, new_label in self.mapping.items():
            tmp_data[data == old_label] = new_label
            if self.verbose:
                print(f"\t|- Mapping label {old_label} to {new_label}")
        if self.verbose:
            print("\t|- Unique labels: ", np.unique(tmp_data))
        return nib.Nifti1Image(tmp_data, image.affine, image.header, dtype=np.float64)


class Reorient(PreprocessingStep):
    """
    Reorient class is used to reorient the image based on the reference image.
    """

    def __init__(self, ref_img, verbose=False):
        super().__init__(verbose)
        self.ref_img = ref_img

    def execute(self, image, is_mask=False):
        """
        Reorient the image based on the reference image.
        """
        old_ornt = nib.io_orientation(image.affine)
        if self.verbose:
            print("\t|- Old orientation: ", nib.orientations.ornt2axcodes(old_ornt))

        copied_ort = nib.io_orientation(self.ref_img.affine)
        copied_codes = nib.orientations.ornt2axcodes(copied_ort)
        new_ornt = axcodes2ornt(copied_codes)
        if self.verbose:
            print("\t|- New orientation: ", copied_codes)

        transform = ornt_transform(old_ornt, new_ornt)

        image = image.as_reoriented(transform)

        return nib.Nifti1Image(
            image.get_fdata(), image.affine, image.header, dtype=np.float64
        )


class N4BiasFieldCorrection(PreprocessingStep):
    """
    N4BiasFieldCorrection class is used to apply N4 bias field correction to the image.
    """

    def __init__(
        self,
        tolerance=0,
        iters=[50, 50, 50, 50],
        spline_parameters=[1, 1, 1],
        spline_order=3,
        rescale_intensities=False,
        masking=False,
        verbose=False,
    ):
        super().__init__(verbose)
        self.tolerance = tolerance
        self.iters = iters
        self.spline_parameters = spline_parameters
        self.spline_order = spline_order
        self.rescale_intensities = rescale_intensities
        self.masking = masking

    def execute(self, image, is_mask=False):
        """
        Apply the N4 bias field correction to the image based on the parameters.\n
        If the image is a mask, return the image as it is.
        """
        if is_mask:
            return image

        ants_image = ants.from_numpy(image.get_fdata())
        convergence = {"iters": self.iters, "tol": self.tolerance}

        ants_image = ants_n4_bias_field_wrapper(
            image=ants_image,
            rescale_intensities=self.rescale_intensities,
            automatic_masking=self.masking,
            convergence=convergence,
            spline_param=self.spline_parameters,
            spline_order=self.spline_order,
        )
        return nib.Nifti1Image(
            ants_image.numpy(), image.affine, image.header, dtype=np.float64
        )


class Normalize(PreprocessingStep):
    """
    Normalize class is used to normalize the image using minmax or zscore normalization.
    """

    def __init__(self, mode="zscore", verbose=False):
        super().__init__(verbose)
        self.mode = mode

    def execute(self, image, is_mask=False):
        """
        Apply the normalization to the image based on the mode.\n
        If the image is a mask, return the image as it is.
        """
        if is_mask:
            return image

        data = image.get_fdata()
        if self.mode == "minmax":
            data = (data - data.min()) / (data.max() - data.min())
        elif self.mode == "zscore":
            data = (data - data.mean()) / data.std()
        else:
            print("ERROR: Unknown normalization mode")
        return nib.Nifti1Image(data, image.affine, image.header, dtype=np.float64)


class Padder(PreprocessingStep):
    def __init__(self, dim, fill_mode="constant", pad_mode="center", verbose=False):
        """
        Apply random or center cropping to the image based on the mode 'random' or 'center'.

        Parameters:
        - dim: tuple, cropping dimensions.
        """
        super().__init__(verbose)
        self.dim = dim
        self.pad_mode = pad_mode
        self.fill_mode = fill_mode

    def generate_padding_values(self, img_shape, target_shape):
        """
        Generate padding values for each dimension to achieve the target shape.
        """
        if self.pad_mode == "random":
            return self.random_padding_values(img_shape, target_shape)
        elif self.pad_mode == "center":
            return self.center_padding_values(img_shape, target_shape)
        else:
            raise ValueError("Invalid mode. Please choose between random and center.")

    def random_padding_values(self, img_shape, target_shape):
        """
        Generate padding values for each dimension to achieve the target shape.
        Randomize the padding amounts for each side independently.
        """
        diff_x = target_shape[0] - img_shape[0]
        diff_y = target_shape[1] - img_shape[1]
        diff_z = target_shape[2] - img_shape[2]

        pad_x1 = pad_x2 = pad_y1 = pad_y2 = pad_z1 = pad_z2 = 0

        if diff_x > 0:
            pad_x1 = np.random.randint(0, diff_x + 1)
            pad_x2 = diff_x - pad_x1
        if diff_y > 0:
            pad_y1 = np.random.randint(0, diff_y + 1)
            pad_y2 = diff_y - pad_y1
        if diff_z > 0:
            pad_z1 = np.random.randint(0, diff_z + 1)
            pad_z2 = diff_z - pad_z1

        return ((pad_x1, pad_x2), (pad_y1, pad_y2), (pad_z1, pad_z2))

    def center_padding_values(self, img_shape, target_shape):
        """
        Generate padding values for each dimension to achieve the target shape.
        Center the image and pad equally on each side.
        """
        diff_x = target_shape[0] - img_shape[0]
        diff_y = target_shape[1] - img_shape[1]
        diff_z = target_shape[2] - img_shape[2]

        pad_x1 = pad_x2 = pad_y1 = pad_y2 = pad_z1 = pad_z2 = 0

        if diff_x > 0:
            pad_x1 = diff_x // 2
            pad_x2 = diff_x - pad_x1
        if diff_y > 0:
            pad_y1 = diff_y // 2
            pad_y2 = diff_y - pad_y1
        if diff_z > 0:
            pad_z1 = diff_z // 2
            pad_z2 = diff_z - pad_z1

        return ((pad_x1, pad_x2), (pad_y1, pad_y2), (pad_z1, pad_z2))

    def execute(self, img, pad_values, fill_mode="reflect"):
        """
        Pad the image using the specified padding values.
        """
        img_data = img.get_fdata()

        # Apply the specified padding values
        img_data = np.pad(img_data, pad_values, mode=fill_mode)

        return Nifti1Image(img_data, img.affine, img.header)

    def iterate(self, imgs=None, masks=None):
        """
        Apply the padding to the images and the masks based on their type.
        All images and masks will be padded based on the same padding values since they should have the same shape.
        """
        # Generate padding values using the shape of the first image (assuming all images have the same shape)
        if imgs:
            img_shape = (
                imgs[0].get_fdata().shape
                if isinstance(imgs, list)
                else imgs.get_fdata().shape
            )
            padding_values = self.generate_padding_values(img_shape, self.dim)

        if imgs:
            imgs = self.iterate_list(imgs, self.execute, padding_values, self.fill_mode)
        if masks:
            masks = self.iterate_list(masks, self.execute, padding_values, "constant")

        return super().return_processed(imgs, masks)


class PadderCutter(PreprocessingStep):
    """
    PadderCutter class is used to pad or cut the image to the desired size.
    """

    def __init__(self, img_size, verbose=False):
        super().__init__(verbose)
        self.img_size = img_size

    def execute(self, image, is_mask=False):
        """
        Pad or cut the image to the desired size based on the image size.
        """
        diff_x = self.img_size[0] - image.shape[0]
        diff_y = self.img_size[1] - image.shape[1]
        diff_z = self.img_size[2] - image.shape[2]

        pad_x1 = pad_x2 = pad_y1 = pad_y2 = pad_z1 = pad_z2 = 0

        if diff_x > 0:
            pad_x1 = diff_x // 2
            pad_x2 = diff_x - pad_x1
        if diff_y > 0:
            pad_y1 = diff_y // 2
            pad_y2 = diff_y - pad_y1
        if diff_z > 0:
            pad_z1 = diff_z // 2
            pad_z2 = diff_z - pad_z1

        data = image.get_fdata()

        # Pad or cut the image along each dimension
        if diff_x > 0:
            data = np.pad(data, ((pad_x1, pad_x2), (0, 0), (0, 0)), mode="constant")
        elif diff_x < 0:
            data = data[: self.img_size[0], :, :]

        if diff_y > 0:
            data = np.pad(data, ((0, 0), (pad_y1, pad_y2), (0, 0)), mode="constant")
        elif diff_y < 0:
            data = data[:, : self.img_size[1], :]

        if diff_z > 0:
            data = np.pad(data, ((0, 0), (0, 0), (pad_z1, pad_z2)), mode="constant")
        elif diff_z < 0:
            data = data[:, :, : self.img_size[2]]

        return nib.Nifti1Image(data, image.affine, image.header, dtype=np.float64)


class SaveNifti(PreprocessingStep):
    """
    SaveNifti class is used to save the image to a Nifti file.
    """

    def __init__(
        self, path=None, postfix=None, reference=None, verbose=False, replace=None
    ):
        super().__init__(verbose)
        self.path = path
        self.replace = replace
        self.postfix = postfix
        self.reference = reference

    def execute(self, image, is_mask=False):
        """
        Save the image to a Nifti file based on the path.\n
        If the reference image is provided when istantiating the class, the metadata of the reference image will be cloned.
        """
        if self.verbose:
            print(f"\t|- Image type: {type(image)}")

        if self.path is None:
            cur_path = self.current_path
            if self.verbose:
                print(
                    f"\t|- No path provided. Image saved on the current path: {cur_path}"
                )
        else:
            cur_path = self.path

        if self.postfix is not None:
            # Add a postfix to the filename
            if self.replace is not None:
                cur_path = cur_path.replace(self.replace, self.postfix)
            else:
                cur_path = cur_path.replace(".nii.gz", f"{self.postfix}.nii.gz")

        if self.reference is None:
            if self.verbose:
                print(f"\t|- Cloning metadata from the original image.")
            image = nib.Nifti1Image(
                image.get_fdata(), image.affine, image.header, dtype=np.float64
            )

        nib.save(image, cur_path)
        if self.verbose:
            print(f"\t|- Image saved to {cur_path}")
        return image


class Scale(PreprocessingStep):
    """
    Scale class is used to scale the image by a factor.
    """

    def __init__(self, scale_factor, inerpolation=0, verbose=False):
        super().__init__(verbose)
        self.scale_factor = scale_factor
        self.interpolation = inerpolation

    def execute(self, image, is_mask=False):
        """
        Scale the image by a factor.
        """
        data = image.get_fdata()
        data = zoom(data, self.scale_factor, order=self.interpolation)
        return nib.Nifti1Image(data, image.affine, image.header, dtype=np.float64)


# Augmentations classes
class Noise(PreprocessingStep):
    """
    Noise class:\n
    This class is used to apply Gaussian noise to an image with a random standard deviation.

    Attributes:
        noise_range (list): Range of the standard deviation of the Gaussian noise.
        probability (float): Probability of applying the augmentation.
        geometric (bool): If the augmentation is geometric or not, to know if it should be applied to the mask.
    """

    def __init__(self, noise_range, probability=0.3, mean=0):
        super().__init__(probability=probability)
        self.noise_range = noise_range
        self.probability = probability
        self.mean = mean

    def execute(self, image, is_mask=False):
        """
        Apply Gaussian noise to the image with a mean and a standard deviation.
        """
        if is_mask == True:
            return image
        image_data = image.get_fdata()
        std = np.random.uniform(self.noise_range[0], self.noise_range[1])
        noise = np.random.normal(self.mean, std, image.shape)
        return nib.Nifti1Image(
            image_data + noise,
            affine=image.affine,
            header=image.header,
            dtype=np.float64,
        )


class GaussianBlur(PreprocessingStep):
    """
    GaussianBlur class:\n
    This class is used to apply Gaussian blur to an image with a random sigma.

    Attributes:
        gaussian_blur_range (list): Range of the sigma of the Gaussian blur.
        probability (float): Probability of applying the augmentation.
        geometric (bool): If the augmentation is geometric or not, to know if it should be applied to the mask.
    """

    def __init__(self, gaussian_blur_range, probability=0.3):
        super().__init__(probability=probability)
        self.gaussian_blur_range = gaussian_blur_range
        self.probability = probability

    def execute(self, img, is_mask=False):
        """
        Apply Gaussian blur to the image using ndimage.

        Parameters:
            img (np.array): Image to augment.
            sigma (float): Sigma of the Gaussian blur.
        """
        if is_mask == True:
            return img
        img_data = img.get_fdata()
        sigmas = np.random.uniform(
            self.gaussian_blur_range[0], self.gaussian_blur_range[1], 3
        )
        return nib.Nifti1Image(
            ndimage.gaussian_filter(img_data, sigmas),
            affine=img.affine,
            header=img.header,
            dtype=np.float64,
        )


class RandomAffine(PreprocessingStep):
    """
    RandomAffine class:\n
    This class is used to apply a random affine transformation to an image with a random rotation and scale using SimpleITK.
    By default, the interpolation method is set to sitkLinear for the image and sitkNearestNeighbor for the mask.
    Another interpolation method can be set for the images using the interpolation attribute.

    Attributes:
        scale_range (list): Range of the scale factor of the affine transformation.
        rotation_range (list): Range of the rotation angles of the affine transformation.
        probability (float): Probability of applying the augmentation.
        interpolation (int): Interpolation method to resample the image.
        geometric (bool): If the augmentation is geometric or not, to know if it should be applied to the mask.
    """

    def __init__(
        self,
        scale_range=None,
        rotation_range=None,
        probability=0.5,
        interpolation=sitk.sitkNearestNeighbor,
        background_label=0,
    ):
        super().__init__(probability=probability)
        self.scale_range = scale_range
        self.rotation_range = rotation_range
        self.probability = probability
        self.interpolation = interpolation
        self.background_label = background_label

    def iterate(self, imgs=None, masks=None):
        """
        Apply the random affine transformation to the images and the masks based on their type.
        All images and masks will be transformed based on the same angles and scale factor since they should have the same shape.
        """
        # Select random angles and scale factor for both input types
        angles = (
            np.random.uniform(self.rotation_range[0], self.rotation_range[1], 3)
            if self.rotation_range is not None
            else None
        )
        scale_factor = (
            np.random.uniform(self.scale_range[0], self.scale_range[1])
            if self.scale_range is not None
            else None
        )

        if masks is not None:
            masks = self.iterate_list(masks, self.execute, angles, scale_factor, True)
        if imgs is not None:
            imgs = self.iterate_list(imgs, self.execute, angles, scale_factor, False)

        return super().return_processed(imgs, masks)

    def execute(self, img, angles=None, scale_factor=None, is_mask=False):
        """
        Apply random rotation and zoom to the image.
        Use nearest neighbor interpolation method.

        Parameters:
            img (np.array): Image to augment.
            angles (list): List of the rotation angles (in degrees).
            scale_factor (float): Scale factor for zoom.
            is_mask (bool): If the image is a mask or not.
        """
        data = img.get_fdata()
        rotation_center = [x / 2 for x in data.shape[::-1]]
        affine_transform = sitk.AffineTransform(3)
        affine_transform.SetIdentity()

        if angles is not None:
            rotation_transform = sitk.Euler3DTransform(
                rotation_center,
                np.radians(angles[0]),
                np.radians(angles[1]),
                np.radians(angles[2]),
            )
            affine_transform = sitk.CompositeTransform(
                [affine_transform, rotation_transform]
            )

        if scale_factor is not None:
            scale_transform = sitk.ScaleTransform(
                3, [scale_factor, scale_factor, scale_factor]
            )
            scale_transform.SetCenter(rotation_center)
            affine_transform = sitk.CompositeTransform(
                [affine_transform, scale_transform]
            )

        if is_mask:
            transformed_img = sitk.Resample(
                sitk.GetImageFromArray(data),
                affine_transform,
                sitk.sitkNearestNeighbor,
                self.background_label,
                sitk.sitkFloat64,
            )
        else:
            transformed_img = sitk.Resample(
                sitk.GetImageFromArray(data),
                affine_transform,
                self.interpolation,
                0.0,
                sitk.sitkFloat64,
            )

        return nib.Nifti1Image(
            sitk.GetArrayFromImage(transformed_img),
            affine=img.affine,
            header=img.header,
            dtype=np.float64,
        )


class Flip(PreprocessingStep):
    """
    Flip class:\n
    This class is used to apply a random flip to an image.

    Attributes:
        flip_axis (list): List of the axis to flip the image.
        probability (float): Probability of applying the augmentation.
        geometric (bool): If the augmentation is geometric or not, to know if it should be applied to the mask.
    """

    def __init__(self, axis_list, probability=0.5):
        super().__init__(probability=probability, global_probability=True)
        self.axis_list = axis_list

    def iterate(self, imgs=None, masks=None):
        random_axis = np.random.choice(self.axis_list)
        if masks is not None:
            masks = self.iterate_list(masks, self.execute, random_axis)
        if imgs is not None:
            imgs = self.iterate_list(imgs, self.execute, random_axis)
        return super().return_processed(imgs, masks)

    def execute(self, img, random_axis):
        img_data = img.get_fdata()
        return nib.Nifti1Image(
            np.flip(img_data, axis=random_axis),
            affine=img.affine,
            header=img.header,
            dtype=np.float64,
        )


class RandomCropping(PreprocessingStep):
    """
    RandomCropping class is used to apply random or center cropping to the image.
    todo: check input shape and cropping dimensions to see if they are compatible. Implement padding if needed.
    """

    def __init__(self, dim, mode="random", std=None, verbose=False, padding=False):
        """
        Apply random or center cropping to the image based on the mode 'random' or 'center'.

        Parameters:
        - dim: tuple, cropping dimensions.
        - mode: str, cropping mode ('random' or 'center').
        - std: tuple, standard deviation for the center cropping.
        """
        super().__init__(verbose)
        self.dim = dim
        self.mode = mode
        self.std = std
        self.padding = padding

    def get_random_cropping_coordinates(self, img):
        """
        Get random cropping coordinate within the image shape by avoiding the borders.
        """

        def get_random_coordinate(patch_size, img_size):
            if patch_size == img_size:
                return patch_size // 2
            return np.random.randint(patch_size // 2, img_size - patch_size // 2)

        x = get_random_coordinate(self.dim[0], img.shape[0])
        y = get_random_coordinate(self.dim[1], img.shape[1])
        z = get_random_coordinate(self.dim[2], img.shape[2])
        return x, y, z

    def get_center_cropping_coordinates(self, img):
        """
        Get random cropping coordinates with a higher probability towards the center of the image using truncated normal distribution.
        """
        center_x, center_y, center_z = (
            img.shape[0] / 2,
            img.shape[1] / 2,
            img.shape[2] / 2,
        )

        upper_x, upper_y, upper_z = (
            img.shape[0] - self.dim[0] // 2,
            img.shape[1] - self.dim[1] // 2,
            img.shape[2] - self.dim[2] // 2,
        )
        lower_x, lower_y, lower_z = self.dim[0] // 2, self.dim[1] // 2, self.dim[2] // 2
        std_x = self.std if self.std is not None else img.shape[0] // 4
        std_y = self.std if self.std is not None else img.shape[1] // 4
        std_z = self.std if self.std is not None else img.shape[2] // 4

        def truncated_normal(mean, std_dev, lower_bound, upper_bound):
            if lower_bound == upper_bound:
                return lower_bound
            a, b = (lower_bound - mean) / std_dev, (upper_bound - mean) / std_dev
            return truncnorm(a, b, loc=mean, scale=std_dev).rvs()

        x = int(truncated_normal(center_x, std_x, lower_x, upper_x))
        y = int(truncated_normal(center_y, std_y, lower_y, upper_y))
        z = int(truncated_normal(center_z, std_z, lower_z, upper_z))

        return x, y, z

    def select_cropping_coordinates(self, img):
        """
        Select the cropping coordinates based on the mode.
        """
        if self.mode == "random":
            return self.get_random_cropping_coordinates(img)
        elif self.mode == "center":
            return self.get_center_cropping_coordinates(img)
        else:
            raise ValueError("Invalid mode. Please choose between random and center.")

    # Override apply method to apply the same cropping coordinates to all images and masks
    def iterate(self, imgs=None, masks=None):
        """
        Apply the cropping to the images and the masks based on their type.
        All images and masks will be cropped based on the same coordinates since they should have the same shape.
        """
        x, y, z = self.select_cropping_coordinates(
            imgs[0] if isinstance(imgs, list) else imgs
        )

        if masks is not None:
            masks = self.iterate_list(masks, self.execute, x, y, z)
        if imgs is not None:
            imgs = self.iterate_list(imgs, self.execute, x, y, z)

        return super().return_processed(imgs, masks)

    def padding(self, img):
        """
        Pad the image in equal amounts on each side to make the dimensions even.
        """
        img_data = img.get_fdata()
        x, y, z = img_data.shape
        diff_x = self.dim[0] - x
        diff_y = self.dim[1] - y
        diff_z = self.dim[2] - z

        if diff_x > 0:
            img_data = np.pad(img_data, (0, diff_x), mode="constant")
        if diff_y > 0:
            img_data = np.pad(img_data, (0, diff_y), mode="constant")
        if diff_z > 0:
            img_data = np.pad(img_data, (0, diff_z), mode="constant")

        return Nifti1Image(img_data, img.affine, img.header, dtype=np.float64)

    def execute(self, img, x, y, z):
        """
        Crop the image based on the coordinates.
        """
        img_data = img.get_fdata()
        img_data = img_data[
            x - self.dim[0] // 2 : x + self.dim[0] // 2,
            y - self.dim[1] // 2 : y + self.dim[1] // 2,
            z - self.dim[2] // 2 : z + self.dim[2] // 2,
        ]
        return nib.Nifti1Image(img_data, img.affine, img.header, dtype=np.float64)
