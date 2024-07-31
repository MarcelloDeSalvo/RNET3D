from ants import process_args as pargs
from ants import get_mask
from ants.core import ants_image as iio
import ants.utils as utils
import os
import ants


def ants_n4_bias_field_wrapper(
    image,
    mask=None,
    automatic_masking=True,
    rescale_intensities=True,
    shrink_factor=4,
    convergence={"iters": [50, 50, 50, 50], "tol": 0},
    spline_param=[1,1,1],
    spline_order=3,
    return_bias_field=False,
    verbose=False,
    weight_mask=None,
):
    """
    N4 Bias Field Correction wrapper that can disable or enable the masking of the bias field.

    ANTsR function: `n4BiasFieldCorrection`

    Arguments
    ---------
    image : ANTsImage
        image to bias correct

    mask : ANTsImage
        input mask, if one is not passed one will be made

    rescale_intensities : boolean
        At each iteration, a new intensity mapping is
        calculated and applied but there is nothing which constrains the new
        intensity range to be within certain values. The result is that the
        range can "drift" from the original at each iteration. This option
        rescales to the [min,max] range of the original image intensities within
        the user-specified mask. A mask is required to perform rescaling.  Default
        is False in ANTsR/ANTsPy but True in ANTs.

    shrink_factor : scalar
        Shrink factor for multi-resolution correction, typically integer less than 4

    convergence : dict w/ keys `iters` and `tol`
        iters : vector of maximum number of iterations for each level
        tol : the convergence tolerance.  Default tolerance is 1e-7 in ANTsR/ANTsPy but 0.0 in ANTs.

    spline_param : float or vector
        Parameter controlling number of control points in spline. Either single value,
        indicating the spacing in each direction, or vector with one entry per
        dimension of image, indicating the mesh size. If None, defaults to mesh size of 1 in all
        dimensions.

    return_bias_field : boolean
        Return bias field instead of bias corrected image.

    verbose : boolean
        enables verbose output.

    weight_mask : ANTsImage (optional)
        antsImage of weight mask

    Returns
    -------
    ANTsImage

    Example
    -------
    >>> image = ants.image_read( ants.get_ants_data('r16') )
    >>> image_n4 = ants.n4_bias_field_correction(image)
    """
    if image.pixeltype != "float":
        image = image.clone("float")
    iters = convergence["iters"]
    tol = convergence["tol"]
    if mask is None and automatic_masking == True:
        mask = get_mask(image)
    elif mask is None and automatic_masking == False:
        mask = None
    if spline_param is None:
        spline_param = [1] * image.dimension

    N4_CONVERGENCE_1 = "[%s, %.10f]" % ("x".join([str(it) for it in iters]), tol)
    N4_SHRINK_FACTOR_1 = str(shrink_factor)
    if (not isinstance(spline_param, (list, tuple))) or (len(spline_param) == 1):
        N4_BSPLINE_PARAMS = "[%i]" % spline_param
    elif (isinstance(spline_param, (list, tuple))) and (
        len(spline_param) == image.dimension
    ):
        N4_BSPLINE_PARAMS = "[%s, %i]" % ("x".join([str(sp) for sp in spline_param]), spline_order)
    else:
        raise ValueError(
            "Length of splineParam must either be 1 or dimensionality of image"
        )

    if weight_mask is not None:
        if not isinstance(weight_mask, iio.ANTsImage):
            raise ValueError("Weight Image must be an antsImage")

    outimage = image.clone("float")
    outbiasfield = image.clone("float")
    i = utils.get_pointer_string(outimage)
    b = utils.get_pointer_string(outbiasfield)
    output = "[%s,%s]" % (i, b)

    kwargs = {
        "d": outimage.dimension,
        "i": image,
        "w": weight_mask,
        "s": N4_SHRINK_FACTOR_1,
        "c": N4_CONVERGENCE_1,
        "b": N4_BSPLINE_PARAMS,
        "x": mask,
        "r": int(rescale_intensities),
        "o": output,
        "v": int(verbose),
    }

    processed_args = pargs._int_antsProcessArguments(kwargs)
    libfn = utils.get_lib_fn("N4BiasFieldCorrection")
    libfn(processed_args)
    if return_bias_field == True:
        return outbiasfield
    else:
        return outimage
    

# def main():
#     # Save the working directory
#     wdir = os.path.join(os.getcwd(), "source")
#     out = os.path.join(os.getcwd(), "N4")
#     for NAME in os.listdir(wdir):
#         # If file does not end with .nii.gz, skip
#         print(NAME)
#         if not NAME.endswith(".nii.gz"):
#             continue

#         # Remove 'anat_orig' from the name
#         _NAME = NAME.replace("_anat_orig.nii.gz", "")

#         #subprocess.run(["3drefit", "-xyzscale", "10", f"{NAME}_anat_orig_x10.nii.gz"])

#         # Magnetic field bias correction
#         image = ants.image_read( os.path.join(wdir, NAME))

#         # Default values for Ants N4BiasFieldCorrection
#         spline_param = [1,1,1]
#         spline_odrer = 3

#         output = ants_n4_bias_field_wrapper(
#             image=image,
#             automatic_masking=False,
#             spline_param=spline_param,
#             spline_order=spline_odrer,
#         )
        
#         # save to output directory
#         if not os.path.exists(out):
#             os.makedirs(out)
#         output.to_filename(os.path.join(out, f"{_NAME}_N4.nii.gz"))


#     os.chdir(wdir)

# if __name__ == "__main__":
#     main()