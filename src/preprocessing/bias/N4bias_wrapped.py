import os
import shutil
import subprocess
import ants
from n4_bias import ants_n4_bias_field_wrapper

def main():
    # Save the working directory
    wdir = os.join(os.getcwd(), "source")
    out = os.path.join(wdir, "N4")
    for NAME in os.listdir(wdir):
        # If file does not end with .nii.gz, skip
        print(NAME)
        if not NAME.endswith(".nii.gz"):
            continue

        # Remove 'anat_orig' from the name
        _NAME = NAME.replace("_anat_orig.nii.gz", "")

        # Magnetic field bias correction
        image = ants.image_read(f"{NAME}")

        # Default values for Ants N4BiasFieldCorrection
        spline_param = [1,1,1]
        spline_odrer = 3

        output = ants_n4_bias_field_wrapper(
            image=image,
            automatic_masking=True,
            spline_param=spline_param,
            spline_order=spline_odrer,
        )
        
        # save to output directory
        if not os.path.exists(out):
            os.makedirs(out)
        output.to_filename(os.path.join(out, f"{_NAME}_N4.nii.gz"))


    os.chdir(wdir)

if __name__ == "__main__":
    main()