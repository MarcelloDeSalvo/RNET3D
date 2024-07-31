import os
import shutil
import subprocess
from ants import atropos, get_ants_data, image_read, resample_image, get_mask

def main():
    # Save the working directory
    wdir = os.getcwd()
    imgdir = os.getcwd()
    print(wdir)
    # Load the list from the file
    with open(os.path.join(wdir, 'list.txt'), 'r') as file:
        file_content = file.read().splitlines()

    print(file_content)
    for NAME in file_content:
        print(NAME)
        # Change to the specified directory
        os.chdir(os.path.join(imgdir, NAME))

        # To re-scale in order to match with templates - only required once
        shutil.copy(f"{NAME}_anat_orig.nii.gz", f"{NAME}_anat_orig_x10.nii.gz")
        subprocess.run(["3drefit", "-xyzscale", "10", f"{NAME}_anat_orig_x10.nii.gz"])

        # Magnetic field bias correction
        subprocess.run(["N4BiasFieldCorrection", "-d", "3", "-i", f"{NAME}_anat_orig_x10.nii.gz", "-o", f"{NAME}_Anat_N4.nii.gz"])

    os.chdir(wdir)

if __name__ == "__main__":
    main()