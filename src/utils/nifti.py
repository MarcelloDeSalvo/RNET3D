import os
import gzip
import shutil
import nibabel as nib
import numpy as np
from fsl.wrappers import fslsplit, fslmerge

def compress_nifti(input_nii_path, remove_original=True):
    '''
    Compresses a NIfTI file into a gzip file.
    
    Parameters:
    - input_nii_path (str): The path to the NIfTI file to compress.
    - remove_original (bool): Whether to remove the original NIfTI file after compressing it.
    '''
    # Load the NIfTI file
    nii_img = nib.load(input_nii_path)
    
    # Save the NIfTI data to a temporary file
    temp_nii_path = "temp.nii"
    nib.save(nii_img, temp_nii_path)
    
    # Compress the temporary NIfTI file into a gzip file
    with open(temp_nii_path, 'rb') as f_in:
        with gzip.open(input_nii_path + '.gz', 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    
    # Clean up the temporary NIfTI file
    os.remove(temp_nii_path)

    # Remove the original NIfTI file
    if remove_original:
        os.remove(input_nii_path)

def compress_nifti_dir(input_dir, remove_original=True):
    '''
    Compresses all NIfTI files in a directory into gzip files.
    
    Parameters:
    - input_dir (str): The path to the directory containing the NIfTI files to compress.
    - remove_original (bool): Whether to remove the original NIfTI files after compressing them.
    '''
    # Get the paths to all NIfTI files in the directory
    nii_paths = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.nii')]

    # Compress each NIfTI file
    for nii_path in nii_paths:
        compress_nifti(nii_path, remove_original)

def load_nifti(path):
    '''
    Loads a NIfTI file.
    
    Parameters:
    - path (str): The path to the NIfTI file.

    Returns:
    - nii_img (Nifti1Image): The NIfTI file.
    - nii_data (np.ndarray): The NIfTI file's data.
    '''
    # Load the NIfTI file
    nii_img = nib.load(path)

    # Get the NIfTI file's data
    nii_data = nii_img.get_fdata()

    return nii_img, nii_data

def split_nifti(input_nii_path, output_dir, dim='t'):
    '''
    Splits a 4D NIfTI file into multiple 3D NIfTI files on the time axis by default.
    '''
    fslsplit(input_nii_path, output_dir, dim=dim)

def merge_nifti(images_paths, output_nii_path, axis='t'):
    '''
    Merges multiple 3D NIfTI files into a single 4D NIfTI file on the time axis.
    '''
    fslmerge(axis, output=output_nii_path, images=images_paths)
    
def estimate_volume(image, resolution=None, verbose=False):
    '''
    Estimate volume of a segmentation mask in mm^3.

    Returns:
    - volume (dict): A dictionary containing the volume of each class in the mask.
    '''
    data = image.get_fdata()

    # determine number of classes
    num_classes = len(np.unique(data))
    if verbose: print("Number of classes: ", num_classes)

    # get zooms and print them
    if resolution is None:
        resolution = image.header.get_zooms()
        if verbose: print("Voxel dimensions: " , resolution)

    # calculate volume for each class
    classes = np.unique(data)

    # dict to store volume for each class if present
    volume = {i:0 for i in classes}

    if verbose: print("\n- Classes present in the mask: ", classes)
    for i in classes:
        n_voxels = np.sum(data == i)

        # calculate volume for each class in mm^3
        _volume = (n_voxels * resolution[0] * resolution[1] * resolution[2])
        if verbose: print(f"Class {i} has {n_voxels} voxels and a volume of {_volume} mm^3")
        volume[i] = _volume

    return volume