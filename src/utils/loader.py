import nibabel as nib
import os

class RodentDatasets:
    '''
    Class to represent a collection of rodent datasets.

    Attributes:
    - dataset_path (str): The main path to the datasets.
    - labels (list): A list of all potentially available labels for each rodent.
    - datasets_dic (dict): A dictionary containing the internal datasets.

    Methods:
    - add_dataset(modality, sub_folder, unavailable_labels, image_suffix, roi_mask_suffix, brain_mask_suffix, sham): Adds an internal dataset to the collection.
    - get_subject(key): Returns the rodent from the dataset.
    - get_subjects_list(): Concatenates all rodents from all datasets into a single list.
    - get_subjects_dic(): Returns a dictionary containing the rodents in the collection.
    - get_modalities(): Returns the modalities of the datasets in the collection.
    - save_ids(path): Saves the IDs of all rodents in the collection to a text file.
    '''
    def __init__(self, labels=None):
        '''
        Initializes the collection of rodent datasets.
        
        Parameters:
        - labels: A list of all potentially available labels for each rodent.
        '''
        self.labels = labels
        self.datasets_dic = {}

    def __create_subject(self, modality, sub_folder, unavailable_labels, image_suffix, roi_mask_suffix, brain_mask_suffix, sham, dataset_path, id, gz=True):
        path = os.path.join(dataset_path, modality)
        if sub_folder:
            sub_path = os.path.join(path, id, sub_folder)
        else:
            sub_path = os.path.join(path, id)

        mice = Rodent(modality,
                      id,
                      os.path.join(sub_path, f'{id}_{image_suffix}' + ('.nii.gz' if gz else '.nii')),
                      os.path.join(sub_path, f'{id}_{roi_mask_suffix}' + ('.nii.gz' if gz else '.nii')),
                      os.path.join(sub_path, f'{id}_{brain_mask_suffix}' + ('.nii.gz' if gz else '.nii')) if brain_mask_suffix else None,
                      unavailable_labels,
                      sham)
        
        return mice

    def add_dataset(self, dataset_path, modality, sub_folder=None, unavailable_labels=None, image_suffix='N4', roi_mask_suffix='Labels', brain_mask_suffix='brain_mask', sham=False, gz=True):
        '''
        Adds an internal dataset to the collection.

        Parameters:
        - modality: The modality of the dataset.
        - sub_folder: The optional sub-folder containing the NIfTI files for the rodents.
        - unavailable_labels: A list of ground truth labels that are not available for the rodents in the dataset (used for domain adaptation).
        - image_suffix: The suffix of the NIfTI files containing the images.
        - roi_mask_suffix: The suffix of the NIfTI files containing the labels.
        - brain_mask_suffix: The suffix of the NIfTI files containing the brain masks.
        - sham: Indicates whether the dataset is sham or not.
        - gz: Whether the NIfTI files are compressed or not.
        '''
        path = os.path.join(dataset_path, modality)
        ids = os.listdir(path)
        subjects = {}
        for id in ids:
            if not os.path.isdir(os.path.join(path, id)):
                continue
            subjects[id] = self.__create_subject(modality, sub_folder, unavailable_labels, image_suffix, roi_mask_suffix, brain_mask_suffix, sham, dataset_path, id, gz)
        
        self.datasets_dic[modality] = subjects
        print(f'Added {len(subjects)} subjects from {modality}')

    def get_subject(self, key):
        '''
        Returns the rodent from the dataset using the key.
        '''
        if isinstance(key, tuple):
            return self.datasets_dic[key[0]][key[1]]
        else:
            raise ValueError("Key must be a tuple with the modality and the subject id")
    
    def get_subjects_list(self):
        '''
        Concatenates all subjects from all datasets into a single list.
        '''
        subjects = []
        for modality, dataset in self.datasets_dic.items():
            for id, subject in dataset.items():
                subjects.append((modality, id))
        
        return subjects
    
    def get_subjects_dic(self):
        return self.datasets_dic
    
    def get_modalities(self):
        return self.datasets_dic.keys()
    
    def save_ids(self, path):
        '''
        Save the IDs of all rodents in the collection to a text file.
        '''
        with open(path, 'w') as f:
            for modality, dataset in self.datasets_dic.items():
                for id in dataset.keys():
                    f.write(f'{modality} {id}\n')
    

class Rodent:
    """
    Class to represent a rodent in the dataset.

    Attributes:
    - modality (str): The modality of the rodent (e.g. T1, T2, FLAIR).
    - id (str): The ID of the rodent.
    - image_path (str): The path to the NIfTI file containing the rodent's image.
    - labels_path (str): The path to the NIfTI file containing the rodent's labels.
    - brain_mask_path (str): The path to the NIfTI file containing the rodent's brain mask.
    - unavailable_labels (list): A list of ground truth labels that are not available for the rodent (used for domain adaptation).
    - sham (bool): Whether the rodent is a sham rodent or not.
    """

    def __init__(self, modality, id, image_path, labels_path, brain_mask_path=None, unavailable_labels=None, sham=False):
        self.modality = modality
        self.id = id
        self.path = image_path
        self.unavailable_labels = unavailable_labels
        self.sham = sham

        # Load the NIfTI files
        self.image = self.load_nifti(image_path)
        self.labels = self.load_nifti(labels_path)
        self.brain_mask = self.load_nifti(brain_mask_path) if brain_mask_path else None

    def load_nifti(self, file_path):
        try:
            return nib.load(file_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found for {self.id}: {file_path}")

    def __str__(self):
        return f'{self.modality} - {self.id}'

    def get_images(self):
        return self.image, self.labels, self.brain_mask
    
    def get_modality(self):
        return self.modality
    
    def get_image(self):
        return self.image
    
    def get_labels(self):
        return self.labels
    
    def get_brain_mask(self):
        return self.brain_mask
    
    def get_unavailable_labels(self):
        return self.unavailable_labels
    
    def get_sham(self):
        return self.sham
    
    def get_path(self):
        return self.path
    
    def get_id(self):
        return self.id
    
# Other utilities
def load_data(patient_folder, patient_id, modalities, gz=True):
    '''
    Extracts the data from the NIfTI files for the given patient ID and modality.
    
    Parameters:
    - patient_folder: The path to the folder containing the NIfTI files.
    - patient_id: The ID of the patient.
    - modalities: A list of the modalities to extract the data from.
    - gz: Whether the NIfTI files are compressed or not.
    
    Requires the following naming convention for the NIfTI files: <patient_id>_<modality>.nii.gz
    '''
    images = {}
    data = {}
    file_paths = {}
    
    for modality in modalities:
        # Construct the file path for the NIfTI file of the given modality
        if gz:
            file_path = os.path.join(patient_folder, f'{patient_id}_{modality}.nii.gz')
        else:
            file_path = os.path.join(patient_folder, f'{patient_id}_{modality}.nii')

        if os.path.exists(file_path):
            # Load the NIfTI image using nibabel
            img = nib.load(file_path)
            
            # Get the NIfTI data as a NumPy array
            img_data = img.get_fdata()
            
            # Store the data in the dictionary
            images[modality] = img
            data[modality] = img_data
            file_paths[modality] = file_path

        else:
            print(f"File not found for modality {modality}")
    
    return images, data, file_paths









