import os
import pandas as pd
import nibabel as nib
from utils.nifti import estimate_volume
from tabulate import tabulate

def save_metrics(results, model, path=None):
    '''
    Save model metrics in a CSV file.
    Print the metrics in a table.
    '''
    headers = ['Metric', 'Value']
    data = []
    for i in range(len(results)):
        data.append([model.metrics_names[i], results[i]])

    # Print the table
    table = tabulate(data, headers, tablefmt='pretty')
    print(table)

    # Save metrics in a CSV file
    df = pd.DataFrame(data, columns=headers)
    df.to_csv(path, index=False)

def save_model_info(model, config, filters, test_ids, path=None):
    '''
    Save model summary, U-Net filters, model configuration, and test IDs in a text file.
    '''
    with open(path, 'w') as f:
        f.write('U-Net Filters: %s\n' % filters)
        f.write('\nModel Configuration:\n')
        df_config = pd.DataFrame(config.items(), columns=['Parameter', 'Value'])
        df_config.to_csv(f, index=False)
        f.write('\nTest IDs:\n')
        df_test_ids = pd.DataFrame(test_ids, columns=['ID', 'Modality'])
        df_test_ids.to_csv(f, index=False)
        f.write('\nModel Summary:\n')
        model.summary(print_fn=lambda x: f.write(x + '\n'))



def save_excel_table(modality_folder, sub_folder, save_folder, name_mapping,
                     pred_roi_name='unet_prediction.nii.gz', pred_brain_name='brain_mask.nii.gz', 
                     file_name='labels_volumes.xlsx', include_only_list=None, postfix_mode=False):
    '''
    Save the volumes of each class in an excel file, including the brain mask volume if available.

    Parameters:
    - modality_folder (str): path to the folder containing the subjects
    - sub_folder (str): name of the subfolder containing the original images
    - save_folder (str): path to the folder where the predictions are saved
    - labels_mapping (dict): mapping from the class index to the class name (e.g. {0: {'Background': 0}, 1: {'Hippocampus': 3}})
    - pred_roi_name (str): name of the prediction file
    - pred_brain_name (str): name of the brain mask file
    - file_name (str): name of the excel file
    - include_only_list (list): list of subjects to include in the analysis (skip the rest)
    '''
    
    def get_subjects_list(modality_folder):
        '''Get the list of subject directories from the modality folder.'''
        return [subject for subject in os.listdir(modality_folder) if os.path.isdir(os.path.join(save_folder, subject))]

    def load_prediction(path):
        '''Attempt to load a NIfTI file and handle the FileNotFoundError.'''
        try:
            return nib.load(path)
        except FileNotFoundError:
            return None

    def process_subject(case, volumes):
        '''Process a single subject by loading predictions and computing volumes.'''
        # Load the ROI prediction mask
        if postfix_mode:
            y_pred_path = os.path.join(save_folder, case, sub_folder, case + pred_roi_name)
            brain_mask_path = os.path.join(save_folder, case, sub_folder, case + pred_brain_name)
        else:
            y_pred_path = os.path.join(save_folder, case, sub_folder, pred_roi_name)
            brain_mask_path = os.path.join(save_folder, case, sub_folder, pred_brain_name)
            
        y_pred = load_prediction(y_pred_path)
        
        if y_pred is None:
            print(f'Prediction not found for {case}')
            return None  # Skip this subject if prediction is missing

        # Estimate ROI volumes
        volumes_dic = estimate_volume(y_pred)
        num_classes = len(name_mapping)

        for i in range(num_classes):
            volumes[name_mapping[i].get('name', f'Class_{i}')].append(volumes_dic.get(name_mapping[i].get('value', i), 0))
        

        # Load the brain mask prediction
        brain_mask = load_prediction(brain_mask_path)

        if brain_mask is None:
            print(f'Brain mask not found for {case}')
            volumes['Brain_Mask'].append(0)
        else:
            brain_volume = estimate_volume(brain_mask).get(1, 0)  # Assuming label 1 is the brain mask
            volumes['Brain_Mask'].append(brain_volume)

        return case

    def save_to_excel(volumes, valid_cases, save_folder, file_name):
        '''Save the volumes dictionary to an Excel file.'''
        df = pd.DataFrame(volumes)
        df.insert(0, 'Case', valid_cases)  # Insert 'Case' column as the first column
        df.to_excel(os.path.join(save_folder, file_name), index=False)

    # Main logic starts here
    subjects = get_subjects_list(modality_folder)

    # Initialize volumes dictionary for each class and brain mask
    num_classes = len(name_mapping)
    volumes = {name_mapping[i].get('name', f'Class_{i}'): [] for i in range(num_classes)}
    volumes['Brain_Mask'] = []  # Add a column for the brain mask volume

    valid_cases = []
    for case in subjects:
        if include_only_list is not None and case not in include_only_list:
            continue

        processed_case = process_subject(case, volumes)
        if processed_case:
            valid_cases.append(processed_case)

    # Save the volumes and cases to Excel
    save_to_excel(volumes, valid_cases, save_folder, file_name)