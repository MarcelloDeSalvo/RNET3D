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

def save_excel_table(subjects, save_folder, labels_mapping, num_classes, pred_name='unet_prediction.nii.gz', file_name='labels_volumes.xlsx', subjects_mask=None):
    '''
    Save the volumes of each class in an excel file.

    Parameters:
    - subjects (list): list of subjects
    - save_folder (str): path to the folder where the predictions are saved
    - labels_mapping (dict): dictionary with the mapping between class index and class name
    - num_classes (int): number of classes
    - pred_name (str): name of the prediction file
    - file_name (str): name of the excel file
    - subjects_mask (list): list of subjects to consider
    '''
    
    # create a dictionary to store the volumes for each class
    volumes = {key: [] for key in labels_mapping.values()}

    valid_cases = []
    for case in subjects:
        # skip if not a directory
        if not os.path.isdir(os.path.join(save_folder, case)): 
            continue

        if subjects_mask is not None:
            if case not in subjects_mask:
                continue
        
        # load the prediction mask
        try:
            y_pred = nib.load(os.path.join(save_folder, case, 'Anat', f'{case}'+pred_name))
        except FileNotFoundError:
            print(f'Prediction not found for {case}')
            continue

        volumes_dic = estimate_volume(y_pred)
        print(volumes_dic)

        for i in range(num_classes):
            # if the class is not present in the dictionary, append 0
            if labels_mapping[i] not in volumes_dic:
                volumes[labels_mapping[i]].append(0)
            else:
                volumes[labels_mapping[i]].append(volumes_dic[labels_mapping[i]])

        valid_cases.append(case)
            
    # create a pandas dataframe
    df = pd.DataFrame(volumes)
    df.insert(0, 'Case', valid_cases)  # Insert 'Case' column as the first column
    # save the dataframe to an excel file
    df.to_excel(os.path.join(save_folder, file_name), index=False)