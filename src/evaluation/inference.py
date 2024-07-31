from abc import abstractmethod
import os
import math
import numpy as np
import pandas as pd
import seaborn as sns
import nibabel as nib
import tensorflow as tf
from tabulate import tabulate
import matplotlib.pyplot as plt
from utils.logger import PrintLogger
from evaluation.metrics import *


class RandomCroppingPrediction:
    '''
    Class that performs inference on an input image that was randomly cropped into patches during training.\n
    It will cover the entire input volume by making a last prediction near the edges if a dimension is not a multiple of the stride.\n
    The final mask is made by taking the mean of the predictions of the overlapping patches.\n
    Expected model input shape is (B, H, W, D, C) where B is the batch size, H is the height, W is the width, D is the depth and C is the number of channels.\n
    If num_classes is 1, the output will be a binary mask, otherwise argmax will be used to get the class with the highest probability.

    Parameters:
    - model (tensorflow.python.keras.engine.functional.Functional): Model to perform inference with.
    - patch_size (tuple): Size of the patches to be used for inference.
    - num_classes (int): Number of classes in the output mask.
    - stride (int): Stride to be used for the random cropping.
    - threshold (float): Threshold value for the binary mask.
    '''

    def __init__(self, model, patch_size, num_classes, stride=16, threshold=0.5):
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.stride = stride
        self.threshold = threshold
        self.model = model

    def fix_dimension(self, depth, height, width):
        '''
        Fix the dimensions by checking if they are smaller than the patch size.
        
        Returns:
        - the patch size if the dimensions are smaller, otherwise the original dimensions.
        '''
        if depth < self.patch_size[0]:
            depth = self.patch_size[0]
        if height < self.patch_size[1]:
            height = self.patch_size[1]
        if width < self.patch_size[2]:
            width = self.patch_size[2]
        return depth, height, width
    
    def padding(self, img_data, fixed_depth, fixed_height, fixed_width):
        '''
        Pad the image to reach the desired dimensions.
        '''
        diff_x = fixed_depth - img_data.shape[0]
        diff_y = fixed_height - img_data.shape[1]
        diff_z = fixed_width - img_data.shape[2]

        if diff_x > 0:
            img_data = np.pad(img_data, ((0, diff_x), (0, 0), (0, 0)), mode='constant')
        if diff_y > 0:
            img_data = np.pad(img_data, ((0, 0), (0, diff_y), (0, 0)), mode='constant')
        if diff_z > 0:
            img_data = np.pad(img_data, ((0, 0), (0, 0), (0, diff_z)), mode='constant')

        return img_data

    def random_cropping_inference(self, input_img, with_brain_mask=False, with_classifier=False):
        '''
        Perform inference on an input image that was randomly cropped into patches during training.\n
        It will cover the entire input volume by making a last prediction near the edges if a dimension is not a multiple of the stride.
        The final mask is made by taking the mean of the predictions of the overlapping patches.

        Parameters:
        - input_img (nibabel.nifti1.Nifti1Image): Input image to perform inference on.
        - with_brain_mask (bool): Whether to return a brain mask alongside the prediction mask.
        - with_classifier (bool): Whether to return the binary output of the classifier.

        Returns:
        - results (dict): Dictionary containing the final prediction mask, brain mask and classifier output if requested.
        - results['roi'] (np.ndarray): Final prediction mask.
        - results['brain_mask'] (np.ndarray): Final brain mask.
        - results['classifier'] (bool): Classifier output.
        '''

        results = {}
        
        input_volume, original_shape, padded_shape = self.preprocess_input(input_img)
        
        predicted_patches, count_map = self.initialize_prediction_arrays(padded_shape)

        if with_brain_mask:
            predicted_brain_patches, brain_count_map = self.initialize_brain_arrays(padded_shape)

        if with_classifier:
            predicted_classifier_patches, classifier_count_map = self.initialize_classifier_arrays()
        
        steps_d, steps_h, steps_w = self.calculate_steps(padded_shape)
        
        for d in range(0, steps_d *  self.stride, self.stride):
            for h in range(0, steps_h *  self.stride, self.stride):
                for w in range(0, steps_w *  self.stride, self.stride):
                    start, end = self.calculate_patch_bounds(d, h, w, padded_shape)
                    patch = self.extract_patch(input_volume, start, end)
                    predicted_patch, brain_patch, predicted_aux = self.perform_inference(patch, with_brain_mask, with_classifier)

                    self.update_prediction_arrays(predicted_patches, count_map, predicted_patch, start, end)

                    if with_brain_mask:
                        self.update_brain_arrays(predicted_brain_patches, brain_count_map, brain_patch, start, end)

                    if with_classifier:
                        self.update_classifier_arrays(predicted_classifier_patches, classifier_count_map, predicted_aux)

        if with_brain_mask:
            final_brain_mask = self.calculate_final_brain_mask(predicted_brain_patches, brain_count_map, original_shape)
            results['brain_mask'] = final_brain_mask

        if with_classifier:
            final_classifier_output = self.calculate_final_classifier_output(predicted_classifier_patches, classifier_count_map)
            results['classifier'] = final_classifier_output

        final_prediction = self.calculate_final_prediction(predicted_patches, count_map, original_shape)
        results['roi'] = final_prediction

        return results

    def preprocess_input(self, input_img):
        '''
        Preprocess the input image by padding it to reach the desired dimensions.
        '''
        input_volume = input_img.get_fdata() if isinstance(input_img, nib.Nifti1Image) else input_img
        original_shape = input_volume.shape
        padded_shape = self.fix_dimension(*original_shape)
        input_volume = self.padding(input_volume, *padded_shape)
        return input_volume, original_shape, padded_shape

    def initialize_prediction_arrays(self, padded_shape):
        '''
        Initialize the arrays to store the predicted patches and the count map.
        '''
        predicted_patches = np.zeros((*padded_shape, self.num_classes))
        count_map = np.zeros((*padded_shape, self.num_classes))
        return predicted_patches, count_map

    def initialize_brain_arrays(self, padded_shape):
        '''
        Initialize the arrays to store the predicted brain patches and the brain count map.
        '''
        predicted_brain_patches = np.zeros((*padded_shape, 1))
        brain_count_map = np.zeros((*padded_shape, 1))
        return predicted_brain_patches, brain_count_map
    
    def initialize_classifier_arrays(self):
        '''
        Initialize the arrays to store the predicted classifier patches and the count map.
        '''
        predicted_classifier_patches = np.zeros((1))
        count_map = np.zeros((1))
        return predicted_classifier_patches, count_map
    
    def update_prediction_arrays(self, predicted_patches, count_map, predicted_patch, start, end):
        '''
        Update the predicted patches and the count map with the current patch.
        '''
        for i in range(self.num_classes):
            predicted_patches[start[0]:end[0], start[1]:end[1], start[2]:end[2], i] += predicted_patch[..., i]
            count_map[start[0]:end[0], start[1]:end[1], start[2]:end[2], i] += 1

    def update_brain_arrays(self, predicted_brain_patches, brain_count_map, brain_patch, start, end):
        '''
        Update the predicted brain patches and the brain count map with the current patch.
        '''
        predicted_brain_patches[start[0]:end[0], start[1]:end[1], start[2]:end[2], 0] += brain_patch[..., 0]
        brain_count_map[start[0]:end[0], start[1]:end[1], start[2]:end[2], 0] += 1

    def update_classifier_arrays(self, predicted_classifier_patches, count_map, predicted_patch):
        '''
        Update the predicted classifier patches and the count map with the current patch.
        '''
        predicted_classifier_patches += 1 if predicted_patch > 0.5 else 0
        count_map += 1

    def calculate_steps(self, padded_shape):
        '''
        Calculate the number of steps to cover the entire volume.
        '''
        depth, height, width = padded_shape
        steps_d = max(1, (depth - self.patch_size[0]) // self.stride +1)
        steps_h = max(1, (height - self.patch_size[1]) // self.stride +1)
        steps_w = max(1, (width - self.patch_size[2]) // self.stride +1)

        # if steps * stride < volume size, add one more step
        if steps_d * self.stride < depth: steps_d += 1
        if steps_h * self.stride < height: steps_h += 1
        if steps_w * self.stride < width: steps_w += 1

        return steps_d, steps_h, steps_w

    def calculate_patch_bounds(self, d, h, w, padded_shape):
        '''
        Calculate the start and end indices of the current patch.
        '''
        depth, height, width = padded_shape
        if d + self.patch_size[0] > depth:
            d = depth - self.patch_size[0]
        if h + self.patch_size[1] > height:
            h = height - self.patch_size[1]
        if w + self.patch_size[2] > width:
            w = width - self.patch_size[2]
        
        ranges_d = (d, d + self.patch_size[0])
        ranges_h = (h, h + self.patch_size[1])
        ranges_w = (w, w + self.patch_size[2])
        
        return (ranges_d[0], ranges_h[0], ranges_w[0]), (ranges_d[1], ranges_h[1], ranges_w[1])

    def extract_patch(self, input_volume, start, end):
        '''
        Extract the current patch from the input volume.
        '''
        return input_volume[start[0]:end[0], start[1]:end[1], start[2]:end[2]]

    def perform_inference(self, patch, with_brain_mask, with_classifier_output):
        '''
        Perform inference on the current patch.
        Extract the predicted patch, brain mask and classifier output if requested.
        '''
        fixed_patch = np.expand_dims(np.expand_dims(patch, axis=-1), axis=0)
        predictions = self.model.predict(fixed_patch, verbose=False)
        predicted_patch = predictions[0][0] # (1, H, W, D, C) -> (H, W, D, C)
        brain_patch = predictions[1][0] if with_brain_mask else None
        aux_patch = predictions[2][0] if with_classifier_output else None
        return predicted_patch, brain_patch, aux_patch

    def calculate_final_prediction(self, predicted_patches, count_map, original_shape):
        '''
        Calculate the final prediction mask by taking the mean of the predictions of the overlapping patches.
        '''
        count_map[count_map == 0] = 1
        predicted_patches /= count_map
        if self.num_classes == 1:
            final_prediction = (predicted_patches >= self.threshold).astype(np.float64)[..., 0]
        else:
            final_prediction = np.argmax(predicted_patches, axis=-1).astype(np.float64)
        return final_prediction[:original_shape[0], :original_shape[1], :original_shape[2]]

    def calculate_final_brain_mask(self, predicted_brain_patches, brain_count_map, original_shape):
        '''
        Calculate the final brain mask by taking the mean of the predictions of the overlapping patches.
        '''
        brain_count_map[brain_count_map == 0] = 1
        predicted_brain_patches /= brain_count_map
        final_brain_mask = (predicted_brain_patches >= self.threshold).astype(np.float64)[..., 0]
        return final_brain_mask[:original_shape[0], :original_shape[1], :original_shape[2]]
    
    def calculate_final_classifier_output(self, predicted_classifier_patches, count_map):
        '''
        Calculate the final classifier output by taking the mean of the predictions of the overlapping patches.
        '''
        print('predicted_classifier_patches: ', predicted_classifier_patches, ' - count_map: ', count_map)
        return True if predicted_classifier_patches/count_map > 0.5 else False 
    
    def set_threshold(self, threshold):
        self.threshold = threshold

    def set_stride(self, stride):
        self.stride = stride


class FullVolumeEvaluation:

    def __init__(self, model, test_ids, config, loader, strides, preprocessor, brain_mask_postprocess_callback=None, roi_postprocess_callback=None,
                path=None, verbose=False, threshold=0.5):
        self.patch_size = config['input_shape']
        self.data_path = config['dataset_path']
        self.num_classes = config['num_classes']
        self.strides = strides
        self.threshold = threshold
        self.model = model
        self.test_ids = test_ids
        self.loader = loader
        self.modalities = loader.get_modalities()
        self.preprocessor = preprocessor
        self.results_path = path
        self.verbose = verbose
        self.brain_mask_postprocess_callback = brain_mask_postprocess_callback
        self.roi_postprocess_callback = roi_postprocess_callback
        self.logger = PrintLogger(os.path.join(self.results_path, 'evaluation.log'), verbose=verbose) if self.results_path is not None else None
        self.metrics = [dice_coefficient(average='micro'), precision_coefficient(average='micro'), specificity_coefficient(average='micro'), 
                        sensitivity_coefficient(average='micro'), iou_coefficient(average='micro'), volume_similarity_coefficient(average='micro')]


    def load_subject(self, subject_key):
        return self.loader.get_subject(subject_key)
    
    def load_data(self, subject):        
        # Extract images and masks
        image, roi_ground_truth, brain_mask_ground_truth = subject.get_images()
        return image, roi_ground_truth, brain_mask_ground_truth
    
    def evaluate(self, with_brain_mask=True, evaluate_brain_mask=False):
        '''
        Evaluate the model using random cropping inference.\n
        The evaluation is performed for each stride and each modality.\n
        Return a DataFrame with the metrics for each modality, subject and class.
        '''
        # Adding new parameters to the class
        self.evaluate_brain_mask = evaluate_brain_mask
        self.with_brain_mask = with_brain_mask

        croppingInference = RandomCroppingPrediction(self.model, self.patch_size, self.num_classes, self.strides, self.threshold)
        unique_test_ids, unique_modality_test_ids = self.get_unique_test_ids()
        self.test_ids = unique_test_ids

        stride_metrics = []
        for stride in self.strides:
            self.logger.write(f'Evaluation for stride {stride}')
            dataset_metrics = self.evaluate_dataset(stride, unique_modality_test_ids, croppingInference)
            stride_metrics.append(dataset_metrics)

        # Aggregate each set's metrics dataframe into a single dataframe
        stride_metrics = pd.concat(stride_metrics, axis=0)
        # Order by subject
        stride_metrics.sort_index(inplace=True)

        self.logger.write(tabulate(stride_metrics, headers='keys', tablefmt='psql'))

        # Save the results in a excel file
        if self.results_path is not None:
            stride_metrics.to_excel(os.path.join(self.results_path, 'evaluation_results.xlsx'))

        # Plot all metrics
        final_df = self.plot_all_metrics(stride_metrics, os.path.join(self.results_path, 'metrics_per_stride_and_modality.png'))
        self.logger.close()
        return final_df


    def evaluate_dataset(self, stride, unique_modality_test_ids, croppingInference):
        '''
        Evaluate the dataset using random cropping inference.\n
        Returns a DataFrame with the metrics for each modality, subject and class.
        '''
        croppingInference.set_stride(stride)
        
        # Declare dataset dictionary to store 'stride', 'modality' and 'subject' metrics
        dataset_metrics = []
        for mod_idx, subjects in enumerate(unique_modality_test_ids):
            # get current modality from dictionary converted to list
            cur_modality = list(self.modalities)[mod_idx]

            if len(subjects) == 0:
                self.logger.write(f'No subjects for modality {cur_modality}')
                continue

            self.logger.write(f'Evaluation for modality {cur_modality}')
            subjects_metrics = []
            for key in subjects:
                subject = self.load_subject(key)
                self.logger.write(f'Evaluation for subject {subject.get_id()}')
                subject_metrics_dataframe = self.compute_metrics_for_subject(subject, croppingInference)
                subjects_metrics.append(subject_metrics_dataframe)
            
            if len(subjects_metrics) > 1:
                # Concatenate all the dataframes
                subjects_metrics = pd.concat(subjects_metrics, axis=0)
            else:
                # Create a DataFrame from the single element in the list
                subjects_metrics = subjects_metrics[0]
            
            # Set the first column name as 'Metric'
            subjects_metrics.index.name = 'Metric'

            # Plot the dice values for each class
            for metric in self.metrics:
                # Extract rows with name equal to the current metric
                sub_series = subjects_metrics.loc[metric.__name__]
               
                # If type not DataFrame, convert to DataFrame
                if not isinstance(sub_series, pd.DataFrame):
                    sub_series = pd.DataFrame(sub_series).T
                
                self.plot_subjects_metric_per_class(sub_series, metric.__name__, f'_stride_{stride}_modality_{cur_modality}')

            # Insert the stride and modality columns to the left of the DataFrame
            subjects_metrics.insert(0, 'Stride', stride)
            subjects_metrics.insert(1, 'Modality', cur_modality)

            # Move the 'Subject' column to the left of the DataFrame
            subjects_metrics = subjects_metrics[['Subject', 'Stride', 'Modality'] + [col for col in subjects_metrics.columns if col not in ['Subject', 'Stride', 'Modality']]]
            # Make subject the index and metric a column
            subjects_metrics.reset_index(inplace=True)
            subjects_metrics.set_index('Subject', inplace=True)

            # Append the metrics for each modality to the dataset_metrics
            dataset_metrics.append(subjects_metrics)

        # Aggregate each modality's metrics dataframe into a single dataframe
        dataset_metrics = pd.concat(dataset_metrics, axis=0)
        self.logger.write(tabulate(dataset_metrics, headers='keys', tablefmt='psql'))

        # Return the metrics in a DataFrame
        return dataset_metrics

    def compute_metrics_for_subject(self, subject, croppingInference):
        '''
        Compute the metrics for a single subject.
        Returns a DataFrame with the metrics for each class.
        '''
        if self.evaluate_brain_mask:
            image, seg_gt, mask_gt = self.load_data(subject)
            processed_image, [preprocessed_gt, preprocessed_mask] = self.preprocess_data(image, [seg_gt, mask_gt])  
        else:
            image, seg_gt, _ = self.load_data(subject)
            processed_image, preprocessed_gt = self.preprocess_data(image, seg_gt)

        results = croppingInference.random_cropping_inference(processed_image, self.with_brain_mask)
        fv_brain_mask = results['brain_mask'] if self.with_brain_mask else None
        fv_roi_mask = results['roi']
        
        # Perform inference
        if self.evaluate_brain_mask:
            if self.brain_mask_postprocess_callback is not None:
                prediction = self.brain_mask_postprocess_callback(fv_brain_mask)
            prep_ground_truth = preprocessed_mask
            self.num_classes = 1
        else:
            # post process the roi mask only if the brain mask is present
            if self.roi_postprocess_callback is not None:
                prediction = self.roi_postprocess_callback(fv_roi_mask)
                self.num_classes = 5
            prep_ground_truth = preprocessed_gt

        # Prepare predictions for metric computation
        final_prediction = np.expand_dims(prediction, axis=(0, -1))
        if self.num_classes > 1:
            final_prediction = tf.keras.utils.to_categorical(final_prediction, self.num_classes)
        else:
            final_prediction = (final_prediction >= self.threshold).astype(np.float32)

        # Prepare ground truth data for metric computation
        prep_gt_data = prep_ground_truth.get_fdata()
        prep_gt_data = np.expand_dims(prep_gt_data, axis=0)

        if self.num_classes > 1:
            prep_gt_data = tf.keras.utils.to_categorical(prep_gt_data, self.num_classes)
        else:
            prep_gt_data = np.expand_dims(prep_gt_data, axis=-1)

        # Ensure correct data types
        final_prediction = final_prediction.astype(np.float32)
        prep_gt_data = prep_gt_data.astype(np.float32)

        # Compute and return metrics
        class_df = self.compute_metrics_per_class(prep_gt_data, final_prediction, subject.get_id())
        return class_df

    def compute_metrics_per_class(self, preprocessed_ground_truth, final_prediction, id):
        '''
        For each class of the prediction, compute every metric in self.metrics and return the results in a dataframe.
        '''

        subject_metrics = { 'Subject': id }

        for class_idx in range(self.num_classes):
            class_gt = preprocessed_ground_truth[..., class_idx]
            class_pred = final_prediction[..., class_idx]

            # Compute the metrics for the current class and store them in the dictionary
            metrics = {
                metric.__name__: metric(class_gt, class_pred).numpy() for metric in self.metrics
            }

            subject_metrics[f'Class: {class_idx}'] = metrics

        # Return the metrics in a DataFrame
        return pd.DataFrame(subject_metrics)
        

    def preprocess_data(self, image, gt):
        processed_image, processed_gt = self.preprocessor.preprocess(image, gt)
        return processed_image, processed_gt

    def get_unique_test_ids(self):
        unique_test_ids = []
        unique_modality_test_ids = [[] for _ in range(len(self.modalities))]

        for id in self.test_ids:
            if id not in unique_test_ids:
                unique_test_ids.append(id)
                for index, modality in enumerate(self.modalities):
                    if id[0] == modality:
                        unique_modality_test_ids[index].append(id)

        for index, modality in enumerate(self.modalities):
            self.logger.write(f'Unique test ids for modality {modality}:\n - {unique_modality_test_ids[index]}')

        return unique_test_ids, unique_modality_test_ids
    
    def plot_subjects_metric_per_class(self, subjects_df, metric, file_name):
        '''
        subjects_df: Dataframe containing the metrics for each class for each subject.
        metric: the metric to plot.
        file_name: the name of the file to save the plot.

        Dataframe example:
        ```
        | Metric  | Subject  | Class 0  | Class 1  | Class 2  |
        |---------|----------|----------|----------|----------|
        |  Dice   | Subject 1| 0.8      | 0.7      | 0.6      |
        |  Dice   | Subject 2| 0.9      | 0.8      | 0.7      |
        |  Dice   | Subject 3| 0.7      | 0.6      | 0.5      |

        '''
        # Write the DataFrame to the log
        self.logger.write(f'{metric} per subject per class')
        self.logger.write(tabulate(subjects_df, headers='keys', tablefmt='psql'))

        # Melting the DataFrame to convert it to long format for plotting
        df = subjects_df.melt(id_vars='Subject', var_name='Class', value_name=metric)

        # Create subplots
        fig, axes = plt.subplots(1, 2, figsize=(15, 7))

        # Plot bar plot with vertical x-axis labels
        sns.pointplot(y=metric, x='Subject', data=df, ax=axes[0], palette='deep', hue='Class')
        axes[0].set_title('Per class ' + metric)
        axes[0].set_ylabel(metric)
        axes[0].set_xlabel('Subject')
        axes[0].tick_params(axis='x', rotation=90)

        # Make legend size smaller
        axes[0].legend(loc='lower left', title='Class', fontsize='small')

        # Plot box plot
        sns.boxplot(y=metric, hue='Class', data=df, ax=axes[1], palette='deep', legend=False)
        axes[1].set_title('Boxplot ' + metric)
        axes[1].set_xlabel('Class')
        axes[1].set_ylabel(metric)
        axes[1].tick_params(axis='x', rotation=90)
        
        # Title
        fig.suptitle(metric + '_' + file_name)

        plt.tight_layout()

        # Save the combined figure
        if self.results_path is not None:
            # if none create the directory 'metrics per class'
            save_path = os.path.join(self.results_path, 'metrics per class')
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            plt.savefig(os.path.join(save_path, metric + '_' + file_name + '.png'))
            # Save excel
            subjects_df.to_excel(os.path.join(save_path, metric + '_' + file_name + '.xlsx'))

        plt.close()

    def plot_all_metrics(self, all_metrics_data, output_file):
        '''
        all_metrics_data: a dataframe containing the metrics for each class for each subject for each modality for each stride.
        output_file: filename to save the plot.
        '''

        all_metrics_data.to_excel(os.path.join(self.results_path, 'all_metrics_data.xlsx'))

        # Calculate the number of metrics
        num_metrics = len(self.metrics)

        # Calculate the number of rows
        num_rows = math.ceil(num_metrics / 3)

        # Create a larger figure with subplots
        fig, axes = plt.subplots(num_rows, 3, figsize=(15, 5*num_rows))

        # Flatten axes if needed
        axes = axes.flatten()

        final_df = pd.DataFrame()
        sub_df = []
        # Iterate over each metric and corresponding data
        for i, metric in enumerate(all_metrics_data['Metric'].unique()):
            # Extract the rows with the current metric
            metric_subset = all_metrics_data.loc[all_metrics_data['Metric'] == metric]

            # order subjects, stride, modality
            metric_subset.sort_index(inplace=True)
            self.logger.write(tabulate(metric_subset, headers='keys', tablefmt='psql'))
            sub_df.append(metric_subset)
            # Plot the boxplot for the current metric on its corresponding subplot
            self.plot_results(metric_subset, metric, axes[i])

        # Concatenate all the dataframes
        final_df = pd.concat(sub_df, axis=0)
        self.logger.write(tabulate(final_df, headers='keys', tablefmt='psql'))
        # Adjust layout and save the plot
        plt.tight_layout()
        plt.savefig(output_file)
        plt.show()

        # Save the final dataframe
        if self.results_path is not None:
            final_df.to_excel(os.path.join(self.results_path, 'final_metrics.xlsx'))

        return final_df

    def plot_results(self, stride_average_dices_df, metric, ax):
        '''
        stride_average_dices_df: DataFrame containing the average dice values for each set for each modality for each stride.
        
        Parameters:
        metric: the metric to plot.
        stride_average_dices_df: a DataFrame containing the average dice values for each set for each modality for each stride.
        ax: subplot axes to plot the boxplot.

        Dataframe example:
        ```
        | Subject  | Stride  | Modality  | Metric  | Class: 0 | Class: 1 | Class: 2 |
        |----------|---------|-----------|---------|----------|----------|----------|
        | Subject 1| 16      | T1        | Dice    | 0.8      | 0.7      | 0.6      |
        | Subject 2| 16      | T1        | Dice    | 0.9      | 0.8      | 0.7      |
        | Subject 3| 16      | T1        | Dice    | 0.7      | 0.6      | 0.5      |
        ```
        '''

        # Calculate the index of the first class column
        first_class_index = stride_average_dices_df.columns.get_loc('Class: 0')

        # Condense all the Class columns into a single column with the mean
        stride_average_dices_df['Mean'] = stride_average_dices_df.iloc[:, first_class_index:].mean(axis=1)

        # Reset index to make 'Subject' a regular column
        stride_average_dices_df.reset_index(inplace=True)

        # Melt DataFrame to have a tidy format for plotting
        melted_df = pd.melt(stride_average_dices_df, id_vars=['Subject', 'Stride', 'Modality', 'Metric'], value_vars=['Mean'], var_name='Class', value_name='Means')

        # Plot using seaborn boxplot
        sns.boxplot(x='Stride', y='Means', hue='Modality', data=melted_df, ax=ax, palette='deep')

        # # Overlay the scatterplot
        sns.stripplot(x='Stride', y='Means', hue='Modality', data=melted_df, dodge=True, jitter=True, color='black', ax=ax, alpha=0.5)

        # Ensure the legend is not duplicated
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[:len(set(labels))], labels[:len(set(labels))],fontsize='small')

        ax.set_title(f'{metric} per Stride')
        ax.set_xlabel('Stride')
        ax.set_ylabel(metric)

