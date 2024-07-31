import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np

# METRICS
# DICE COEFFICIENT
def dice_coefficient(smooth=1e-5, mode='binary', average='macro', domain_adaptation=False, exclude_background=True, class_index=None):
    '''
    Dice score coefficient
    Parameters:
    - smooth: smoothing factor
    - mode: 'soft', 'squared', 'binary'
    '''
    def dice(y_true, y_pred):
  
        if mode == 'squared':
            union = K.sum(tf.square(y_pred)) + K.sum(tf.square(y_true))

        elif mode == 'soft':
            union = K.sum(y_pred) + K.sum(y_true)

        elif mode == 'binary':
            y_pred = tf.cast(tf.greater(y_pred, 0.5), dtype=tf.float32)
            y_true = tf.cast(tf.greater(y_true, 0.5), dtype=tf.float32)
            union = K.sum(y_pred) + K.sum(y_true)

        else:
            raise ValueError('Mode not recognized. Choose from: soft, squared, binary')
        
        intersection = K.sum(y_true * y_pred)
        dice_coefficient = (2. * intersection + smooth) / (union + smooth)
        return dice_coefficient
    
    return compute_metric(dice, average, domain_adaptation, exclude_background, class_index)


# VOLUME SIMILARITY
def volume_similarity_coefficient(average='macro', domain_adaptation=False, exclude_background=True):
    '''
    Volume similarity coefficient (VSC)
    source: https://link.springer.com/article/10.1186/s12880-015-0068-x
    Definition: VSC = 1 - | |V_pred| - |V_true| | / (|V_true| + |V_pred|)
    '''
    def volume_similarity(y_true, y_pred):
        y_pred = K.greater(y_pred, 0.5)
        y_pred = K.cast(y_pred, dtype='float32')
        volume_pred = K.sum(y_pred)
        volume_true = K.sum(y_true)
        absoulte_difference = K.abs(volume_pred - volume_true)
        sum_volumes = volume_true + volume_pred
        volume_coeff = 1 - (absoulte_difference / (sum_volumes + K.epsilon()))
        return volume_coeff
    
    return compute_metric(volume_similarity, average, domain_adaptation, exclude_background)


# IoU (Intersection over Union) coefficient
def iou_coefficient(average='macro', domain_adaptation=False, exclude_background=True):
    '''
    IoU (Intersection over Union) coefficient
    '''
    def iou(y_true, y_pred):
        y_pred = K.greater(y_pred, 0.5)
        y_pred = K.cast(y_pred, dtype='float32')

        # Intersection and Union
        intersection = K.sum(y_true * y_pred)
        union = K.sum(K.maximum(y_true, y_pred))
        
        # Calculate IoU (Intersection over Union)
        iou = (intersection + K.epsilon()) / (union + K.epsilon())
        return iou
    
    return compute_metric(iou, average, domain_adaptation, exclude_background)

# ACCURACY, PRECISION, SENSITIVITY, SPECIFICITY
def accuracy_coefficient(average='macro', domain_adaptation=False, exclude_background=True):
    '''
    Accuracy metric (TP + TN) / (TP + TN + FP + FN)
    '''
    def accuracy(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        true_negatives = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
        false_positives = K.sum(K.round(K.clip((1 - y_true) * y_pred, 0, 1)))
        false_negatives = K.sum(K.round(K.clip(y_true * (1 - y_pred), 0, 1)))
        accuracy = (true_positives + true_negatives) / (true_positives + true_negatives + false_positives + false_negatives + K.epsilon())
        return accuracy
       
    return compute_metric(accuracy, average=average, domain_adaptation=domain_adaptation, exclude_background=exclude_background)

def precision_coefficient(average='macro', domain_adaptation=False, exclude_background=True):
    '''
    Precision metric (TP / (TP + FP))
    '''
    def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    
    return compute_metric(precision, average, domain_adaptation, exclude_background)
        
def sensitivity_coefficient(average='macro', domain_adaptation=False, exclude_background=True):
    '''
    Sensitivity metric (TP / (TP + FN))
    '''
    def sensitivity(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        sensitivity =  true_positives / (possible_positives + K.epsilon())
        return sensitivity
    
    return compute_metric(sensitivity, average, domain_adaptation, exclude_background)

def specificity_coefficient(average='macro', domain_adaptation=False, exclude_background=True):
    '''
    Specificity metric (TN / (TN + FP))
    '''   
    def specificity(y_true, y_pred):
        true_negatives = K.sum(K.round(K.clip((1-y_true) * (1-y_pred), 0, 1)))
        possible_negatives = K.sum(K.round(K.clip(1-y_true, 0, 1)))
        specificity = true_negatives / (possible_negatives + K.epsilon())
        return specificity
   
    return compute_metric(specificity, average, domain_adaptation, exclude_background)

# UTILS
def ignore_background(y_true, y_pred):
    '''
    Ignore background class (0) for evaluation.
    '''
    # If shape is equal to 1 then skip or array is 4D then skip
    if K.int_shape(y_true)[-1] == 1 or len(K.int_shape(y_true)) == 4:
        return y_true, y_pred
    
    y_true = y_true[:, :, :, :, 1:]
    y_pred = y_pred[:, :, :, :, 1:]
    return y_true, y_pred

def ignore_unavailable_ground_truth(y_true, y_pred, ignore_value=-1):
    '''
    Ignore predictions for specific values.

    Parameters:
    - ignore_value: value to ignore and replace with 0
    '''
    
    mask = tf.equal(y_true, ignore_value)

    # convert values to 0 for y_true and y_pred
    y_true = tf.where(mask, tf.zeros_like(y_true), y_true)
    y_pred = tf.where(mask, tf.zeros_like(y_pred), y_pred)

    return y_true, y_pred

def compute_metric(metric, average='macro', domain_adaptation=False, exclude_background=False, class_index=None):
    '''
    Get the average metric over all classes

    Parameters:
    - metric: metric function
    - average: 'macro' or 'micro', macro computes the average over all classes, micro computes the metric globally
    - domain_adaptation: ignore predictions for specific values
    - exclude_background: ignore background class (0) for evaluation
    - class_index: index of the class to compute the metric
    '''

    def filter_classes(y_true, y_pred):
        if domain_adaptation == True:
            y_true, y_pred = ignore_unavailable_ground_truth(y_true, y_pred)

        if exclude_background == True:
            y_true, y_pred = ignore_background(y_true, y_pred)
    
        return y_true, y_pred
    
    def macro_average(y_true, y_pred):
        y_true, y_pred = filter_classes(y_true, y_pred)
        num_classes = K.cast(K.shape(y_true)[-1], dtype='float32')
        # Take the average of the metric for each class
        macro_average = 0.0
        for i in range(num_classes):
            class_metric = metric(y_true[:,:,:,:,int(i)], y_pred[:,:,:,:,int(i)])
            macro_average += class_metric
        macro_average /= num_classes
        return macro_average
    
    def micro_average(y_true, y_pred):
        y_true, y_pred = filter_classes(y_true, y_pred)
        micro_average = metric(y_true, y_pred)
        return micro_average
    
    def index_class(y_true, y_pred):
        y_true, y_pred = filter_classes(y_true, y_pred)
        class_metric = metric(y_true[:,:,:,:,class_index], y_pred[:,:,:,:,class_index])
        return class_metric
    
    # Print metric name during training
    if class_index is not None:
        index_class.__name__ = "class_{}_{}".format(class_index, metric.__name__)
        return index_class
    if average == 'macro':
        macro_average.__name__ = "mean_{}".format(metric.__name__)
        return macro_average
    if average == 'micro':
        micro_average.__name__ = metric.__name__
        return micro_average
    else:
        raise ValueError('Average not recognized. Choose from: macro, micro')