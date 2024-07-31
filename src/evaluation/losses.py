import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np

# LOSSES
## DICE LOSS
def diceLoss(smooth=1e-5, axis=(1,2,3)):

    def loss(target, predicted):
        # Compute intersection and union for each sample
        intersection = K.sum(predicted * target, axis=axis)
        union = K.sum(predicted, axis=axis) + K.sum(target, axis=axis)
        
        # Compute the Dice loss for each sample
        loss_per_sample = 1 - 2 * (intersection + smooth) / (union + smooth)
        
        # Compute the mean loss across all samples in the batch and classes
        return K.mean(loss_per_sample)
    
    return loss
    

def batchDiceLoss(smooth=1e-5):

    def loss(target, predicted):
        # Iterate over all classes and compute the loss
        loss = 0.0
        num_classes = K.cast(K.shape(target)[-1], 'float32')
        for i in range(num_classes):
            loss += computeScore(target[:, :, :, :, int(i)], predicted[:, :, :, :, int(i)], smooth)

        return 1 - loss / num_classes

    def computeScore(target, predicted, smooth=1e-5):
        # Flatten the predicted and target tensors to treat the batch as a big flattened image
        predicted = tf.reshape(predicted, [-1])
        target = tf.reshape(target, [-1])

        # Compute intersection and union
        intersection = K.sum(predicted * target)
        union = K.sum(predicted) + K.sum(target)

        # Compute the Dice score
        return  2 * (intersection + smooth) / (union + smooth)

    return loss


## TVERSKY LOSS
def batchTverskyLoss(alpha=0.5, smooth=1e-5):
    
    def loss(target, predicted):
        loss = 0
        num_classes = K.cast(K.shape(target)[-1], 'float32')
        for i in range(num_classes):
            # If alpha is a list, use the corresponding value for each class
            current_alpha = alpha[i] if isinstance(alpha, list) else alpha
            loss += computeScore(target[:, :, :, :, int(i)], predicted[:, :, :, :, int(i)], current_alpha, smooth)
        
        return 1 - loss / num_classes

    def computeScore(target, predicted, alpha, smooth):
        y_true_pos = K.flatten(target)
        y_pred_pos = K.flatten(predicted)
        true_pos = K.sum(y_true_pos * y_pred_pos)
        false_neg = K.sum(y_true_pos * (1 - y_pred_pos))
        false_pos = K.sum((1 - y_true_pos) * y_pred_pos)
        return (true_pos + smooth) / (true_pos + alpha * false_neg + (1 - alpha) * false_pos + smooth)
        
    return loss

def tverskyLoss(alpha=0.5, smooth=1e-5, sum_axis=(1,2,3), focal=None):
        def loss(target, predicted):
            true_positives = K.sum(predicted * target, axis=sum_axis)
            false_negatives = K.sum(target * (1 - predicted), axis=sum_axis)
            false_positives = K.sum((1 - target) * predicted, axis=sum_axis)
            tversky = (true_positives + smooth) / (true_positives + alpha * false_negatives + (1 - alpha) * false_positives + smooth)
            
            if focal is not None: return K.mean(tf.math.pow(1 - tversky, focal))
            return 1 - K.mean(tversky)
        
        return loss


## CROSS-ENTROPY BASED LOSS
def weightedBinaryCrossentropy(alpha=0.5, smooth=K.epsilon()):

    def loss(target, predicted):
        predicted = tf.clip_by_value(predicted, smooth, 1 - smooth)
        cross_entropy = (alpha * target * K.log(predicted) + (1 - alpha) * (1 - target) * K.log(1 - predicted))
        return - K.mean(cross_entropy)
    
    return loss

def focalLoss(gamma=2.0, alpha=1, smooth=K.epsilon()):

    def loss(target, predicted):
        predicted = tf.clip_by_value(predicted, smooth, 1 - smooth)
        cross_entropy = target * K.log(predicted)
        focal = - alpha * K.pow(1 - predicted, gamma) * cross_entropy
        return K.mean(focal)
    
    return loss
    
def crossEntropyLoss(smooth=K.epsilon()):
    def loss(target, predicted):
        predicted = tf.clip_by_value(predicted, smooth, 1 - smooth)
        loss = - target * K.log(predicted)
        return K.mean(loss)
    
    return loss
        

## COMBINED LOSSES
def diceCELoss(gamma=0.5, smooth=1e-5, batch_wise=False, focal_gamma=None, domain_adaptation=False):
    
        def loss(target, predicted):
            # Check if domain adaptation is enabled
            if domain_adaptation: target, predicted = ignore_unavailable_ground_truth(target, predicted)
            # Compute the Dice loss
            dice_loss = batchDiceLoss(smooth)(target,predicted) if batch_wise else diceLoss(smooth=smooth)(target,predicted)
            # Compute the crossentropy loss
            ce_loss = crossEntropyLoss(smooth)(target,predicted) if focal_gamma is None else focalLoss(focal_gamma, smooth=smooth)(target,predicted)
            
            return gamma * ce_loss + (1 - gamma) * dice_loss
        
        return loss

def tverskyCELoss(alpha=0.5, smooth=1e-5, gamma=0.5, batch_wise=False, focal_gamma=None):
    
    def loss(target, predicted):
        # Compute the Tversky loss
        tversky_loss = batchTverskyLoss(alpha, smooth)(target, predicted) if batch_wise else tverskyLoss(alpha, smooth)(target, predicted)
        
        # Compute the crossentropy loss
        ce_loss = crossEntropyLoss(smooth)(target, predicted) if focal_gamma is None else focalLoss(focal_gamma, smooth=smooth)(target, predicted)
        
        total_loss = gamma * ce_loss + (1 - gamma) * tversky_loss 
        
        return total_loss
    
    return loss

def diceBCELoss(gamma=0.5, alpha=0.5, smooth=1e-5, batch_wise=False):

    def loss(target, predicted):
        # Compute the Dice loss
        dice_loss = batchDiceLoss(smooth)(target,predicted) if batch_wise else diceLoss(smooth=smooth)(target,predicted)
        
        # Compute the binary crossentropy loss
        bce_loss = weightedBinaryCrossentropy(alpha, smooth)(target, predicted)
        
        return gamma * bce_loss + (1 - gamma) * dice_loss
    
    return loss

# UTILS
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