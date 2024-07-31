import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow.keras.models as models
import tensorflow.keras.regularizers as regularizers
from tensorflow.keras.layers import SpatialDropout3D, add, Conv3D, BatchNormalization, Activation, add, Dropout, MaxPooling3D, UpSampling3D, concatenate, multiply, AveragePooling3D, GlobalAveragePooling3D, Input, Add, Dense, Flatten, Conv3DTranspose, LayerNormalization, Lambda

# Normalization blocks
def normalization_block(x, norm_type='batch'):
    '''
    Normalization block that applies batch normalization or layer normalization.
    '''
    if norm_type == 'batch':
        x = BatchNormalization()(x)
    elif norm_type == 'layer':
        x = LayerNormalization()(x)
    return x
    
# Basic blocks
def conv_block(x, filters, kernel_size=3, padding='same', activation='relu', kernel_initializer='he_normal', transpose=False,
               strides=1, bn=True, number=1, kernel_regularizer=regularizers.l2(1e-5), bias_regularizer=regularizers.l2(1e-5), **kwargs):
    '''
    A convolutional block that applies a convolution followed by batch normalization and an activation function.
    '''
    for i in range(number):
        if transpose == True:
            x = Conv3DTranspose(filters, kernel_size=kernel_size, padding=padding, kernel_initializer=kernel_initializer, strides=strides, 
                                kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer, **kwargs)(x)
        else:
            x = Conv3D(filters, kernel_size=kernel_size, padding=padding, kernel_initializer=kernel_initializer, strides=strides, 
                          kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer, **kwargs)(x)

        if bn == True: x = normalization_block(x)
        if activation == True: x = Activation(activation)(x)
    return x

def conv_downsampling_block(x, filters, **kwargs):
    '''
    A convolutional layer used as an alternative to max pooling to halve the spatial dimensions.
    '''
    return conv_block(x, filters, kernel_size=2, strides=2, **kwargs)

def conv_upsampling_block(x, filters, **kwargs):
    '''
    A convolutional transpose layer used as an alternative to upsampling to double the spatial dimensions.
    '''
    return conv_block(x, filters, kernel_size=2, strides=2, transpose=True, **kwargs)

def bottleneck_block(x, filters, residual=True, concat=None, dropout=0.0):
    '''
    The bottleneck block of the U-Net model composed of a residual block or two convolutional blocks.
    '''
    if concat is not None:
        x = concatenate([x, concat], axis=-1)

    if residual == True:
        x = residual_block(x, filters)
    else:
        x = conv_block(x, filters, number=2)

    return x

# Residual blocks
def residual_block(res, filters, activation='relu', **kwargs):
    '''
    A residual block composed by two convolutional blocks with a skip connection.
    '''
    # Level 0
    r = conv_block(res, kernel_size=1, filters=filters, activation=None, bn=False, **kwargs)

    # Level 1    
    x = conv_block(res, kernel_size=3, filters=filters, **kwargs)
    
    # Level 2
    x = conv_block(x, kernel_size=3, filters=filters, activation=None, **kwargs)

    o = Add()([x, r])
    o = Activation(activation)(o)
    return o

# Attention blocks
def attention_block(x, s, filters):
    '''
    An attention block that highlights the most relevant features by processing the skip connection and the feature map coming from a lower layer.
    - x: input tensor coming from a the previous lower layer
    - s: input tensor coming from the skip connection
    '''
    # Match dimensions and number of filters by downsampling the skip connection
    phi_x = conv_block(x, filters, kernel_size=1, strides=1, activation=None, bn=False)
    theta_s = conv_downsampling_block(s, filters, activation=None, bn=False)

    # Add and apply sigmoid activation function
    f = Activation('relu')(add([phi_x, theta_s]))
    psi = conv_block(f, 1, kernel_size=1, strides=1, activation=None, bn=False, kernel_initializer='glorot_uniform')
    psi = Activation('sigmoid')(psi)

    # Upsample the attention coefficients to match the skip connection
    psi = UpSampling3D(size=(2, 2, 2))(psi)

    # Repeat the attention coefficients to match the number of filters in the skip connection
    psi = repeat_elements(psi, filters)

    # Multiply the attention coefficients with the skip connection
    attn_out = multiply([psi, s])

    # Final convolutional block to consolidate the attention coefficients
    attn_out = conv_block(attn_out, filters, kernel_size=1, strides=1, activation=None, bn=True)
    return attn_out

# Encoder and decoder blocks
def encoder_block(x, filters, residual=True, concat=None, max_pooling=True, dropout=0.0):
    '''
    An encoder block composed by a convolutional block followed by a downsampling layer.
    '''
    if concat is not None:
        x = concatenate([x, concat], axis=-1)

    if residual == True:
        s = residual_block(x, filters)
    else:
        s = conv_block(x, filters, number=2)

    if max_pooling == True:
        p = MaxPooling3D(pool_size=(2, 2, 2))(s)
    else:
        p = conv_downsampling_block(s, filters)
    
    p = SpatialDropout3D(dropout)(p)
    return p, s

def decoder_block(x, skip, filters, residual=True, attention=False, conv_transpose=True, dropout=0.0):
    '''
    A decoder block composed by an upsampling layer and a convolutional block that is merged with the skip connection (with or without attention).
    '''
    if attention == True:
        skip = attention_block(x, skip, filters)

    if conv_transpose == True:
        x = conv_upsampling_block(x, filters)
    else:
        x = UpSampling3D(size=(2, 2, 2))(x)
        x = conv_block(x, filters)

    x = concatenate([x, skip], axis=-1)
    
    if residual == True:
        x = residual_block(x, filters)
    else:
        x = conv_block(x, filters, number=2)

    return x

# Other blocks
def deep_supervision_block(x, output_channels, size, name):
    '''
    A deep supervision block that applies a transposed convolution with given kernel size and stride to produce a segmentation map.
    '''
    x = conv_block(x, output_channels, kernel_size=1, strides=1, activation=None, transpose=False, bn=False, name=name)
    if size != (1, 1, 1): x = UpSampling3D(size=size)(x)
    return x

def classifier_block(x, name, dropout=0.5):
    '''
    A classifier block compoesed by a global average pooling layer and two dense layers with dropout.
    '''
    x = GlobalAveragePooling3D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(dropout)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(dropout)(x)
    x = Dense(1, activation='sigmoid', name=name, dtype='float32')(x)
    return x

def multiscale_block(res, filters, size, activation='relu', **kwargs):
    '''
    Multiscale block composed by an average pooling layer followed by a residual block.
    '''
    i = AveragePooling3D(pool_size=size)(res) if size != (1, 1, 1) else res
    x = residual_block(i, filters, activation=activation, **kwargs)
    return x, i

# Utils
def repeat_elements(tensor, rep):
    '''
    A lambda layer that repeats the elements of a tensor.
    Assumes channels last format.
    '''
    return Lambda(lambda x, repnum: tf.keras.backend.repeat_elements(x, repnum, axis=-1), arguments={'repnum': rep})(tensor)

# Models
def mt_r_net_3d(in_shape, in_channels, num_classes, filters, residual=True, attention=True, max_pooling=True, with_classifier=False):
    '''
    Deep supervision multitask 3D U-Net model.
    '''
    _i = Input(shape=(in_shape[0], in_shape[1], in_shape[2], in_channels))

    m3, avg_img3 = multiscale_block(_i, filters[1], (2, 2, 2))
    m2, avg_img2 = multiscale_block(avg_img3, filters[2], (2, 2, 2))
    m1, ________ = multiscale_block(avg_img2, filters[3], (2, 2, 2))

    # Encoder
    e4, skip_1 = encoder_block(_i, filters[0], residual, max_pooling=max_pooling, dropout=0.0)
    e3, skip_2 = encoder_block(e4, filters[1], residual, concat=m3, max_pooling=max_pooling, dropout=0.1)
    e2, skip_3 = encoder_block(e3, filters[2], residual, concat=m2, max_pooling=max_pooling, dropout=0.15)
    e1, skip_4 = encoder_block(e2, filters[3], residual, concat=m1, max_pooling=max_pooling, dropout=0.2)

    # Bottom layer
    bn = bottleneck_block(e1, filters[4], residual, dropout=0)

    # Decoder
    d4 = decoder_block(bn, skip_4, filters[3], residual, attention=attention, dropout=0.0)
    d3 = decoder_block(d4, skip_3, filters[2], residual, attention=attention, dropout=0.0)
    d2 = decoder_block(d3, skip_2, filters[1], residual, attention=attention, dropout=0.0)
    d1 = decoder_block(d2, skip_1, filters[0], residual, attention=attention, dropout=0.0)

    # Deep supervision - brain mask
    b4 = deep_supervision_block(d4, 1, (8, 8, 8), 'b_output_4')
    b3 = deep_supervision_block(d3, 1, (4, 4, 4), 'b_output_3') 
    b2 = deep_supervision_block(d2, 1, (2, 2, 2), 'b_output_2') 
    b1 = deep_supervision_block(d1, 1, (1, 1, 1), 'b_output_1')
    fusion_brain = Add()([b4, b3, b2, b1])
    brain_mask = Conv3D(kernel_size=1, filters=1)(fusion_brain) 
    brain_mask = Activation('sigmoid', dtype='float32', name='brain_mask')(brain_mask)

    # Deep supervision - region segmentation
    r4 = deep_supervision_block(d4, num_classes, (8, 8, 8), 'r_output_4')
    r3 = deep_supervision_block(d3, num_classes, (4, 4, 4), 'r_output_3')
    r2 = deep_supervision_block(d2, num_classes, (2, 2, 2), 'r_output_2')
    r1 = deep_supervision_block(d1, num_classes, (1, 1, 1), 'r_output_1')
    fusion_regions = Add()([r4, r3, r2, r1])
    regions = Conv3D(kernel_size=1, filters=num_classes)(fusion_regions)
    regions = Activation('softmax', dtype='float32', name='regions')(regions)

    if with_classifier == True:
        cls = classifier_block(bn, name='classifier', dropout=0.2)
        return tf.keras.models.Model(inputs=_i, outputs=[regions, brain_mask, cls])
    else:
        return tf.keras.models.Model(inputs=_i, outputs=[regions, brain_mask])

# Baseline
def standard_unet_3d(in_shape, in_channels, num_classes, filters, residual=False, attention=False, max_pooling=True):
    '''
    Standard 3D U-Net model.
    '''
    _i = Input(shape=(in_shape[0], in_shape[1], in_shape[2], in_channels))

    # Encoder
    e4, skip_1 = encoder_block(_i, filters[0], residual, max_pooling=max_pooling, dropout=0.0)
    e3, skip_2 = encoder_block(e4, filters[1], residual, max_pooling=max_pooling, dropout=0.1)
    e2, skip_3 = encoder_block(e3, filters[2], residual, max_pooling=max_pooling, dropout=0.15)
    e1, skip_4 = encoder_block(e2, filters[3], residual, max_pooling=max_pooling, dropout=0.2)

    # Bottom layer
    bn = bottleneck_block(e1, filters[4], residual, dropout=0)

    # Decoder
    d4 = decoder_block(bn, skip_4, filters[3], residual, attention=attention, dropout=0.0)
    d3 = decoder_block(d4, skip_3, filters[2], residual, attention=attention, dropout=0.0)
    d2 = decoder_block(d3, skip_2, filters[1], residual, attention=attention, dropout=0.0)
    d1 = decoder_block(d2, skip_1, filters[0], residual, attention=attention, dropout=0.0)

    # Output - brain mask
    brain_mask = Conv3D(kernel_size=1, filters=1)(d1) 
    brain_mask = Activation('sigmoid', dtype='float32', name='brain_mask')(brain_mask)

    # Output - region segmentation
    regions = Conv3D(kernel_size=1, filters=num_classes)(d1)
    regions = Activation('softmax', dtype='float32', name='regions')(regions)

    return tf.keras.models.Model(inputs=_i, outputs=[regions, brain_mask])


# Double decoder
def dd_r_net_3d(in_shape, in_channels, num_classes, filters, residual=True, attention=True, strided=False):
    '''
    Deep supervision multitask 3D U-Net model with attention blocks.
    '''
    _i = Input(shape=(in_shape[0], in_shape[1], in_shape[2], in_channels))

    # Encoder
    e4, skip_1 = encoder_block(_i, filters[0], residual, strided=strided, dropout=0.0)
    e3, skip_2 = encoder_block(e4, filters[1], residual, strided=strided, dropout=0.1)
    e2, skip_3 = encoder_block(e3, filters[2], residual, strided=strided, dropout=0.15)
    e1, skip_4 = encoder_block(e2, filters[3], residual, strided=strided, dropout=0.2)

    # Bottom layer
    bnk = bottleneck_block(e1, filters[4], residual, dropout=0.25) # 8x8x8

    # Decoder - brain mask
    db4 = decoder_block(bnk, skip_4, filters[3], residual, attention=attention, strided=True, dropout=0.0)
    db3 = decoder_block(db4, skip_3, filters[2], residual, attention=attention, strided=True, dropout=0.0)
    db2 = decoder_block(db3, skip_2, filters[1], residual, attention=attention, strided=True, dropout=0.0)
    db1 = decoder_block(db2, skip_1, filters[0], residual, attention=attention, strided=True, dropout=0.0)

    # Deep supervision - brain mask
    b3 = deep_supervision_block(db3, 1, (4, 4, 4), 'b_output_3')
    b2 = deep_supervision_block(db2, 1, (2, 2, 2), 'b_output_2')
    b1 = deep_supervision_block(db1, 1, (1, 1, 1), 'b_output_1')

    brain_fusion = Add()([b3, b2, b1])
    brain_mask = Conv3D(kernel_size=1, filters=1)(brain_fusion)
    brain_mask = Activation('sigmoid', dtype='float32', name='brain_mask')(b1)

    # Decoder - region segmentation
    dr4 = decoder_block(bnk, skip_4, filters[3], residual, attention=attention, strided=True, dropout=0.0)
    dr3 = decoder_block(dr4, skip_3, filters[2], residual, attention=attention, strided=True, dropout=0.0)
    dr2 = decoder_block(dr3, skip_2, filters[1], residual, attention=attention, strided=True, dropout=0.0)
    dr1 = decoder_block(dr2, skip_1, filters[0], residual, attention=attention, strided=True, dropout=0.0)

    # Deep supervision - region segmentation
    r3 = deep_supervision_block(dr3, num_classes, (4, 4, 4), 'r_output_3')
    r2 = deep_supervision_block(dr2, num_classes, (2, 2, 2), 'r_output_2')
    r1 = deep_supervision_block(dr1, num_classes, (1, 1, 1), 'r_output_1')

    regions_fusion = Add()([r3, r2, r1])
    regions = Conv3D(kernel_size=1, filters=num_classes)(regions_fusion)
    regions = Activation('softmax', dtype='float32', name='regions')(r1)

    return tf.keras.models.Model(inputs=_i, outputs=[regions, brain_mask])