"""3D R-Net builders with configurable residual blocks, attention and dropout."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, TypeAlias

import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.layers import (
    Activation, Add, AveragePooling3D, BatchNormalization, Conv3D,
    Conv3DTranspose, Dense, Dropout, GlobalAveragePooling3D, Input,
    Lambda, LayerNormalization, MaxPooling3D, SpatialDropout3D,
    UpSampling3D, add, concatenate, multiply,
)


Tensor: TypeAlias = tf.Tensor
SpatialShape: TypeAlias = tuple[int | None, int | None, int | None]
SpatialSize: TypeAlias = int | tuple[int, int, int]


# Normalization blocks
def normalization_block(x: Tensor, norm_type: str = 'batch') -> Tensor:
    """
    Normalization block that applies batch normalization or layer normalization.
    """
    if norm_type == 'batch':
        x = BatchNormalization()(x)
    elif norm_type == 'layer':
        x = LayerNormalization()(x)

    return x

# Basic blocks
def conv_block(
    x: Tensor, filters: int, kernel_size: SpatialSize = 3, padding: str = 'same',
    activation: str | bool | None = 'relu', kernel_initializer: str = 'he_normal',
    transpose: bool = False, strides: SpatialSize = 1, bn: bool = True, number: int = 1,
    kernel_regularizer: regularizers.Regularizer | None = regularizers.l2(1e-5),
    bias_regularizer: regularizers.Regularizer | None = regularizers.l2(1e-5),
    **kwargs: Any,
) -> Tensor:
    """
    Repeat convolution, optional batch normalization and activation.

    ``None`` or ``False`` disables activation; ``True`` selects ReLU.
    String values select the corresponding Keras activation.
    """
    activation_name: str | None
    if activation is True:
        activation_name = 'relu'
    elif activation is False or activation is None:
        activation_name = None
    else:
        activation_name = activation

    for i in range(number):
        if transpose:
            x = Conv3DTranspose(filters, kernel_size=kernel_size, padding=padding, kernel_initializer=kernel_initializer, strides=strides,
                                kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer, **kwargs)(x)
        else:
            x = Conv3D(filters, kernel_size=kernel_size, padding=padding, kernel_initializer=kernel_initializer, strides=strides,
                          kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer, **kwargs)(x)

        if bn:
            x = normalization_block(x)

        if activation_name is not None:
            x = Activation(activation_name)(x)

    return x

def conv_downsampling_block(x: Tensor, filters: int, **kwargs: Any) -> Tensor:
    """
    A convolutional layer used as an alternative to max pooling to halve the spatial dimensions.
    """
    return conv_block(x, filters, kernel_size=2, strides=2, **kwargs)

def conv_upsampling_block(x: Tensor, filters: int, **kwargs: Any) -> Tensor:
    """
    A convolutional transpose layer used as an alternative to upsampling to double the spatial dimensions.
    """
    return conv_block(x, filters, kernel_size=2, strides=2, transpose=True, **kwargs)

def bottleneck_block(
    x: Tensor, filters: int, residual: bool = True,
    concat: Tensor | None = None, dropout: float = 0.0,
) -> Tensor:
    """
    Apply a residual block or two convolutions, optionally joining extra features.

    Apply spatial dropout after feature processing when its rate is nonzero.
    """
    if concat is not None:
        features: list[Tensor] = [x, concat]
        x = concatenate(features, axis=-1)

    if residual:
        x = residual_block(x, filters)
    else:
        x = conv_block(x, filters, number=2)

    if dropout != 0:
        x = SpatialDropout3D(dropout)(x)

    return x

# Residual blocks
def residual_block(res: Tensor, filters: int, activation: str = 'relu', **kwargs: Any) -> Tensor:
    """
    A residual block composed by two convolutional blocks with a skip connection.
    """
    # Level 0
    r = conv_block(res, kernel_size=1, filters=filters, activation=None, bn=False, **kwargs)

    # Level 1
    x = conv_block(res, kernel_size=3, filters=filters, **kwargs)

    # Level 2
    x = conv_block(x, kernel_size=3, filters=filters, activation=None, **kwargs)

    residual_features: list[Tensor] = [x, r]
    o = Add()(residual_features)
    o = Activation(activation)(o)

    return o

# Attention blocks
def attention_block(x: Tensor, s: Tensor, filters: int) -> Tensor:
    """
    An attention block that highlights the most relevant features by processing the skip connection and the feature map coming from a lower layer.
    - x: input tensor coming from a the previous lower layer
    - s: input tensor coming from the skip connection
    """
    # Match dimensions and number of filters by downsampling the skip connection
    phi_x = conv_block(x, filters, kernel_size=1, strides=1, activation=None, bn=False)
    theta_s = conv_downsampling_block(s, filters, activation=None, bn=False)

    # Add and apply sigmoid activation function
    attention_features: list[Tensor] = [phi_x, theta_s]
    combined: Tensor = add(attention_features)
    f = Activation('relu')(combined)
    psi = conv_block(f, 1, kernel_size=1, strides=1, activation=None, bn=False, kernel_initializer='glorot_uniform')
    psi = Activation('sigmoid')(psi)

    # Upsample the attention coefficients to match the skip connection
    psi = UpSampling3D(size=(2, 2, 2))(psi)

    # Repeat the attention coefficients to match the number of filters in the skip connection
    psi = repeat_elements(psi, filters)

    # Multiply the attention coefficients with the skip connection
    weighted_features: list[Tensor] = [psi, s]
    attn_out = multiply(weighted_features)

    # Final convolutional block to consolidate the attention coefficients
    attn_out = conv_block(attn_out, filters, kernel_size=1, strides=1, activation=None, bn=True)

    return attn_out

# Encoder and decoder blocks
def encoder_block(
    x: Tensor, filters: int, residual: bool = True, concat: Tensor | None = None,
    max_pooling: bool = True, dropout: float = 0.0,
) -> tuple[Tensor, Tensor]:
    """
    An encoder block composed by a convolutional block followed by a downsampling layer.
    """
    if concat is not None:
        features: list[Tensor] = [x, concat]
        x = concatenate(features, axis=-1)

    if residual:
        s = residual_block(x, filters)
    else:
        s = conv_block(x, filters, number=2)

    if max_pooling:
        p = MaxPooling3D(pool_size=(2, 2, 2))(s)
    else:
        p = conv_downsampling_block(s, filters)

    p = SpatialDropout3D(dropout)(p)

    return p, s

def decoder_block(
    x: Tensor, skip: Tensor, filters: int, residual: bool = True,
    attention: bool = False, conv_transpose: bool = True, dropout: float = 0.0,
) -> Tensor:
    """
    Upsample, join the optionally attended skip, then process the merged features.

    Apply spatial dropout after feature processing when its rate is nonzero.
    """
    if attention:
        skip = attention_block(x, skip, filters)

    if conv_transpose:
        x = conv_upsampling_block(x, filters)
    else:
        x = UpSampling3D(size=(2, 2, 2))(x)
        x = conv_block(x, filters)

    features: list[Tensor] = [x, skip]
    x = concatenate(features, axis=-1)

    if residual:
        x = residual_block(x, filters)
    else:
        x = conv_block(x, filters, number=2)

    if dropout != 0:
        x = SpatialDropout3D(dropout)(x)

    return x

# Other blocks
def deep_supervision_block(
    x: Tensor, output_channels: int, size: tuple[int, int, int], name: str,
) -> Tensor:
    """
    Project features with a 1x1x1 convolution and optionally upsample the logits.
    """
    x = conv_block(x, output_channels, kernel_size=1, strides=1, activation=None, transpose=False, bn=False, name=name)

    if size != (1, 1, 1):
        x = UpSampling3D(size=size)(x)

    return x

def classifier_block(x: Tensor, name: str, dropout: float = 0.5) -> Tensor:
    """
    A classifier block composed by a global average pooling layer and two dense layers with dropout.
    """
    x = GlobalAveragePooling3D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(dropout)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(dropout)(x)
    x = Dense(1, activation='sigmoid', name=name, dtype='float32')(x)

    return x

def multiscale_block(
    res: Tensor, filters: int, size: tuple[int, int, int], activation: str = 'relu',
    *, residual: bool = True, **kwargs: Any,
) -> tuple[Tensor, Tensor]:
    """
    Pool the input, then apply a residual block or two plain convolutions.
    """
    if size != (1, 1, 1):
        i = AveragePooling3D(pool_size=size)(res)
    else:
        i = res

    if residual:
        x = residual_block(i, filters, activation=activation, **kwargs)
    else:
        x = conv_block(i, filters, activation=activation, number=2, **kwargs)

    return x, i

# Utils
def repeat_elements(tensor: Tensor, rep: int) -> Tensor:
    """
    A lambda layer that repeats the elements of a tensor.
    Assumes channels last format.
    """
    arguments: dict[str, int] = {'repnum': rep}
    repeat_layer = Lambda(
        lambda x, repnum: tf.keras.backend.repeat_elements(x, repnum, axis=-1),
        arguments=arguments,
    )

    return repeat_layer(tensor)

def _segmentation_head(x: Tensor, channels: int, name: str, activation: str) -> Tensor:
    """Apply the historical unregularized output projection and float32 activation."""
    logits: Tensor = Conv3D(kernel_size=1, filters=channels)(x)
    output: Tensor = Activation(activation, dtype='float32', name=name)(logits)

    return output


def _deep_supervision_head(
    features: Sequence[Tensor], channels: int, prefix: str, name: str, activation: str,
    *, deep_supervision: bool = True,
) -> Tensor:
    """Fuse four decoder levels in their original coarse-to-fine order.

    Disabling fusion retains the original finest-level projection and final
    output projection. There are no separate auxiliary losses in either mode.
    """
    sizes: tuple[tuple[int, int, int], ...] = ((8, 8, 8), (4, 4, 4), (2, 2, 2), (1, 1, 1))
    levels: tuple[int, ...] = (4, 3, 2, 1)
    if not deep_supervision:
        features = (features[-1],)
        sizes = ((1, 1, 1),)
        levels = (1,)

    projections: list[Tensor] = []

    for level, feature, size in zip(levels, features, sizes):
        projection_name: str = f'{prefix}_output_{level}'
        projection: Tensor = deep_supervision_block(feature, channels, size, projection_name)
        projections.append(projection)

    fused: Tensor = projections[0]
    if deep_supervision:
        fused = Add()(projections)

    return _segmentation_head(fused, channels, name, activation)


# Models
def mt_r_net_3d(
    in_shape: SpatialShape, in_channels: int, num_classes: int, filters: Sequence[int],
    residual: bool = True, attention: bool = True, max_pooling: bool = True,
    with_classifier: bool = False, *, multiscale: bool = True,
    deep_supervision: bool = True,
) -> tf.keras.Model:
    """
    Build the multiscale, deeply supervised multitask 3D R-Net.

    Supply five filter widths; concrete spatial dimensions should be divisible
    by 16. Output order is regions, brain_mask, then optional classifier.
    The residual flag controls encoder, bottleneck, decoder and multiscale
    branches. Bottleneck and decoder dropout rates remain zero in this builder.
    Multiscale input branches and decoder-head fusion can be disabled separately;
    both default to True, preserving the original graph and weight ordering.
    """
    input_shape: tuple[int | None, ...] = (in_shape[0], in_shape[1], in_shape[2], in_channels)
    _i: Tensor = Input(shape=input_shape)
    level_1: int = filters[0]
    level_2: int = filters[1]
    level_3: int = filters[2]
    level_4: int = filters[3]
    level_5: int = filters[4]

    m3: Tensor | None = None
    m2: Tensor | None = None
    m1: Tensor | None = None
    if multiscale:
        m3, avg_img3 = multiscale_block(_i, level_2, (2, 2, 2), residual=residual)
        m2, avg_img2 = multiscale_block(avg_img3, level_3, (2, 2, 2), residual=residual)
        m1, _ = multiscale_block(avg_img2, level_4, (2, 2, 2), residual=residual)

    # Encoder
    e4, skip_1 = encoder_block(_i, level_1, residual, max_pooling=max_pooling, dropout=0.0)
    e3, skip_2 = encoder_block(e4, level_2, residual, concat=m3, max_pooling=max_pooling, dropout=0.1)
    e2, skip_3 = encoder_block(e3, level_3, residual, concat=m2, max_pooling=max_pooling, dropout=0.15)
    e1, skip_4 = encoder_block(e2, level_4, residual, concat=m1, max_pooling=max_pooling, dropout=0.2)

    # Bottom layer
    bn = bottleneck_block(e1, level_5, residual, dropout=0)

    # Decoder
    d4 = decoder_block(bn, skip_4, level_4, residual, attention=attention, dropout=0.0)
    d3 = decoder_block(d4, skip_3, level_3, residual, attention=attention, dropout=0.0)
    d2 = decoder_block(d3, skip_2, level_2, residual, attention=attention, dropout=0.0)
    d1 = decoder_block(d2, skip_1, level_1, residual, attention=attention, dropout=0.0)

    # Brain head is created first to retain automatic layer/weight names.
    decoder_features: tuple[Tensor, ...] = (d4, d3, d2, d1)
    brain_mask = _deep_supervision_head(
        decoder_features, 1, 'b', 'brain_mask', 'sigmoid',
        deep_supervision=deep_supervision,
    )
    regions = _deep_supervision_head(
        decoder_features, num_classes, 'r', 'regions', 'softmax',
        deep_supervision=deep_supervision,
    )

    outputs: list[Tensor] = [regions, brain_mask]

    if with_classifier:
        classifier = classifier_block(bn, name='classifier', dropout=0.2)
        outputs.append(classifier)

    return tf.keras.models.Model(inputs=_i, outputs=outputs)


# Baseline
def standard_unet_3d(
    in_shape: SpatialShape, in_channels: int, num_classes: int, filters: Sequence[int],
    residual: bool = False, attention: bool = False, max_pooling: bool = True,
) -> tf.keras.Model:
    """
    Build the existing two-head U-Net without multiscale input or deep supervision.

    Output order is regions, brain_mask. Plain convolution blocks use ReLU.
    Bottleneck and decoder dropout rates remain zero in this builder.
    """
    input_shape: tuple[int | None, ...] = (in_shape[0], in_shape[1], in_shape[2], in_channels)
    _i: Tensor = Input(shape=input_shape)
    level_1: int = filters[0]
    level_2: int = filters[1]
    level_3: int = filters[2]
    level_4: int = filters[3]
    level_5: int = filters[4]

    # Encoder
    e4, skip_1 = encoder_block(_i, level_1, residual, max_pooling=max_pooling, dropout=0.0)
    e3, skip_2 = encoder_block(e4, level_2, residual, max_pooling=max_pooling, dropout=0.1)
    e2, skip_3 = encoder_block(e3, level_3, residual, max_pooling=max_pooling, dropout=0.15)
    e1, skip_4 = encoder_block(e2, level_4, residual, max_pooling=max_pooling, dropout=0.2)

    # Bottom layer
    bn = bottleneck_block(e1, level_5, residual, dropout=0)

    # Decoder
    d4 = decoder_block(bn, skip_4, level_4, residual, attention=attention, dropout=0.0)
    d3 = decoder_block(d4, skip_3, level_3, residual, attention=attention, dropout=0.0)
    d2 = decoder_block(d3, skip_2, level_2, residual, attention=attention, dropout=0.0)
    d1 = decoder_block(d2, skip_1, level_1, residual, attention=attention, dropout=0.0)

    # Preserve head construction order and public output order.
    brain_mask = _segmentation_head(d1, 1, 'brain_mask', 'sigmoid')
    regions = _segmentation_head(d1, num_classes, 'regions', 'softmax')
    outputs: list[Tensor] = [regions, brain_mask]

    return tf.keras.models.Model(inputs=_i, outputs=outputs)
