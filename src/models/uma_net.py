"""
UMA-Net: U-Net with Multi-scale Attention for Medical Image Segmentation
"""

import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate,
    BatchNormalization, Activation, Dropout, Add, Multiply, Lambda
)
from tensorflow.keras.models import Model

def conv_block(x, filters, kernel_size=3, padding='same', activation='relu',
               kernel_initializer='he_normal', use_bn=True):
    """Convolutional block with batch normalization and activation."""
    x = Conv2D(filters, kernel_size, padding=padding,
               kernel_initializer=kernel_initializer)(x)
    if use_bn:
        x = BatchNormalization()(x)
    x = Activation(activation)(x)
    return x

def attention_block(x, g, inter_channel):
    """Attention block for feature recalibration."""
    theta_x = Conv2D(inter_channel, [1, 1], strides=[1, 1])(x)
    phi_g = Conv2D(inter_channel, [1, 1], strides=[1, 1])(g)
    
    f = Activation('relu')(Add()([theta_x, phi_g]))
    psi_f = Conv2D(1, [1, 1], strides=[1, 1], activation='sigmoid')(f)
    
    return Multiply()([x, psi_f])

def res_block(x, filters, kernel_size=3, dropout=0.1, use_bn=True):
    """Residual block with skip connection."""
    # First convolution
    h = conv_block(x, filters, kernel_size, use_bn=use_bn)
    h = Dropout(dropout)(h)
    
    # Second convolution
    h = conv_block(h, filters, kernel_size, activation=None, use_bn=use_bn)
    
    # Skip connection
    if x.shape[-1] != filters:
        x = Conv2D(filters, 1, padding='same', use_bias=False)(x)
        if use_bn:
            x = BatchNormalization()(x)
    
    return Activation('relu')(Add()([x, h]))

def AC_block(x, filters, dilation_rates=[1, 6, 12, 18]):
    """Atrous Convolution block."""
    dims = tf.keras.backend.int_shape(x)
    
    # Image pooling
    pool = tf.keras.layers.GlobalAveragePooling2D(keepdims=True)(x)
    pool = Conv2D(filters, 1, padding='same', activation='relu')(pool)
    pool = tf.keras.layers.UpSampling2D(
        size=(dims[1] // pool.shape[1], dims[2] // pool.shape[2]),
        interpolation='bilinear')(pool)
    
    # 1x1 convolution
    conv1x1 = Conv2D(filters, 1, padding='same', activation='relu')(x)
    
    # Atrous convolutions with different rates
    conv3x3_1 = Conv2D(filters, 3, padding='same', dilation_rate=dilation_rates[0], activation='relu')(x)
    conv3x3_2 = Conv2D(filters, 3, padding='same', dilation_rate=dilation_rates[1], activation='relu')(x)
    conv3x3_3 = Conv2D(filters, 3, padding='same', dilation_rate=dilation_rates[2], activation='relu')(x)
    
    # Concatenate all branches
    x = Concatenate()([pool, conv1x1, conv3x3_1, conv3x3_2, conv3x3_3])
    x = Conv2D(filters, 1, padding='same')(x)
    
    return x

def UMA_Net(input_shape=(256, 256, 3), num_classes=1, dropout_rate=0.1, use_bn=True):
    """
    UMA-Net: U-Net with Multi-scale Attention for Medical Image Segmentation
    
    Args:
        input_shape: Input shape (height, width, channels)
        num_classes: Number of output classes
        dropout_rate: Dropout rate
        use_bn: Whether to use batch normalization
        
    Returns:
        A Keras model instance
    """
    inputs = Input(shape=input_shape)
    
    # Encoder
    # Block 1
    x1 = conv_block(inputs, 64, use_bn=use_bn)
    x1 = res_block(x1, 64, dropout=dropout_rate, use_bn=use_bn)
    p1 = MaxPooling2D((2, 2))(x1)
    
    # Block 2
    x2 = conv_block(p1, 128, use_bn=use_bn)
    x2 = res_block(x2, 128, dropout=dropout_rate, use_bn=use_bn)
    p2 = MaxPooling2D((2, 2))(x2)
    
    # Block 3
    x3 = conv_block(p2, 256, use_bn=use_bn)
    x3 = res_block(x3, 256, dropout=dropout_rate, use_bn=use_bn)
    p3 = MaxPooling2D((2, 2))(x3)
    
    # Block 4 (Bottleneck)
    x4 = conv_block(p3, 512, use_bn=use_bn)
    x4 = res_block(x4, 512, dropout=dropout_rate, use_bn=use_bn)
    p4 = MaxPooling2D((2, 2))(x4)
    
    # Bridge
    bridge = conv_block(p4, 1024, use_bn=use_bn)
    bridge = res_block(bridge, 1024, dropout=dropout_rate, use_bn=use_bn)
    
    # ASPP Module
    aspp = AC_block(bridge, 1024)
    
    # Decoder
    # Block 1
    up1 = UpSampling2D((2, 2), interpolation='bilinear')(aspp)
    att1 = attention_block(x4, up1, 512)
    up1 = Concatenate()([up1, att1])
    up1 = conv_block(up1, 512, use_bn=use_bn)
    up1 = res_block(up1, 512, dropout=dropout_rate, use_bn=use_bn)
    
    # Block 2
    up2 = UpSampling2D((2, 2), interpolation='bilinear')(up1)
    att2 = attention_block(x3, up2, 256)
    up2 = Concatenate()([up2, att2])
    up2 = conv_block(up2, 256, use_bn=use_bn)
    up2 = res_block(up2, 256, dropout=dropout_rate, use_bn=use_bn)
    
    # Block 3
    up3 = UpSampling2D((2, 2), interpolation='bilinear')(up2)
    att3 = attention_block(x2, up3, 128)
    up3 = Concatenate()([up3, att3])
    up3 = conv_block(up3, 128, use_bn=use_bn)
    up3 = res_block(up3, 128, dropout=dropout_rate, use_bn=use_bn)
    
    # Block 4
    up4 = UpSampling2D((2, 2), interpolation='bilinear')(up3)
    att4 = attention_block(x1, up4, 64)
    up4 = Concatenate()([up4, att4])
    up4 = conv_block(up4, 64, use_bn=use_bn)
    up4 = res_block(up4, 64, dropout=dropout_rate, use_bn=use_bn)
    
    # Output
    outputs = Conv2D(num_classes, (1, 1), activation='sigmoid')(up4)
    
    model = Model(inputs=[inputs], outputs=[outputs])
    return model
