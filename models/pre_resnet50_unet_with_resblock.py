# -*- coding:utf-8 -*-
# Author : zhaijianwei
# Date : 2018/9/21 21:33

from __future__ import absolute_import, division, print_function
from keras import layers
from keras import models
from keras import backend as K
from keras.utils.data_utils import get_file

WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'


def identity_block(input_tensor, kernel_size, filters, stage, block):
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = layers.Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters2, kernel_size, padding='same', name=conv_name_base + '2b')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = layers.add([x, input_tensor])
    x = layers.Activation('relu')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = layers.Conv2D(filters1, (1, 1), strides=strides, name=conv_name_base + '2a')(input_tensor)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters2, kernel_size, padding='same', name=conv_name_base + '2b')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = layers.Conv2D(filters3, (1, 1), strides=strides,
                             name=conv_name_base + '1')(input_tensor)
    shortcut = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = layers.add([x, shortcut])
    x = layers.Activation('relu')(x)
    return x


def resnet_50(input_x_shape, weights='imagenet'):
    input_x = layers.Input(shape=input_x_shape)
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    x = layers.Conv2D(64, (7, 7), strides=(2, 2), padding='same', name='conv1')(input_x)
    x = layers.BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    out = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

    model = models.Model(inputs=input_x, outputs=out)

    if weights == 'imagenet':
        weights_path = get_file('resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                WEIGHTS_PATH_NO_TOP,
                                cache_subdir='models',
                                md5_hash='a268eb855778b3df3c7506639542a6af')
        model.load_weights(weights_path, by_name=True)

    return model


def BatchActivate(x):
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    return x


def convolution_block(x, filters, size, strides=(1, 1), padding='same', activation=True):
    x = layers.Conv2D(filters, size, strides=strides, padding=padding)(x)
    if activation:
        x = BatchActivate(x)
    return x


def residual_block(blockInput, num_filters=16, batch_activate=False):
    x = BatchActivate(blockInput)
    x = convolution_block(x, num_filters, (3, 3))
    x = convolution_block(x, num_filters, (3, 3), activation=False)
    x = layers.Add()([x, blockInput])
    if batch_activate:
        x = BatchActivate(x)
    return x


def create_model(input_shape=(128, 128, 3)):
    resnet_base = resnet_50(input_x_shape=input_shape)
    for l in resnet_base.layers:
        l.trainable = True

    conv1 = resnet_base.get_layer("activation").output  # 64, 64, 64
    conv2 = resnet_base.get_layer("activation_9").output  # 32, 32, 256
    conv3 = resnet_base.get_layer("activation_21").output  # 16, 16, 512
    conv4 = resnet_base.get_layer("activation_39").output  # 8, 8, 1024

    mid = resnet_base.get_layer("activation_48").output  # 4, 4, 2048
    # mid = residual_block(mid, num_filters=2048)
    # mid = residual_block(mid, num_filters=2014, batch_activate=True)

    deconv4 = layers.Conv2DTranspose(256, (3, 3), strides=(2, 2), padding="same")(mid)
    uconv4 = layers.concatenate([deconv4, conv4])
    uconv4 = layers.Dropout(0.5)(uconv4)
    uconv4 = layers.Conv2D(256, (3, 3), padding="same")(uconv4)
    uconv4 = residual_block(uconv4, 256)
    uconv4 = residual_block(uconv4, 256, batch_activate=True)

    deconv3 = layers.Conv2DTranspose(192, (3, 3), strides=(2, 2), padding="same")(uconv4)
    uconv3 = layers.concatenate([deconv3, conv3])
    uconv3 = layers.Dropout(0.5)(uconv3)
    uconv3 = layers.Conv2D(192, (3, 3), padding="same")(uconv3)
    uconv3 = residual_block(uconv3, 192)
    uconv3 = residual_block(uconv3, 192, batch_activate=True)

    deconv2 = layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding="same")(uconv3)
    uconv2 = layers.concatenate([deconv2, conv2])
    uconv2 = layers.Dropout(0.5)(uconv2)
    uconv2 = layers.Conv2D(128, (3, 3), padding="same")(uconv2)
    uconv2 = residual_block(uconv2, 128)
    uconv2 = residual_block(uconv2, 128, batch_activate=True)

    deconv1 = layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding="same")(uconv2)
    uconv1 = layers.concatenate([deconv1, conv1])
    uconv1 = layers.Dropout(0.5)(uconv1)
    uconv1 = layers.Conv2D(64, (3, 3), padding="same")(uconv1)
    uconv1 = residual_block(uconv1, 64)
    uconv1 = residual_block(uconv1, 64, batch_activate=True)

    up0 = layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), padding="same")(uconv1)
    uconv0 = residual_block(up0, 32)
    uconv0 = residual_block(uconv0, 32, batch_activate=True)
    uconv0 = layers.SpatialDropout2D(0.2)(uconv0)

    out = layers.Conv2D(1, (1, 1), activation="sigmoid", name="prediction")(uconv0)
    model = models.Model(resnet_base.input, out)
    return model


if __name__ == '__main__':
    model = create_model()
    print(model.summary())
