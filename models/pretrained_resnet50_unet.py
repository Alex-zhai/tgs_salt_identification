# -*- coding:utf-8 -*-
# Author : zhaijianwei
# Date : 2018/9/21 14:47

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


def conv_block_simple(prevlayer, filters, prefix, strides=(1, 1)):
    conv = layers.Conv2D(filters, (3, 3), padding="same", kernel_initializer="he_normal", strides=strides,
                         name=prefix + "_conv")(prevlayer)
    conv = layers.BatchNormalization(name=prefix + "_bn")(conv)
    conv = layers.Activation('relu', name=prefix + "_activation")(conv)
    return conv


def get_unet_resnet(input_shape=(128, 128, 3)):
    resnet_base = resnet_50(input_x_shape=input_shape)
    for l in resnet_base.layers:
        l.trainable = True

    conv1 = resnet_base.get_layer("activation").output  # 64, 64, 64
    conv2 = resnet_base.get_layer("activation_9").output  # 32, 32, 256
    conv3 = resnet_base.get_layer("activation_21").output  # 16, 16, 512
    conv4 = resnet_base.get_layer("activation_39").output  # 8, 8, 1024

    mid = resnet_base.get_layer("activation_48").output  # 4, 4, 2048

    up4 = layers.concatenate([layers.UpSampling2D()(mid), conv4], axis=-1)
    deconv4 = conv_block_simple(up4, 256, "deconv4_1")
    deconv4 = conv_block_simple(deconv4, 256, "deconv4_2")

    up3 = layers.concatenate([layers.UpSampling2D()(deconv4), conv3], axis=-1)
    deconv3 = conv_block_simple(up3, 192, "deconv3_1")
    deconv3 = conv_block_simple(deconv3, 192, "deconv3_2")

    up2 = layers.concatenate([layers.UpSampling2D()(deconv3), conv2], axis=-1)
    deconv2 = conv_block_simple(up2, 128, "deconv2_1")
    deconv2 = conv_block_simple(deconv2, 128, "deconv2_0")

    up1 = layers.concatenate([layers.UpSampling2D()(deconv2), conv1], axis=-1)
    deconv1 = conv_block_simple(up1, 64, "deconv1_1")
    deconv1 = conv_block_simple(deconv1, 64, "deconv1_0")

    up0 = layers.UpSampling2D()(deconv1)
    deconv0 = conv_block_simple(up0, 32, "deconv0_0")
    deconv0 = conv_block_simple(deconv0, 32, "deconv0_1")
    deconv0 = layers.SpatialDropout2D(0.2)(deconv0)
    out = layers.Conv2D(1, (1, 1), activation="sigmoid", name="prediction")(deconv0)
    model = models.Model(resnet_base.input, out)
    return model


if __name__ == '__main__':
    for i in range(2):
        model = get_unet_resnet((128, 128, 3))
        print(model.summary())
        K.clear_session()
