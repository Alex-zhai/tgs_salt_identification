# -*- coding:utf-8 -*-
# Author : zhaijianwei
# Date : 2018/8/22 15:29

from __future__ import absolute_import, division, print_function
from tensorflow.python.keras import layers
from tensorflow.python.keras import models


def conv_block(input_tensor, filter_nums):
    x = layers.Conv2D(filter_nums, (3, 3), padding='same')(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(filter_nums, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    return x


def encoder_block(input_tensor, filter_nums):
    x = conv_block(input_tensor, filter_nums)
    x_pool = layers.MaxPooling2D((2, 2), strides=(2, 2))(x)
    return x, x_pool


def decoder_block(input_tensor, concat_tensor, filter_nums):
    x = layers.Conv2DTranspose(filter_nums, (2, 2), strides=(2, 2), padding='same')(input_tensor)
    x = layers.concatenate([concat_tensor, x], axis=-1)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(filter_nums, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(filter_nums, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    return x


def u_net(input_shape=(128, 128, 3)):
    input = layers.Input(shape=input_shape)
    encoder0, encoder0_pool = encoder_block(input, 16)
    encoder1, encoder1_pool = encoder_block(encoder0_pool, 32)
    encoder2, encoder2_pool = encoder_block(encoder1_pool, 64)
    encoder3, encoder3_pool = encoder_block(encoder2_pool, 128)
    encoder4, encoder4_pool = encoder_block(encoder3_pool, 126)
    center = conv_block(encoder4_pool, 512)
    decoder4 = decoder_block(center, encoder4, 256)
    decoder3 = decoder_block(decoder4, encoder3, 128)
    decoder2 = decoder_block(decoder3, encoder2, 64)
    decoder1 = decoder_block(decoder2, encoder1, 32)
    decoder0 = decoder_block(decoder1, encoder0, 16)
    output = layers.Conv2D(1, (1, 1), padding='same', activation='sigmoid')(decoder0)
    return models.Model(inputs=input, outputs=output)


if __name__ == '__main__':
    model = u_net()
    print(model.summary())
