# -*- coding:utf-8 -*-
# Author : zhaijianwei
# Date : 2018/9/25 13:21

from tensorflow.python.keras import layers, models


def conv_block(input_x, filter_nums):
    x = layers.Conv2D(filter_nums, (3, 3), padding='same')(input_x)
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


def up_conv(input_x, filter_nums):
    x = layers.Conv2DTranspose(filter_nums, (2, 2), strides=(2, 2), padding='same')(input_x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(filter_nums, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    # x = layers.Conv2D(filter_nums, (3, 3), padding='same')(x)
    # x = layers.BatchNormalization()(x)
    # x = layers.Activation('relu')(x)
    return x


def attention_block(input_g, input_x, filter_nums):
    g1 = layers.Conv2D(filter_nums, (1, 1), padding='same')(input_g)
    g1 = layers.BatchNormalization()(g1)
    x1 = layers.Conv2D(filter_nums, (1, 1), padding='same')(input_x)
    x1 = layers.BatchNormalization()(x1)
    gx1 = layers.add([g1, x1])
    gx1 = layers.Activation('relu')(gx1)
    psi = layers.Conv2D(1, (1, 1), padding='same')(gx1)
    psi = layers.BatchNormalization()(psi)
    psi = layers.Activation('sigmoid')(psi)
    out = layers.multiply([input_x, psi])
    return out


def create_model(input_shape=(224, 224, 3)):
    input = layers.Input(shape=input_shape)
    encoder0, encoder0_pool = encoder_block(input, 16)
    encoder1, encoder1_pool = encoder_block(encoder0_pool, 32)
    encoder2, encoder2_pool = encoder_block(encoder1_pool, 64)
    encoder3, encoder3_pool = encoder_block(encoder2_pool, 128)
    encoder4, encoder4_pool = encoder_block(encoder3_pool, 256)

    center = conv_block(encoder4_pool, 512)

    decoder4 = up_conv(center, 256)
    att4 = attention_block(decoder4, encoder4, 256)
    decoder4 = layers.concatenate([decoder4, att4])
    decoder4 = conv_block(decoder4, 256)

    decoder3 = up_conv(decoder4, 128)
    att3 = attention_block(decoder3, encoder3, 128)
    decoder3 = layers.concatenate([decoder3, att3])
    decoder3 = conv_block(decoder3, 128)

    decoder2 = up_conv(decoder3, 64)
    att2 = attention_block(decoder2, encoder2, 64)
    decoder2 = layers.concatenate([decoder2, att2])
    decoder2 = conv_block(decoder2, 64)

    decoder1 = up_conv(decoder2, 32)
    att1 = attention_block(decoder1, encoder1, 32)
    decoder1 = layers.concatenate([decoder1, att1])
    decoder1 = conv_block(decoder1, 32)

    decoder0 = up_conv(decoder1, 16)
    att0 = attention_block(decoder0, encoder0, 16)
    decoder0 = layers.concatenate([decoder0, att0])
    decoder0 = conv_block(decoder0, 16)

    output = layers.Conv2D(1, (1, 1), padding='same', activation='sigmoid')(decoder0)
    return models.Model(inputs=input, outputs=output)


if __name__ == '__main__':
    model = create_model()
    print(model.summary())
