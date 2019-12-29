# -*- coding:utf-8 -*-
# Author : zhaijianwei
# Date : 2018/9/25 14:19


from __future__ import absolute_import, division, print_function
from tensorflow.python.keras import layers
from tensorflow.python.keras import models


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


# Build model
def build_model(input_shape=(128, 128, 3), start_neurons=16, DropoutRatio=0.5):
    # 128 -> 64
    input_layer = layers.Input(shape=input_shape)
    conv1 = layers.Conv2D(start_neurons * 1, (3, 3), padding="same")(input_layer)
    conv1 = residual_block(conv1, start_neurons * 1)
    conv1 = residual_block(conv1, start_neurons * 1, True)
    pool1 = layers.MaxPooling2D((2, 2))(conv1)
    pool1 = layers.Dropout(DropoutRatio / 2)(pool1)

    # 64 -> 32
    conv2 = layers.Conv2D(start_neurons * 2, (3, 3), padding="same")(pool1)
    conv2 = residual_block(conv2, start_neurons * 2)
    conv2 = residual_block(conv2, start_neurons * 2, True)
    pool2 = layers.MaxPooling2D((2, 2))(conv2)
    pool2 = layers.Dropout(DropoutRatio)(pool2)

    # 32 -> 16
    conv3 = layers.Conv2D(start_neurons * 4, (3, 3), padding="same")(pool2)
    conv3 = residual_block(conv3, start_neurons * 4)
    conv3 = residual_block(conv3, start_neurons * 4, True)
    pool3 = layers.MaxPooling2D((2, 2))(conv3)
    pool3 = layers.Dropout(DropoutRatio)(pool3)

    # 16 -> 8
    conv4 = layers.Conv2D(start_neurons * 8, (3, 3), padding="same")(pool3)
    conv4 = residual_block(conv4, start_neurons * 8)
    conv4 = residual_block(conv4, start_neurons * 8, True)
    pool4 = layers.MaxPooling2D((2, 2))(conv4)
    pool4 = layers.Dropout(DropoutRatio)(pool4)

    # Middle
    convm = layers.Conv2D(start_neurons * 16, (3, 3), padding="same")(pool4)
    convm = residual_block(convm, start_neurons * 16)
    convm = residual_block(convm, start_neurons * 16, True)

    # 8 -> 16
    deconv4 = layers.Conv2DTranspose(start_neurons * 8, (3, 3), strides=(2, 2), padding="same")(convm)
    att4 = attention_block(deconv4, conv4, start_neurons * 8)
    deconv4 = layers.concatenate([deconv4, att4])
    deconv4 = layers.Dropout(DropoutRatio)(deconv4)
    deconv4 = layers.Conv2D(start_neurons * 8, (3, 3), padding="same")(deconv4)
    deconv4 = residual_block(deconv4, start_neurons * 8)
    deconv4 = residual_block(deconv4, start_neurons * 8, True)

    # 16 -> 32
    deconv3 = layers.Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same")(deconv4)
    att3 = attention_block(deconv3, conv3, start_neurons * 4)
    deconv3 = layers.concatenate([deconv3, att3])
    deconv3 = layers.Dropout(DropoutRatio)(deconv3)
    deconv3 = layers.Conv2D(start_neurons * 4, (3, 3), padding="same")(deconv3)
    deconv3 = residual_block(deconv3, start_neurons * 4)
    deconv3 = residual_block(deconv3, start_neurons * 4, True)

    # 32 -> 64
    deconv2 = layers.Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2), padding="same")(deconv3)
    att2 = attention_block(deconv2, conv2, start_neurons * 2)
    deconv2 = layers.concatenate([deconv2, att2])
    deconv2 = layers.Dropout(DropoutRatio)(deconv2)
    deconv2 = layers.Conv2D(start_neurons * 2, (3, 3), padding="same")(deconv2)
    deconv2 = residual_block(deconv2, start_neurons * 2)
    deconv2 = residual_block(deconv2, start_neurons * 2, True)

    # 64 -> 128
    deconv1 = layers.Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="same")(deconv2)
    att1 = attention_block(deconv1, conv1, start_neurons * 1)
    deconv1 = layers.concatenate([deconv1, att1])
    deconv1 = layers.Dropout(DropoutRatio)(deconv1)
    deconv1 = layers.Conv2D(start_neurons * 1, (3, 3), padding="same")(deconv1)
    deconv1 = residual_block(deconv1, start_neurons * 1)
    deconv1 = residual_block(deconv1, start_neurons * 1, True)

    output = layers.Conv2D(1, (1, 1), padding="same", activation='sigmoid')(deconv1)
    model = models.Model(input_layer, output)
    return model


if __name__ == '__main__':
    model = build_model()
    print(model.summary())
