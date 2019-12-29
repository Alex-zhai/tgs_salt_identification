# -*- coding:utf-8 -*-
# Author : zhaijianwei
# Date : 2018/9/29 17:32


from resnet_pretrain import models
from keras import layers
from keras.models import Model
from keras.backend import tf as ktf


class Interp(layers.Layer):
    def __init__(self, new_size, **kwargs):
        self.new_size = new_size
        super(Interp, self).__init__(**kwargs)

    def build(self, input_shape):
        super(Interp, self).build(input_shape)

    def call(self, inputs, **kwargs):
        new_height, new_width = self.new_size
        resized = ktf.image.resize_images(inputs, [new_height, new_width], align_corners=True)
        return resized

    def compute_output_shape(self, input_shape):
        return tuple([None, self.new_size[0], self.new_size[1], input_shape[3]])

    def get_config(self):
        config = super(Interp, self).get_config()
        config['new_size'] = self.new_size
        return config


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


# Build model
def build_model(input_shape=(224, 224, 3), DropoutRatio=0.5):
    input_layer = layers.Input(shape=input_shape)

    base_model = models.ResNet34(input_shape=input_shape, weights='imagenet', include_top=False,
                                 input_tensor=input_layer)

    conv4 = base_model.get_layer("stage4_unit1_relu1").output
    conv3 = base_model.get_layer("stage3_unit1_relu1").output
    conv2 = base_model.get_layer("stage2_unit1_relu1").output
    conv1 = base_model.get_layer("relu0").output

    mid = base_model.get_layer("relu1").output

    # mid = layers.Conv2D(512, (3, 3), padding="same")(mid)
    # mid = residual_block(mid, 512)
    # mid = residual_block(mid, 512)

    # 4 -> 8
    deconv4 = layers.Conv2DTranspose(256, (3, 3), strides=(2, 2), padding="same")(mid)
    uconv4 = layers.concatenate([deconv4, conv4])
    uconv4 = layers.Dropout(DropoutRatio)(uconv4)
    uconv4 = layers.Conv2D(256, (3, 3), padding="same")(uconv4)
    uconv4 = residual_block(uconv4, 256)
    uconv4 = residual_block(uconv4, 256, True)

    # 8 -> 16
    deconv3 = layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding="same")(uconv4)
    uconv3 = layers.concatenate([deconv3, conv3])
    uconv3 = layers.Dropout(DropoutRatio)(uconv3)
    uconv3 = layers.Conv2D(128, (3, 3), padding="same")(uconv3)
    uconv3 = residual_block(uconv3, 128)
    uconv3 = residual_block(uconv3, 128, True)

    # 16 -> 32
    deconv2 = layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding="same")(uconv3)
    uconv2 = layers.concatenate([deconv2, conv2])
    uconv2 = layers.Dropout(DropoutRatio)(uconv2)
    uconv2 = layers.Conv2D(64, (3, 3), padding="same")(uconv2)
    uconv2 = residual_block(uconv2, 64)
    uconv2 = residual_block(uconv2, 64, True)

    # 32 -> 64
    deconv1 = layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), padding="same")(uconv2)
    uconv1 = layers.concatenate([deconv1, conv1])
    uconv1 = layers.Dropout(DropoutRatio)(uconv1)
    uconv1 = layers.Conv2D(32, (3, 3), padding="same")(uconv1)
    uconv1 = residual_block(uconv1, 32)
    uconv1 = residual_block(uconv1, 32, True)

    # 64 -> 128
    uconv0 = layers.Conv2DTranspose(32, (3, 3), activation='relu', strides=(2, 2), padding="same")(uconv1)

    hyper1 = Interp(new_size=(input_shape[0], input_shape[1]))(uconv1)
    hyper2 = Interp(new_size=(input_shape[0], input_shape[1]))(uconv2)
    hyper3 = Interp(new_size=(input_shape[0], input_shape[1]))(uconv3)
    hyper4 = Interp(new_size=(input_shape[0], input_shape[1]))(uconv4)

    hyper_out = layers.concatenate([uconv0, hyper1, hyper2, hyper3, hyper4])
    hyper_out = layers.Dropout(DropoutRatio)(hyper_out)

    output_layer = layers.Conv2D(1, (1, 1), strides=(1, 1), padding="same")(hyper_out)
    output_layer = layers.Activation('sigmoid')(output_layer)

    model = Model(input_layer, output_layer)
    return model


if __name__ == '__main__':
    model = build_model()
    print(model.summary())
