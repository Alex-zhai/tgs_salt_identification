# -*- coding:utf-8 -*-
# Author : zhaijianwei
# Date : 2018/9/19 14:55


# reference: https://github.com/ykamikawa/PSPNet/blob/master/train.py

import keras.backend as K
from tensorflow.python.keras import layers
from tensorflow.python.keras import Model


def identity_block(input_tensor, kernel_size, filters, stage, block, dilation_rate=1, multigrid=[1, 2, 1], use_se=True):
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    if dilation_rate < 2:
        multigrid = [1, 1, 1]

    x = layers.Conv2D(filters1, kernel_size=(1, 1), name=conv_name_base + '2a',
                      dilation_rate=dilation_rate * multigrid[0])(input_tensor)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters2, kernel_size=kernel_size, padding='same', name=conv_name_base + '2b',
                      dilation_rate=dilation_rate * multigrid[1])(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters3, kernel_size=(1, 1), name=conv_name_base + '2c',
                      dilation_rate=dilation_rate * multigrid[2])(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    # stage 5 after squeeze and excinttation layer
    if use_se and stage < 5:
        se = _squeeze_excite_block(x, filters3, k=1, name=conv_name_base + '_se')
        x = layers.multiply([x, se])
    x = layers.add([x, input_tensor])
    x = layers.Activation('relu')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2), dilation_rate=1, multigrid=[1, 2, 1],
               use_se=True):
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    if dilation_rate > 1:
        strides = (1, 1)
    else:
        multigrid = [1, 1, 1]

    x = layers.Conv2D(filters1, kernel_size=(1, 1), strides=strides, name=conv_name_base + '2a',
                      dilation_rate=dilation_rate * multigrid[0])(input_tensor)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters2, kernel_size=kernel_size, padding='same', name=conv_name_base + '2b',
                      dilation_rate=dilation_rate * multigrid[1])(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters3, kernel_size=(1, 1), name=conv_name_base + '2c',
                      dilation_rate=dilation_rate * multigrid[2])(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = layers.Conv2D(filters3, kernel_size=(1, 1), strides=strides, name=conv_name_base + '1')(input_tensor)

    if use_se and stage < 5:
        se = _squeeze_excite_block(x, filters3, k=1, name=conv_name_base + '_se')
        x = layers.multiply([x, se])
    x = layers.add([x, shortcut])
    x = layers.Activation('relu')(x)
    return x


def _conv(**conv_params):
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", (1, 1))
    dilation_rate = conv_params.setdefault('dilation_rate', (1, 1))
    kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
    padding = conv_params.setdefault("padding", "same")
    block = conv_params.setdefault("block", "assp")

    def f(input):
        conv = layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding,
                             dilation_rate=dilation_rate, kernel_initializer=kernel_initializer,
                             activation='linear')(input)
        return conv

    return f


# squeeze and excitation function
def _squeeze_excite_block(input, filters, k=1, name=None):
    init = input
    se_shape = (1, 1, filters * k) if K.image_data_format() == 'channels_last' else (filters * k, 1, 1)

    se = layers.GlobalAveragePooling2D()(init)
    se = layers.Reshape(se_shape)(se)
    se = layers.Dense((filters * k) // 16, activation='relu', kernel_initializer='he_normal', use_bias=False,
                      name=name + '_fc1')(se)
    se = layers.Dense(filters * k, activation='sigmoid', kernel_initializer='he_normal', use_bias=False,
                      name=name + '_fc2')(se)
    return se


def pyramid_pooling_module(x, num_filters=512, input_shape=(224, 224, 3), output_stride=16, levels=[6, 3, 2, 1]):
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    pyramid_pooling_blocks = [x]
    for level in levels:
        pyramid_pooling_blocks.append(
            interp_block(x, num_filters=num_filters, level=level, input_shape=input_shape, output_stride=output_stride))
    y = layers.concatenate(pyramid_pooling_blocks)
    y = _conv(filters=num_filters, kernel_size=(3, 3), padding='same', block='pyramid_out_%s' % output_stride)(y)
    y = layers.BatchNormalization(axis=bn_axis, name='bn_pyramid_out_%s' % output_stride)(y)
    y = layers.Activation('relu')(y)
    return y


# interpolation
def Interp(x, shape):
    from keras.backend import tf as ktf
    new_height, new_width = shape
    resized = ktf.image.resize_images(x, [int(new_height), int(new_width)], align_corners=True)
    return resized


# interpolation block
def interp_block(x, num_filters=512, level=1, input_shape=(224, 224, 3), output_stride=16):
    feature_map_shape = (input_shape[0] / output_stride, input_shape[1] / output_stride)

    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    # if output_stride == 16:
    #     scale = 5
    # elif output_stride == 8:
    #     scale = 10
    scale = 1
    kernel = (level * scale, level * scale)
    strides = (level * scale, level * scale)
    global_feat = layers.AveragePooling2D(kernel, strides=strides, name='pool_level_%s_%s' % (level, output_stride))(x)
    global_feat = _conv(
        filters=num_filters,
        kernel_size=(1, 1),
        padding='same',
        name='conv_level_%s_%s' % (level, output_stride))(global_feat)
    global_feat = layers.BatchNormalization(axis=bn_axis, name='bn_level_%s_%s' % (level, output_stride))(global_feat)
    global_feat = layers.Lambda(Interp, arguments={'shape': feature_map_shape})(global_feat)

    return global_feat


def resnet_50(x, multigrid=[1, 1, 1], output_stride=8, num_blocks=4, use_se=True):
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    # resnet50 stage1
    x = layers.Conv2D(64, kernel_size=(7, 7), strides=(2, 2), padding='same', name='conv1')(x)
    x = layers.BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)
    # stage 2
    x = conv_block(x, kernel_size=3, filters=[64, 64, 256], stage=2, block='a', strides=(1, 1), use_se=use_se)
    x = identity_block(x, kernel_size=3, filters=[64, 64, 256], stage=2, block='b', use_se=use_se)
    x = identity_block(x, kernel_size=3, filters=[64, 64, 256], stage=2, block='c', use_se=use_se)
    # stage 3
    x = conv_block(x, kernel_size=3, filters=[128, 128, 512], stage=3, block='a', use_se=use_se)
    x = identity_block(x, kernel_size=3, filters=[128, 128, 512], stage=3, block='b', use_se=use_se)
    x = identity_block(x, kernel_size=3, filters=[128, 128, 512], stage=3, block='c', use_se=use_se)
    x = identity_block(x, kernel_size=3, filters=[128, 128, 512], stage=3, block='d', use_se=use_se)

    if output_stride == 8:
        rate_scale = 2
    else:
        rate_scale = 1

    # stage 4
    x = conv_block(x, kernel_size=3, filters=[256, 256, 1024], stage=4, block='a', dilation_rate=1 * rate_scale,
                   multigrid=multigrid, use_se=use_se)
    x = identity_block(x, kernel_size=3, filters=[256, 256, 1024], stage=4, block='b', dilation_rate=1 * rate_scale,
                       multigrid=multigrid, use_se=use_se)
    x = identity_block(x, kernel_size=3, filters=[256, 256, 1024], stage=4, block='c', dilation_rate=1 * rate_scale,
                       multigrid=multigrid, use_se=use_se)
    x = identity_block(x, kernel_size=3, filters=[256, 256, 1024], stage=4, block='d', dilation_rate=1 * rate_scale,
                       multigrid=multigrid, use_se=use_se)
    x = identity_block(x, kernel_size=3, filters=[256, 256, 1024], stage=4, block='e', dilation_rate=1 * rate_scale,
                       multigrid=multigrid, use_se=use_se)
    x = identity_block(x, kernel_size=3, filters=[256, 256, 1024], stage=4, block='f', dilation_rate=1 * rate_scale,
                       multigrid=multigrid, use_se=use_se)

    # stage 5
    init_rate = 2
    for block in range(4, num_blocks + 1):
        if block == 4:
            block = ''
        x = conv_block(x, kernel_size=3, filters=[512, 512, 2048], stage=5, block='a%s' % block,
                       dilation_rate=init_rate * rate_scale, multigrid=multigrid, use_se=use_se)
        x = identity_block(x, kernel_size=3, filters=[512, 512, 2048], stage=5, block='b%s' % block,
                           dilation_rate=init_rate * rate_scale, multigrid=multigrid, use_se=use_se)
        x = identity_block(x, kernel_size=3, filters=[512, 512, 2048], stage=5, block='c%s' % block,
                           dilation_rate=init_rate * rate_scale, multigrid=multigrid, use_se=use_se)
        init_rate *= 2
    return x


def psp_net_with_dilation(input_shape=(128, 128, 1), n_labels=2, output_stride=8, levels=[6, 3, 2, 1],
                          upsample_type='deconv'):
    img_input = layers.Input(shape=input_shape)

    x = resnet_50(img_input)
    x = pyramid_pooling_module(x, num_filters=512, input_shape=input_shape, output_stride=output_stride, levels=levels)

    if upsample_type == 'deconv':
        out = layers.Conv2DTranspose(filters=n_labels, kernel_size=(output_stride * 2, output_stride * 2),
                                     strides=(output_stride, output_stride), padding='same',
                                     kernel_initializer='he_normal', activation='sigmoid',
                                     kernel_regularizer=None, use_bias=False, name='upscore_{}'.format('out'))(x)
    elif upsample_type == 'bilinear':
        # x = layers.Conv2D(512, (3, 3), strides=(1, 1), padding='same', name="out_conv1", use_bias=False)(x)
        # x = layers.BatchNormalization(axis=bn_axis, momentum=0.95, epsilon=1e-5, name="out_conv1_bn")(x)
        # x = layers.Activation('relu')(x)
        # x = layers.Dropout(0.1)(x)
        x = _conv(filters=n_labels, kernel_size=(1, 1), padding='same', block='out_bilinear_%s' % output_stride)(x)
        out = layers.Lambda(Interp, arguments={'shape': (input_shape[0], input_shape[1])})(x)

    model = Model(inputs=img_input, outputs=out)

    return model


if __name__ == '__main__':
    model = psp_net_with_dilation(upsample_type='bilinear')
    print(model.summary())
