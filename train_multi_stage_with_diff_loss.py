# -*- coding:utf-8 -*-
# Author : zhaijianwei
# Date : 2018/9/26 13:44

# -*- coding:utf-8 -*-
# Author : zhaijianwei
# Date : 2018/9/19 13:56

import numpy as np
import pandas as pd
import os
import sys
import cv2
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.preprocessing.image import load_img
from keras.models import load_model, Model
from keras.optimizers import Adam
from keras import backend as K
from skimage.transform import resize
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

file_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(file_path + '/utils')
sys.path.append(file_path + '/losses')
sys.path.append(file_path + '/models')

import pretrained_resnet34_u_net, u_net_with_res_block
import get_batch_iou, RLenc
from losses import *
from losses import lovasz_loss

img_size_ori = 101
# img_size_target = 128
img_size_target = 224


def get_mask_type(mask):
    border = 10
    outer = np.zeros((101 - 2 * border, 101 - 2 * border), np.float32)
    outer = cv2.copyMakeBorder(outer, border, border, border, border, borderType=cv2.BORDER_CONSTANT, value=1)

    cover = (mask > 0.5).sum()
    if cover < 8:
        return 0  # empty
    if cover == ((mask * outer) > 0.5).sum():
        return 1  # border
    if np.all(mask == mask[0]):
        return 2  # vertical

    percentage = cover / (101 * 101)
    if percentage < 0.15:
        return 3
    elif percentage < 0.25:
        return 4
    elif percentage < 0.50:
        return 5
    elif percentage < 0.75:
        return 6
    else:
        return 7


def histcoverage(coverage):
    histall = np.zeros((1, 8))
    for c in coverage:
        histall[0, c] += 1
    return histall


def get_iou_vector(A, B):
    batch_size = A.shape[0]
    metric = []
    for batch in range(batch_size):
        t, p = A[batch] > 0, B[batch] > 0
        intersection = np.logical_and(t, p)
        union = np.logical_or(t, p)
        iou = (np.sum(intersection > 0) + 1e-10) / (np.sum(union > 0) + 1e-10)
        thresholds = np.arange(0.5, 1, 0.05)
        s = []
        for thresh in thresholds:
            s.append(iou > thresh)
        metric.append(np.mean(s))

    return np.mean(metric)


def my_iou_metric(label, pred):
    return tf.py_func(get_iou_vector, [label, pred > 0.5], tf.float64)


def my_iou_metric_2(label, pred):
    return tf.py_func(get_iou_vector, [label, pred > 0], tf.float64)


def upsample(img):
    if img_size_ori == img_size_target:
        return img
    return resize(img, (img_size_target, img_size_target), mode='constant', preserve_range=True)


def downsample(img):
    if img_size_ori == img_size_target:
        return img
    return resize(img, (img_size_ori, img_size_ori), mode='constant', preserve_range=True)


# def cov_to_class(val):
#    for i in range(0, 11):
#        if val * 10 <= i:
#            return i


# step1: prepare data
def gernerate_data():
    train_df = pd.read_csv("data/train.csv", index_col="id", usecols=[0])
    depths_df = pd.read_csv("data/depths.csv", index_col="id")
    train_df = train_df.join(depths_df)
    test_df = depths_df[~depths_df.index.isin(train_df.index)]

    train_df["images"] = [np.array(load_img("data/train/images/{}.png".format(idx), grayscale=False)) / 255 for idx in
                          tqdm(train_df.index)]
    train_df["masks"] = [np.array(load_img("data/train/masks/{}.png".format(idx), grayscale=True)) / 255 for idx in
                         tqdm(train_df.index)]
    train_df["coverage"] = train_df.masks.map(np.sum) / pow(img_size_ori, 2)
    # train_df["coverage_class"] = train_df.coverage.map(cov_to_class)
    train_df["coverage_class"] = train_df.masks.map(get_mask_type)

    x_test = np.array(
        [upsample(np.array(load_img("data/test/images/{}.png".format(idx), grayscale=False))) / 255 for idx in
         tqdm(test_df.index)]).reshape(-1, img_size_target, img_size_target, 3)
    return train_df, test_df, x_test


train_df, test_df, x_test = gernerate_data()


# step2: get train, val data  k-fold
def get_train_and_val(train_index, valid_index):
    # k_fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=1234)
    # for train_index, valid_index in k_fold.split(train_df.index.values, train_df.coverage_class):
    ids_train, ids_valid = train_df.index.values[train_index], train_df.index.values[valid_index]
    x_train = np.array(train_df.images.map(upsample).tolist()).reshape(-1, img_size_target, img_size_target, 3)[
        train_index]
    x_valid = np.array(train_df.images.map(upsample).tolist()).reshape(-1, img_size_target, img_size_target, 3)[
        valid_index]
    y_train = np.array(train_df.masks.map(upsample).tolist()).reshape(-1, img_size_target, img_size_target, 1)[
        train_index]
    y_valid = np.array(train_df.masks.map(upsample).tolist()).reshape(-1, img_size_target, img_size_target, 1)[
        valid_index]
    return ids_train, ids_valid, x_train, x_valid, y_train, y_valid


# step3: data augmentation
def data_augmentation(x_train, x_valid, y_train, y_valid):
    x_train = np.append(x_train, [np.fliplr(x) for x in x_train], axis=0)
    y_train = np.append(y_train, [np.fliplr(x) for x in y_train], axis=0)
    return x_train, x_valid, y_train, y_valid


# step4: create model
def create_model(model_name):
    if model_name == "pretrained_resnet50_unet":
        model = pretrained_resnet50_unet.get_unet_resnet()
    elif model_name == "psp_net":
        model = psp_net.psp_net_with_dilation()
    elif model_name == "u_net_with_res_block":
        model = u_net_with_res_block.build_model()
    elif model_name == "pretrained_resnet34_u_net":
        model = pretrained_resnet34_u_net.build_model()

    model.compile(loss=weighted_bce_dice_loss, optimizer="adam", metrics=[my_iou_metric])
    return model


# step5: train_and_eval model
def train_and_eval_and_submit(model_name):
    threshold_best_sum = 0
    iou_best_sum = 0
    num_folds = 5
    fold_count = 1
    k_fold = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=1234)

    for train_index, valid_index in k_fold.split(train_df.index.values, train_df.coverage_class):
        ids_train, ids_valid, x_train, x_valid, y_train, y_valid = get_train_and_val(train_index, valid_index)
        x_train, x_valid, y_train, y_valid = data_augmentation(x_train, x_valid, y_train, y_valid)

        # stage1 use weighted_bce_dice_loss
        model = create_model(model_name)
        early_stopping = EarlyStopping(monitor='my_iou_metric', mode='max', patience=20)
        lr_on_plateau = ReduceLROnPlateau(monitor='my_iou_metric', mode='max', factor=0.5, patience=5, min_lr=0.00001)
        model_checkpoint = ModelCheckpoint("./model/" + model_name + str(fold_count) + ".model",
                                           monitor="my_iou_metric",
                                           mode="max", save_best_only=True, verbose=1)
        model.fit(x_train, y_train, validation_data=[x_valid, y_valid], epochs=100, batch_size=64,
                  callbacks=[early_stopping, model_checkpoint, lr_on_plateau])
        # model.save_weights("./weight/" + model_name + str(fold_count) + ".h5")

        # stage2 use lovasz_loss

        model = load_model("./model/" + model_name + str(fold_count) + ".model",
                           custom_objects={'my_iou_metric': my_iou_metric,
                                           "weighted_bce_dice_loss": weighted_bce_dice_loss})
        input_x = model.layers[0].input
        output_layer = model.layers[-1].input
        model = Model(input_x, output_layer)
        model.compile(loss=lovasz_loss, optimizer=Adam(lr=0.005), metrics=[my_iou_metric_2])

        early_stopping = EarlyStopping(monitor='val_my_iou_metric_2', mode='max', patience=20)
        reduce_lr = ReduceLROnPlateau(monitor='my_iou_metric_2', mode='max', factor=0.5, patience=5, min_lr=0.00005)
        model_checkpoint = ModelCheckpoint("./model/" + model_name + "_ensemble" + str(fold_count) + ".model",
                                           monitor='my_iou_metric_2',
                                           mode='max', save_best_only=True, verbose=1)
        model.fit(x_train, y_train, validation_data=[x_valid, y_valid], epochs=100, batch_size=64,
                  callbacks=[model_checkpoint, reduce_lr, early_stopping])

        preds_valid = predict_result(model, x_valid)
        preds_valid = np.array([downsample(x) for x in preds_valid])
        y_valid = np.array([downsample(x) for x in y_valid])

        thresholds = np.linspace(0.3, 0.7, 31)
        ious = np.array([get_batch_iou.iou_metric_batch(y_valid, np.int32(preds_valid > threshold)) for threshold in
                         tqdm(thresholds)])
        threshold_best_index = np.argmax(ious)
        iou_best = ious[threshold_best_index]
        threshold_best = thresholds[threshold_best_index]
        print("fold " + str(fold_count) + " threshold best: " + str(threshold_best))
        print("fold " + str(fold_count) + " iou best: " + str(iou_best))

        threshold_best_sum += threshold_best
        iou_best_sum += iou_best
        if fold_count == 1:
            preds_test = predict_result(model, x_test)
        else:
            preds_test += predict_result(model, x_test)
        fold_count += 1
        # K.clear_session()

    print("avg threshold: " + str(threshold_best_sum / num_folds))
    print("avg iou: " + str(iou_best_sum / num_folds))

    pred_dict = {idx: RLenc.rle_encode(np.round(downsample(preds_test[i]) > threshold_best_sum)) for i, idx in
                 enumerate(tqdm(test_df.index.values))}

    sub = pd.DataFrame.from_dict(pred_dict, orient='index')
    sub.index.names = ['id']
    sub.columns = ['rle_mask']
    sub.to_csv('submission_u_net_add_resblock_new_kfold5_batch64_224_multi_stage_training.csv')


# TTA
def predict_result(model, x_test):  # predict both orginal and reflect x
    preds_test1 = model.predict(x_test).reshape(-1, img_size_target, img_size_target)
    x_test_reflect = np.array([np.fliplr(x) for x in x_test])
    preds_test2_reflect = model.predict(x_test_reflect).reshape(-1, img_size_target, img_size_target)
    preds_test2 = np.array([np.fliplr(x) for x in preds_test2_reflect])
    preds_avg = (preds_test1 + preds_test2) / 2
    return preds_avg


if __name__ == '__main__':
    model_name = "pretrained_resnet34_u_net"
    train_and_eval_and_submit(model_name=model_name)
