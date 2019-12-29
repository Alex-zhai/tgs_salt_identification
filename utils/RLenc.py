# -*- coding:utf-8 -*-
# Author : zhaijianwei
# Date : 2018/9/20 15:47

import numpy as np


def rle_encode(im):
    pixels = im.flatten(order='F')
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)
