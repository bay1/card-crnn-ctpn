# encoding:utf-8
import os
import re
import shutil
import random

import cv2
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
import numpy as np

import params
import utils


def add_noise(img):
    for i in range(20):
        temp_x = np.random.randint(0, img.shape[0])
        temp_y = np.random.randint(0, img.shape[1])
        img[temp_x][temp_y] = 255
    return img


def add_erode(img):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    img = cv2.erode(img, kernel)
    return img


def add_dilate(img):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    img = cv2.dilate(img, kernel)
    return img


def do(img):
    if random.random() < 0.5:
        im = add_noise(img)
    if random.random() < 0.5:
        im = add_dilate(img)
    else:
        im = add_erode(img)
    return im


# 加载图像，并转化为numpy的array
utils.check_floder_exist(params.augmentation_output)
for file in os.listdir(params.augmentation_original_dir):
    if file.endswith('.png') or file.endswith('.jpg'):
        image_path = '%s/%s' % (params.augmentation_original_dir, file)
        filename = file.split('.')[0]  # 图片名称
        prefix = re.sub("\D", "", filename)[:-1]  # 正确文字
        image = load_img(image_path)
        image = img_to_array(image)
        image = do(image)
        # 增加一个维度
        image = np.expand_dims(image, axis=0)  # 在0位置增加数据，主要是batch size

        aug = ImageDataGenerator(
            width_shift_range=0.05,  # 水平平移幅度
            height_shift_range=0.05,  # 上下平移幅度
            shear_range=0.2,  # 逆时针方向的剪切变黄角度
            zoom_range=0.2,  # 随机缩放的角度
            fill_mode='nearest'  # 变换超出边界的处理
        )
        # 初始化目前为止的图片产生数量
        total = 0

        print("[INFO] generating images...%s" % filename)

        imageGen = aug.flow(image, batch_size=1, save_to_dir=params.augmentation_output,
                            save_prefix=prefix, save_format='png')

        for image in imageGen:
            total += 1
            if total == params.total_num:
                break
