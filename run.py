# coding:utf-8
from math import *

import cv2
import numpy as np
from PIL import Image
import sys
import os

sys.path.append("ocr")

from crnn.test import test_image

from ctpn.main import get_part_image

def get_images(test_data_path):
    files = []
    exts = ['jpg', 'png', 'jpeg', 'JPG']
    for parent, dirnames, filenames in os.walk(test_data_path):
        for filename in filenames:
            for ext in exts:
                if filename.endswith(ext):
                    files.append(os.path.join(parent, filename))
                    break
    print('Find {} images'.format(len(files)))
    return files

def crnnRec(partImg):
    """
    crnn模型，ocr识别
    """
    crnn_model_path = 'crnn/trained_models/crnn_Rec_done_155_1084.pth'

    # 根据ctpn进行识别出的文字区域，进行不同文字区域的crnn识别
    image = Image.fromarray(partImg) # 图像矩阵转化成Image对象 并灰度

  
    test_image(image, crnn_model_path)


if __name__ == '__main__':
    '''
    result-识别结果
    '''
    print("---------------------------------------")
    test_data_path = 'data/test_images'
    im_fn_list = get_images(test_data_path)
    for im_fn in im_fn_list:
        # 进行图像中的文字区域的识别
        partImg=get_part_image(im_fn)
        crnnRec(partImg)
