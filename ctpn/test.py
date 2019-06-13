# coding=utf-8
import os
import shutil

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

import ctpn.params as params
from ctpn.utils.text_connector.detectors import TextDetector
from ctpn.utils.rpn_msr.proposal_layer import proposal_layer
from ctpn.nets import model_train as model


def get_wh(box_coordinate):
    """
    根据box坐标宽高
    宽最大的很大可能就是银行卡号位置
    box格式： [xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax, score]
    """
    xmin = box_coordinate[0]
    xmax = box_coordinate[2]
    ymin = box_coordinate[1]
    ymax = box_coordinate[5]
    width = xmax - xmin
    height = ymax - ymin
    return width, height


def resize_image(img):
    img_size = img.shape
    im_size_min = np.min(img_size[0:2])
    im_size_max = np.max(img_size[0:2])

    im_scale = float(600) / float(im_size_min)
    if np.round(im_scale * im_size_max) > 1200:
        im_scale = float(1200) / float(im_size_max)
    new_h = int(img_size[0] * im_scale)
    new_w = int(img_size[1] * im_scale)

    new_h = new_h if new_h // 16 == 0 else (new_h // 16 + 1) * 16
    new_w = new_w if new_w // 16 == 0 else (new_w // 16 + 1) * 16

    re_im = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return re_im, (new_h / img_size[0], new_w / img_size[1])


def ctpn_recognition(test_images_path, app):
    if os.path.exists(params.middle_path):
        shutil.rmtree(params.middle_path)
    os.makedirs(params.middle_path)
    os.environ['CUDA_VISIBLE_DEVICES'] = params.gpu

    with tf.get_default_graph().as_default():
        input_image = tf.placeholder(
            tf.float32, shape=[None, None, None, 3], name='input_image')
        input_im_info = tf.placeholder(
            tf.float32, shape=[None, 3], name='input_im_info')

        global_step = tf.get_variable(
            'global_step', [], initializer=tf.constant_initializer(0), trainable=False)

        bbox_pred, cls_pred, cls_prob = model.model(input_image)

        variable_averages = tf.train.ExponentialMovingAverage(
            0.997, global_step)
        saver = tf.train.Saver(variable_averages.variables_to_restore())

        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            ckpt_state = tf.train.get_checkpoint_state(params.checkpoint_path)
            model_path = os.path.join(params.checkpoint_path, os.path.basename(
                ckpt_state.model_checkpoint_path))
            app.logger.info('Restore from {}'.format(model_path))
            saver.restore(sess, model_path)

            im = cv2.imread(test_images_path)[:, :, ::-1]

            img, (rh, rw) = resize_image(im)
            h, w, c = img.shape
            im_info = np.array([h, w, c]).reshape([1, 3])
            bbox_pred_val, cls_prob_val = sess.run([bbox_pred, cls_prob],
                                                   feed_dict={input_image: [img],
                                                              input_im_info: im_info})

            textsegs, _ = proposal_layer(
                cls_prob_val, bbox_pred_val, im_info)
            scores = textsegs[:, 0]
            textsegs = textsegs[:, 1:5]

            textdetector = TextDetector(DETECT_MODE='H')
            boxes = textdetector.detect(
                textsegs, scores[:, np.newaxis], img.shape[:2])
            boxes = np.array(boxes, dtype=np.int)

            img_copy = img.copy()

            boxes_array = np.array(boxes, dtype=np.int)

            widths = {}
            for i, box in enumerate(boxes_array):
                width, height = get_wh(box[:8].tolist())  # 计算宽高比
                widths[width] = [i, height]

            width_max = max(widths)
            width_max_value = widths[width_max]
            part_img = img.copy()

            for i, box in enumerate(boxes_array):

                color = (0, 255, 0)

                if i == width_max_value[0] and width_max_value[1] > 20:
                    color = (255, 0, 0)
                    box[0] = box[0] - 5
                    box[2] = box[2] + 5
                    part_img = img[box[1]:box[5], box[0]:box[2]][:, :, 0]

                cv2.polylines(img_copy, [box[:8].astype(np.int32).reshape((-1, 1, 2))], True, color=color,
                              thickness=2)

            img_copy = cv2.resize(
                img_copy, None, None, fx=1.0 / rh, fy=1.0 / rw, interpolation=cv2.INTER_LINEAR)
            cv2.imwrite(os.path.join(params.middle_path,
                                     os.path.basename(test_images_path)), img_copy[:, :, ::-1])
            # cv2.imwrite(os.path.join('data/',
            #                              os.path.basename(test_images_path)), part_img)

            part_img = Image.fromarray(part_img.astype('uint8'))

            return part_img
