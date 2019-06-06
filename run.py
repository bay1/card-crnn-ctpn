# -*- coding: utf-8 -*-
import os
import time
import uuid
import base64

from flask import Flask
from flask import jsonify
from flask import request
from flask import render_template

import ctpn.params as ctpn_params
from ctpn.test import ctpn_recognition
from crnn.test import crnn_recognition

app = Flask(__name__)
app.debug = True


def check_floder(floder):
    """
    check floder
    """
    if not os.path.exists(floder):
        os.mkdir(floder)


def handle_img(image):
    """
    crnn + ctpn handle image
    """
    _img = image.encode().split(b';base64,')[-1]
    _img = base64.b64decode(_img)
    jobid = uuid.uuid1().__str__()
    check_floder('data/test_images')
    test_images_path = 'data/test_images/{}.jpg'.format(jobid)
    with open(test_images_path, 'wb') as f:
        f.write(_img)

    timeTake = time.time()

    part_image = ctpn_recognition(test_images_path, app)

    middle_path = os.path.join(ctpn_params.middle_path + '{}.jpg'.format(jobid))
    ctpn_result_image = open(middle_path, 'rb')

    sim_pred = crnn_recognition(part_image, app)

    timeTake = time.time() - timeTake

    os.remove(test_images_path)

    return {
        'text': sim_pred,
        'result_image': base64.b64encode(ctpn_result_image.read()).decode('utf-8'),
        'timeTake': round(timeTake, 4)
    }


@app.route("/", methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        data = request.get_json(force=True)
        ocr_result = handle_img(data['img'])
        return jsonify(ocr_result)
    return render_template('index.html')


if __name__ == "__main__":
    app.run()
