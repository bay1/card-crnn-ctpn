# -*- coding: utf-8 -*-
import os
import cv2
import time
import uuid
import base64

from PIL import Image
from flask import Flask
from flask import jsonify
from flask import request
from flask import render_template

from ctpn.demo import main
from crnn.test import crnn_recognition

app = Flask(__name__)
app.debug = True


def handle_img(image):
    imgString = image.encode().split(b';base64,')[-1]
    imgString = base64.b64decode(imgString)
    jobid = uuid.uuid1().__str__()
    path = 'data/test_images/{}.jpg'.format(jobid)
    with open(path, 'wb') as f:
        f.write(imgString)
    img = cv2.imread(path)
    H, W = img.shape[:2]

    timeTake = time.time()
    main()
    middle_path = 'data/middle_result/{}.jpg'.format(jobid)
    result_image = open(middle_path, 'rb') # 轮廓识别结果

    res_image = 'data/res/{}.jpg'.format(jobid)
    try:
        # read an image
        image = Image.open(res_image)
    except:
        print("Error reading image")

    result = crnn_recognition(image)

    timeTake = time.time() - timeTake

    os.remove(path)
    os.remove(middle_path)
    os.remove(res_image)

    return {
        'text': result,
        'result_image': base64.b64encode(result_image.read()).decode('utf-8'),
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
