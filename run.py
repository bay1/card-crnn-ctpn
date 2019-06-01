# -*- coding: utf-8 -*-
import os
import cv2
import time
import uuid
import base64

from flask import Flask
from flask import jsonify
from flask import request
from flask import render_template

app = Flask(__name__)
app.debug = True


def handle_img(image):
    imgString = image.encode().split(b';base64,')[-1]
    imgString = base64.b64decode(imgString)
    jobid = uuid.uuid1().__str__()
    path = '{}.jpg'.format(jobid)
    with open(path, 'wb') as f:
        f.write(imgString)
    img = cv2.imread(path)  # GBR
    H, W = img.shape[:2]

    timeTake = time.time()
    # path
    timeTake = time.time()-timeTake
    res = [{'text': 'text', 'name': '0', 'box': [0, 0, W, 0, W, H, 0, H]}]

    os.remove(path)
    return {'res': res, 'timeTake': round(timeTake, 4)}


@app.route("/", methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        data = request.get_json(force=True)
        ocr_result = handle_img(data['img'])
        return jsonify(ocr_result)
    return render_template('index.html')


if __name__ == "__main__":
    app.run()
