#!/usr/bin/python
# encoding: utf-8
import numpy as np
import sys, os
import time

sys.path.append(os.getcwd())


# crnn packages
import torch
from torch.autograd import Variable
import crnn.utils
import crnn.dataset
from PIL import Image
import crnn.models.crnn as crnn

str1 = '1234567890' # 要识别的数字类型


# crnn params
alphabet = str1
nclass = len(alphabet)+1


# crnn文本信息识别
def crnn_recognition(cropped_image, model):

    converter = utils.strLabelConverter(alphabet)
  
    image = cropped_image.convert('L')
    print(image.size[0])

    ## 
    w = int(image.size[0] / (280 * 1.0 / 160))
    transformer = dataset.resizeNormalize((w, 32))
    image = transformer(image)
    if torch.cuda.is_available():
        image = image.cuda()
    image = image.view(1, *image.size())
    image = Variable(image)

    model.eval()
    preds = model(image)

    _, preds = preds.max(2)
    preds = preds.transpose(1, 0).contiguous().view(-1)

    preds_size = Variable(torch.IntTensor([preds.size(0)]))
    sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
    print('results: {0}'.format(sim_pred))


def test_image(partImg, crnn_model_path):

	# crnn network
    model = crnn.CRNN(32, 1, nclass, 256)
    if torch.cuda.is_available():
        model = model.cuda()
    print('loading pretrained model from {0}'.format(crnn_model_path))
    # 导入已经训练好的crnn模型
    model.load_state_dict(torch.load(crnn_model_path))
    
    started = time.time()

    crnn_recognition(partImg, model)
    finished = time.time()
    print('elapsed time: {0}'.format(finished-started))
    