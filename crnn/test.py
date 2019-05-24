import numpy as np
import sys, os
import time

sys.path.append(os.getcwd())

# crnn packages
import torch
from torch.autograd import Variable
import utils
import dataset
from PIL import Image
import models.crnn as crnn
import alphabets

str1 = alphabets.alphabet

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--images_path', type=str, default='data/res/', help='the path to your images')
opt = parser.parse_args()

# crnn params
crnn_model_path = 'crnn/trained_models/crnn_Rec_done.pth'
alphabet = str1
nclass = len(alphabet) + 1


def get_images(images_path):
    files = []
    exts = ['jpg', 'png', 'jpeg', 'JPG']
    for parent, dirnames, filenames in os.walk(images_path):
        for filename in filenames:
            for ext in exts:
                if filename.endswith(ext):
                    files.append(os.path.join(parent, filename))
                    break
    print('Find {} images'.format(len(files)))
    return files


# crnn文本信息识别
def crnn_recognition(cropped_image, model):
    converter = utils.strLabelConverter(alphabet)

    image = cropped_image.convert('L')

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


if __name__ == '__main__':

    # crnn network
    model = crnn.CRNN(32, 1, nclass, 256)
    if torch.cuda.is_available():
        model = model.cuda()
    print('loading pretrained model from {0}'.format(crnn_model_path))
    # 导入已经训练好的crnn模型
    model.load_state_dict(torch.load(crnn_model_path))

    im_fn_list = get_images(opt.images_path)
    for im_fn in im_fn_list:
        print('===============')
        print(im_fn)
        start = time.time()
        try:
            ## read an image
            image = Image.open(im_fn)
        except:
            print("Error reading image {}!".format(im_fn))
            continue

        crnn_recognition(image, model)
        finished = time.time()
        print('elapsed time: {0}'.format(finished - start))
