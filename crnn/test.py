import sys
import os
import time

sys.path.append(os.getcwd())

# crnn packages
import torch
from torch.autograd import Variable
import crnn.utils
import crnn.dataset
from PIL import Image
from crnn.models.crnn import CRNN
import crnn.alphabets
from collections import OrderedDict

str1 = crnn.alphabets.alphabet

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


def crnn_recognition(cropped_image):
    model = CRNN(32, 1, nclass, 256)
    # if torch.cuda.is_available():
    #     model = model.cuda()
    print('loading pretrained model from {0}'.format(crnn_model_path))

    trainWeights = torch.load(crnn_model_path, map_location=lambda storage, loc: storage)
    modelWeights = OrderedDict()
    for k, v in trainWeights.items():
        name = k.replace('module.', '')  # remove `module.`
        modelWeights[name] = v

    model.load_state_dict(modelWeights)
    converter = crnn.utils.strLabelConverter(alphabet)

    image = cropped_image.convert('L')

    w = int(image.size[0] / (280 * 1.0 / 160))
    transformer = crnn.dataset.resizeNormalize((w, 32))
    image = transformer(image)
    # if torch.cuda.is_available():
    #     image = image.cuda()
    image = image.view(1, *image.size())
    image = Variable(image)

    model.eval()
    preds = model(image)

    _, preds = preds.max(2)
    preds = preds.transpose(1, 0).contiguous().view(-1)

    preds_size = Variable(torch.IntTensor([preds.size(0)]))
    sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
    return sim_pred


if __name__ == '__main__':
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

        result = crnn_recognition(image)
        finished = time.time()
        print('results: {0}'.format(result))
        print('elapsed time: {0}'.format(finished - start))
