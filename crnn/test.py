from collections import OrderedDict

import torch
from torch.autograd import Variable

import crnn.params as params
import crnn.utils
import crnn.dataset
from crnn.models.crnn import CRNN

alphabet = params.alphabet
nclass = len(alphabet) + 1


def crnn_recognition(part_image, app):
    model = CRNN(32, 1, nclass, 256)
    # if torch.cuda.is_available():
    #     model = model.cuda()
    app.logger.info('loading pretrained model from {0}'.format(params.crnn_model_path))

    trainWeights = torch.load(params.crnn_model_path, map_location=lambda storage, loc: storage)
    modelWeights = OrderedDict()
    for k, v in trainWeights.items():
        name = k.replace('module.', '')  # remove `module.`
        modelWeights[name] = v

    model.load_state_dict(modelWeights)
    converter = crnn.utils.strLabelConverter(alphabet)

    image = part_image.convert('L')

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
