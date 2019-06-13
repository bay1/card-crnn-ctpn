import os
import random

import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import numpy as np
from warpctc_pytorch import CTCLoss

import utils
import dataset
import params
import models.crnn as crnn


# custom weights initialization called on crnn
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def val(net, criterion, max_iter=100):
    print('Start val')
    # read test set
    test_dataset = dataset.lmdbDataset(
        root=params.valroot, transform=dataset.resizeNormalize((params.imgW, params.imgH)))

    for p in crnn.parameters():
        p.requires_grad = False
    net.eval()
    try:
        data_loader = torch.utils.data.DataLoader(
            test_dataset, shuffle=True, batch_size=params.batchSize, num_workers=int(params.workers))
        val_iter = iter(data_loader)
        i = 0
        n_correct = 0
        loss_avg = utils.averager()

        max_iter = min(max_iter, len(data_loader))
        for i in range(max_iter):
            data = val_iter.next()
            i += 1
            cpu_images, cpu_texts = data
            batch_size = cpu_images.size(0)
            utils.loadData(image, cpu_images)
            t, l = converter.encode(cpu_texts)
            utils.loadData(text, t)
            utils.loadData(length, l)
            preds = crnn(image)
            preds_size = Variable(torch.IntTensor(
                [preds.size(0)] * batch_size))
            cost = criterion(preds, text, preds_size, length) / batch_size
            loss_avg.add(cost)

            _, preds = preds.max(2)
            # preds = preds.squeeze(2)
            preds = preds.transpose(1, 0).contiguous().view(-1)
            sim_preds = converter.decode(
                preds.data, preds_size.data, raw=False)
            list_1 = []
            for i in cpu_texts:
                list_1.append(i.decode('utf-8', 'strict'))
            for pred, target in zip(sim_preds, list_1):
                if pred == target:
                    n_correct += 1

        raw_preds = converter.decode(preds.data, preds_size.data, raw=True)[
            :params.n_test_disp]
        for raw_pred, pred, gt in zip(raw_preds, sim_preds, list_1):
            print('%-20s => %-20s, gt: %-20s' % (raw_pred, pred, gt))

        # print(n_correct)
        # print(max_iter * params.batchSize)
        accuracy = n_correct / float(max_iter * params.batchSize)
        print('Test loss: %f, accuray: %f' % (loss_avg.val(), accuracy))
    except:
        pass


def train_batch(crnn, criterion, optimizer, train_iter):
    data = train_iter.next()
    cpu_images, cpu_texts = data
    batch_size = cpu_images.size(0)
    utils.loadData(image, cpu_images)
    t, l = converter.encode(cpu_texts)
    utils.loadData(text, t)
    utils.loadData(length, l)

    preds = crnn(image)
    preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))
    cost = criterion(preds, text, preds_size, length) / batch_size
    crnn.zero_grad()
    cost.backward()
    optimizer.step()
    return cost


def training(crnn, train_loader, criterion, optimizer):
    for total_steps in range(params.niter):
        train_iter = iter(train_loader)
        i = 0
        while i < len(train_loader):
            for p in crnn.parameters():
                p.requires_grad = True
            crnn.train()
            cost = train_batch(crnn, criterion, optimizer, train_iter)
            loss_avg.add(cost)
            i += 1
            if i % params.displayInterval == 0:
                print('[%d/%d][%d/%d] Loss: %f' %
                      (total_steps, params.niter, i, len(train_loader), loss_avg.val()))
                loss_avg.reset()
            if i % params.valInterval == 0:
                val(crnn, criterion)
        if total_steps % params.saveInterval == 0:
            save_name = '{0}/crnn_Rec_done_{1}_{2}.pth'.format(
                params.experiment, total_steps, i)
            torch.save(crnn.state_dict(), save_name)
            print('%s saved' % save_name)


if __name__ == '__main__':
    manualSeed = random.randint(1, 10000)  # fix seed
    random.seed(manualSeed)
    np.random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    cudnn.benchmark = True

    # store model path
    if not os.path.exists(params.experiment):
        os.mkdir(params.experiment)

    # read train set
    train_dataset = dataset.lmdbDataset(root=params.trainroot)
    assert train_dataset
    if not params.random_sample:
        sampler = dataset.randomSequentialSampler(
            train_dataset, params.batchSize)
    else:
        sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=params.batchSize,
        shuffle=True, sampler=sampler,
        num_workers=int(params.workers),
        collate_fn=dataset.alignCollate(imgH=params.imgH, imgW=params.imgW, keep_ratio=params.keep_ratio))

    nclass = len(params.alphabet) + 1
    nc = 1

    converter = utils.strLabelConverter(params.alphabet)
    criterion = CTCLoss()
    # criterion = torch.nn.CTCLoss()

    # cnn and rnn
    image = torch.FloatTensor(params.batchSize, 3, params.imgH, params.imgH)
    text = torch.IntTensor(params.batchSize * 5)
    length = torch.IntTensor(params.batchSize)

    crnn = crnn.CRNN(params.imgH, nc, nclass, params.nh)
    if params.cuda:
        crnn.cuda()
        image = image.cuda()
        criterion = criterion.cuda()

    crnn.apply(weights_init)
    if params.crnn != '':
        print('loading pretrained model from %s' % params.crnn)

        preWeightDict = torch.load(
            params.crnn, map_location=lambda storage, loc: storage)  # 加入项目训练的权重
        modelWeightDict = crnn.state_dict()
        for k, v in preWeightDict.items():
            name = k.replace('module.', '')  # remove `module.`
            if 'rnn.1.embedding' not in name:  # 不加载最后一层权重
                modelWeightDict[name] = v

        crnn.load_state_dict(modelWeightDict)

    image = Variable(image)
    text = Variable(text)
    length = Variable(length)

    # loss averager
    loss_avg = utils.averager()

    # setup optimizer
    if params.adam:
        optimizer = optim.Adam(crnn.parameters(), lr=params.lr,
                               betas=(params.beta1, 0.999))
    elif params.adadelta:
        optimizer = optim.Adadelta(crnn.parameters(), lr=params.lr)
    else:
        optimizer = optim.RMSprop(crnn.parameters(), lr=params.lr)

    training(crnn, train_loader, criterion, optimizer)
