from torch.utils.data import DataLoader
import torch.nn.functional as F
# from functools import partial
from sklearn.metrics import *
import torch.optim as optim
# import torch.nn as nn
# from tqdm import tqdm
import torch, sys, os
import pandas as pd
from util import *
import numpy as np
from data import *
from tnet import *

''' RUN TRAIN '''
def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    if args.verbose:
        print(' [ %d train' % epoch)
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if args.verbose:
            if batch_idx % args.log == 0:
                print('  | {:.0f}%\tloss:\t{:.6f}'.format(
                    100. * batch_idx / len(train_loader),
                    loss.item()))

''' RUN TEST '''
def test(args, model, device, test_loader, epoch):
    model.eval()
    test_loss = 0
    correct = 0
    dfp, dfl = pd.DataFrame(), pd.DataFrame()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            dfp = dfp.append(pd.DataFrame(F.softmax(output,dim=1).tolist()), sort=False, ignore_index=True)
            dfl = dfl.append(pd.DataFrame(target.tolist()), sort=False, ignore_index=True)

    y = [l[0] for l in dfl.values]
    test_loss /= len(test_loader.dataset)
    accuracy = float(100. * correct) / float(len(test_loader.dataset))
    score = 100 / (1 + log_loss(y, dfp.values, eps=1E-15))
    print(' [ {}\ttest\t{:.5f}\t{:.4f}\t{:.2f}%'.format(epoch, test_loss, score, accuracy))
    return test_loss

''' RUN '''
def tnet(args, train_raw, test_raw, dset='stack', net='stack'):
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    # if dset != 'raw':
    dir, base = 'data', 'mnist_%s' % dset
    names = map(lambda s: '%s_%s.pkl' % (base, s), ('train','test'))
    train_path, test_path = map(lambda s: os.path.join(dir, s), names)

    if not os.path.exists(dir):
        print('[ creating directory %s' % dir)
        os.mkdir(dir)
    if os.path.exists(train_path):
        print('[ loading %s' % train_path)
        train_data = load_pkl(train_path)
    else:
        train_data = DATASET[dset](train_raw)
        print('[ saving %s' % train_path)
        save_pkl(train_path, train_data)
    if os.path.exists(test_path):
        print('[ loading %s' % test_path)
        test_data = load_pkl(test_path)
    else:
        test_data = DATASET[dset](test_raw, train_data)
        print('[ saving %s' % test_path)
        save_pkl(test_path, test_data)
    # else:
    #     train_data = RawDataset(train_raw)
    #     test_data = RawDataset(test_raw, train_data)

    # return train_data, test_data

    train_loader = DataLoader(train_data, batch_size=args.batch, shuffle=True, **kwargs)
    test_loader = DataLoader(test_data, batch_size=args.test_batch, shuffle=True, **kwargs)

    model = NET[dset]((28, 28)).to(device)
    print(str(model)[:-2])

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose=True,
                                                factor=0.05, cooldown=5, patience=10)

    print('[epoch\tmode\tloss\tscore\taccuracy')
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        scheduler.step(test(args, model, device, test_loader, epoch))

    return model

    # DATA = DATASETS[args.data]
    # shape = SHAPE[args.data]

    # trans = [transforms.ToTensor(), MaskTensor(masks, shape), transforms.Normalize(*stats)]
    # transform = transforms.Compose(trans)

    # train_data = DATA('../data', train=True, download=True, transform=transform)
    # test_data = DATA('../data', train=False, transform=transform)
