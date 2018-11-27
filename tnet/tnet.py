# from torch.utils.data import DataLoader
import torch.nn.functional as F
# from functools import partial
# from sklearn.metrics import *
# import torch.optim as optim
import torch.nn as nn
# from tqdm import tqdm
import torch, sys, os
# import pandas as pd
from util import *
import numpy as np
# from data import *

class Net(nn.Module):
    def __init__(self, s):
        super(Net, self).__init__()
        ''' in/out '''
        self.w1, self.w2 = s
        self.n, self.n0 = 10, 1
        ''' neurons '''
        # convolution
        self.n1 = self.n0 * 2
        self.n2 = self.n0 * 4
        self.k1, self.k2 = 4, 2
        self.s1, self.s2 = 1, 1
        self.p1,self.p2 = self.k1 / 2, self.k2 / 2
        self.x1 = self.w1 / (self.n1/self.n0) / (self.n2/self.n1)
        self.x2 = self.w2 / (self.n1/self.n0) / (self.n2/self.n1)
        self.n3 = self.n2 * self.x1 * self.x2
        self.n4 = self.n3 / 2
        self.n5 = self.n4 / 2
        self.n6 = self.n5 / 2

        ''' layers '''
        # convolution
        self.conv1 = nn.Conv2d(self.n0, self.n1, self.k1, self.s1, self.p1)
        self.conv2 = nn.Conv2d(self.n1, self.n2, self.k2, self.s2, self.p2)
        self.conv2_drop = nn.Dropout2d()
        # connected
        self.fc1 = nn.Linear(self.n3, self.n4)
        self.fc2 = nn.Linear(self.n4, self.n5)
        self.fc3 = nn.Linear(self.n5, self.n6)
        self.fc4 = nn.Linear(self.n6, self.n)

    def view(self, x):
        return x.view(-1, self.n3)

    def forward(self, x):
        ''' convolution '''
        # in -> conv1
        x = self.conv1(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        # conv1 -> conv2
        x = self.conv2(x)
        x = self.conv2_drop(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)

        ''' connected '''
        x = self.view(x)
        # conv2 -> linear1
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        # linear1 -> linear2
        x = self.fc2(x)
        x = F.relu(x)
        # linear2 -> linear3
        x = self.fc3(x)
        x = F.relu(x)
        # linear3 -> linear4
        x = self.fc4(x)
        return F.log_softmax(x, dim=1)

class StackedNet(nn.Module):
    def __init__(self, s, k=3):
        super(StackedNet, self).__init__()
        ''' in/out '''
        self.w1, self.w2 = s
        self.n, self.n0 = 10, k + 1
        ''' neurons '''
        # convolution
        self.n1 = self.n0 * 2
        self.n2 = self.n0 * 4
        self.k1, self.k2 = 4, 2
        self.s1, self.s2 = 1, 1
        self.p1,self.p2 = self.k1 / 2, self.k2 / 2
        self.x1 = self.w1 / (self.n1/self.n0) / (self.n2/self.n1)
        self.x2 = self.w2 / (self.n1/self.n0) / (self.n2/self.n1)
        self.n3 = self.n2 * self.x1 * self.x2
        self.n4 = self.n3 / 2
        self.n5 = self.n4 / 2
        self.n6 = self.n5 / 2
        self.n7 = self.n6 / 2
        self.n8 = self.n7 / 2
        # self.n9 = self.n8 / 2

        ''' layers '''
        # convolution
        self.conv1 = nn.Conv2d(self.n0, self.n1, self.k1, self.s1, self.p1)
        self.conv2 = nn.Conv2d(self.n1, self.n2, self.k2, self.s2, self.p2)
        self.conv2_drop = nn.Dropout2d()
        # connected
        self.fc1 = nn.Linear(self.n3, self.n4)
        self.fc2 = nn.Linear(self.n4, self.n5)
        self.fc3 = nn.Linear(self.n5, self.n6)
        self.fc4 = nn.Linear(self.n6, self.n7)
        self.fc5 = nn.Linear(self.n7, self.n8)
        self.fc6 = nn.Linear(self.n8, self.n)

    def view(self, x):
        return x.view(-1, self.n3)

    def forward(self, x):
        ''' convolution '''
        # in -> conv1
        x = self.conv1(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        # conv1 -> conv2
        x = self.conv2(x)
        x = self.conv2_drop(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)

        ''' connected '''
        x = self.view(x)
        # conv2 -> linear1
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        # linear1 -> linear2
        x = self.fc2(x)
        x = F.relu(x)
        # linear2 -> linear3
        x = self.fc3(x)
        x = F.relu(x)
        # linear3 -> linear4
        x = self.fc4(x)
        x = F.relu(x)
        # linear4 -> linear5
        x = self.fc5(x)
        x = F.relu(x)
        # linear5 -> linear6
        x = self.fc6(x)
        return F.log_softmax(x, dim=1)

class StackedNet3D(nn.Module):
    def __init__(self, s, k=3):
        super(StackedNet3D, self).__init__()
        ''' in/out '''
        # self.masks = masks
        # self.c, self.w1, self.w2 = s
        # self.n0 = self.c * (len(self.masks) if len(self.masks) > 0 else 1)
        # self.n = len(masks)
        self.n = 10
        self.n0 = k + 1
        self.w1, self.w2 = s
        ''' neurons '''
        # convolution
        self.n1 = self.n0 * 2
        self.n2 = self.n0 * 4
        self.k1, self.k2 = 4, 2
        self.s1, self.s2 = 1, 1
        self.p1,self.p2 = self.k1 / 2, self.k2 / 2
        self.x1 = self.w1 / (self.n1/self.n0) / (self.n2/self.n1)
        self.x2 = self.w2 / (self.n1/self.n0) / (self.n2/self.n1)
        self.n3 = self.n2 * self.x1 * self.x2
        self.n4 = self.n3 / 2
        self.n5 = self.n4 / 2
        self.n6 = self.n5 / 2
        self.n7 = self.n6 / 2
        self.n8 = self.n7 / 2
        # self.n9 = self.n8 / 2

        ''' layers '''
        # convolution
        self.conv1 = nn.Conv3d(self.n0, self.n1, self.k1, self.s1, self.p1)
        self.conv2 = nn.Conv3d(self.n1, self.n2, self.k2, self.s2, self.p2)
        self.conv2_drop = nn.Dropout3d()
        # connected
        self.fc1 = nn.Linear(self.n3, self.n4)
        self.fc2 = nn.Linear(self.n4, self.n5)
        self.fc3 = nn.Linear(self.n5, self.n6)
        self.fc4 = nn.Linear(self.n6, self.n7)
        self.fc5 = nn.Linear(self.n7, self.n8)
        self.fc6 = nn.Linear(self.n8, self.n)
        # self.fc6 = nn.Linear(self.n8, self.n9)
        # self.fc7 = nn.Linear(self.n9, self.n)

    def view(self, x):
        return x.view(-1, self.n3)

    def forward(self, x):
        ''' convolution '''
        # in -> conv1
        # print(x.shape)
        x = self.conv1(x)
        x = F.max_pool3d(x, 2)
        x = F.relu(x)
        # conv1 -> conv2
        # print(x.shape)
        x = self.conv2(x)
        x = self.conv2_drop(x)
        x = F.max_pool3d(x, 2)
        x = F.relu(x)
        # print(x.shape)

        ''' connected '''
        x = self.view(x)
        # print(x.shape)
        # conv2 -> linear1
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        # linear1 -> linear2
        x = self.fc2(x)
        x = F.relu(x)
        # linear2 -> linear3
        x = self.fc3(x)
        x = F.relu(x)
        # linear3 -> linear4
        x = self.fc4(x)
        x = F.relu(x)
        # linear4 -> linear5
        x = self.fc5(x)
        x = F.relu(x)
        # linear5 -> linear6
        x = self.fc6(x)
        return F.log_softmax(x, dim=1)
        # x = F.relu(x)
        # # linear6 -> linear7 (out)
        # x = self.fc7(x)
        # return F.log_softmax(x, dim=1)

NET = {'raw' : Net, 'cycle' : Net, 'stack' : StackedNet, 'stack3d' : StackedNet3D}

#
# ''' RUN TRAIN '''
# def train(args, model, device, train_loader, optimizer, epoch):
#     model.train()
#     if args.verbose:
#         print(' [ %d train' % epoch)
#     for batch_idx, (data, target) in enumerate(train_loader):
#         data, target = data.to(device), target.to(device)
#         optimizer.zero_grad()
#         output = model(data)
#         loss = F.nll_loss(output, target)
#         loss.backward()
#         optimizer.step()
#         if args.verbose:
#             if batch_idx % args.log == 0:
#                 print('  | {:.0f}%\tloss:\t{:.6f}'.format(
#                     100. * batch_idx / len(train_loader),
#                     loss.item()))
#
# ''' RUN TEST '''
# def test(args, model, device, test_loader, epoch):
#     model.eval()
#     test_loss = 0
#     correct = 0
#     dfp, dfl = pd.DataFrame(), pd.DataFrame()
#     with torch.no_grad():
#         for data, target in test_loader:
#             data, target = data.to(device), target.to(device)
#             output = model(data)
#             test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
#             pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
#             correct += pred.eq(target.view_as(pred)).sum().item()
#             dfp = dfp.append(pd.DataFrame(F.softmax(output,dim=1).tolist()), sort=False, ignore_index=True)
#             dfl = dfl.append(pd.DataFrame(target.tolist()), sort=False, ignore_index=True)
#
#     y = [l[0] for l in dfl.values]
#     test_loss /= len(test_loader.dataset)
#     accuracy = float(100. * correct) / float(len(test_loader.dataset))
#     score = 100 / (1 + log_loss(y, dfp.values, eps=1E-15))
#     print(' [ {}\ttest\t{:.5f}\t{:.4f}\t{:.2f}%'.format(epoch, test_loss, score, accuracy))
#     return test_loss
#
# ''' RUN '''
# def tnet(args, train_raw, test_raw, do_cycles=True):
#     use_cuda = not args.no_cuda and torch.cuda.is_available()
#     device = torch.device("cuda" if use_cuda else "cpu")
#
#     kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
#
#     if do_cycles:
#         dir = 'data'
#         train_name = 'mnist_stack_train.pkl'
#         test_name = 'mnist_stack_test.pkl'
#         train_path = os.path.join(dir, train_name)
#         test_path = os.path.join(dir, test_name)
#
#         if not os.path.exists(dir):
#             print('[ creating directory %s' % dir)
#             os.mkdir(dir)
#         if os.path.exists(train_path):
#             print('[ loading %s' % train_path)
#             train_data = load_pkl(train_path)
#         else:
#             # train_data = CycleDataset(train_raw)
#             train_data = StackedDataset(train_raw)
#             print('[ saving %s' % train_path)
#             save_pkl(train_path, train_data)
#         if os.path.exists(test_path):
#             print('[ loading %s' % test_path)
#             test_data = load_pkl(test_path)
#         else:
#             # test_data = CycleDataset(test_raw)
#             test_data = StackedDataset(test_raw, train_data)
#             print('[ saving %s' % test_path)
#             save_pkl(test_path, test_data)
#     else:
#         train_data = RawDataset(train_raw)
#         test_data = RawDataset(test_raw, train_data)
#
#     # return train_data, test_data
#
#     train_loader = DataLoader(train_data, batch_size=args.batch, shuffle=True, **kwargs)
#     test_loader = DataLoader(test_data, batch_size=args.test_batch, shuffle=True, **kwargs)
#
#     model = StackedNet((28, 28)).to(device)
#     print(str(model)[:-2])
#
#     optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
#     scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose=True,
#                                                 factor=0.05, cooldown=5, patience=10)
#
#     print('[epoch\tmode\tloss\tscore\taccuracy')
#     for epoch in range(1, args.epochs + 1):
#         train(args, model, device, train_loader, optimizer, epoch)
#         scheduler.step(test(args, model, device, test_loader, epoch))
#
#     return model
#
#     # DATA = DATASETS[args.data]
#     # shape = SHAPE[args.data]
#
#     # trans = [transforms.ToTensor(), MaskTensor(masks, shape), transforms.Normalize(*stats)]
#     # transform = transforms.Compose(trans)
#
#     # train_data = DATA('../data', train=True, download=True, transform=transform)
#     # test_data = DATA('../data', train=False, transform=transform)
