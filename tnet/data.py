from torch.utils.data import Dataset
from functools import partial
import circular as coords
import dionysus as dio
from tqdm import tqdm
import torch, sys, os
from util import *
import numpy as np

# DATASETS = {'mnist' : MNIST, 'cifar' : CIFAR10}
# SHAPE = {'mnist' : (28, 28), 'cifar' : (3, 32, 32)}
# CLASS = {'mnist' : 10, 'cifar' : 10}

def circular(prime, x):
    x = x.numpy()
    x = x / float(x.max())
    print(x)
    shape = x.shape
    R = dio.fill_freudenthal(x)
    H = dio.cohomology_persistence(R, prime)
    D = dio.init_diagrams(H, R)
    p = max(D[1], key=lambda pt: pt.death - pt.birth)
    if len(D[1]) > 0:
        C = H.cocycle(p.data)
        F = dio.Filtration([s for s in R if s.data <= (p.death + p.birth)/2])
        v = coords.smooth(R, F, C, prime)
        return normalize(v.reshape(*shape))
    else:
        return np.zeros(shape, dtype=float)

def cycle(x):
    x = x.numpy()
    x = x / float(x.max())
    shape = x.shape
    R = dio.fill_freudenthal(x)
    H = dio.homology_persistence(R)
    D = dio.init_diagrams(H, R)
    M = np.zeros(shape[0]*shape[1], dtype=int)
    if len(D[1]) > 0:
        p = max(D[1], key=lambda pt: pt.death - pt.birth)
        c = np.array([list(R[s.index]) for s in H[H.pair(p.data)]])
        M[np.unique(c)] = p.death - p.birth
    return M.reshape(*shape)


def get_cycle(R, H, D, pt):
    return np.array([list(R[s.index]) for s in H[H.pair(p.data)]])

def get_cycles(x, k=3):
    x = x.numpy()
    x = x / float(x.max())
    shape = x.shape
    R = dio.fill_freudenthal(x)
    H = dio.homology_persistence(R)
    D = dio.init_diagrams(H, R)
    Ms = np.stack([np.zeros(shape[0]*shape[1], dtype=float) for _ in range(k)], axis=0)
    ps = sorted(D[1], key=lambda pt: pt.death - pt.birth, reverse=True)
    for i in range(k):
        if i < len(ps):
            c = np.array([list(R[s.index]) for s in H[H.pair(ps[i].data)]])
            Ms[i, np.unique(c)] = ps[i].death - ps[i].birth
    return np.concatenate([x[np.newaxis], Ms.reshape(k, *shape)], axis=0)

class CycleDataset(Dataset):
    def __init__(self, data, train=None, prime=2):
        super(Dataset, self).__init__()
        raw, self.labels = data
        # self.data = map(cycle, tqdm(raw))
        C = np.stack(map(cycle, tqdm(raw)), axis=0)
        self.mean = train.mean if train != None else C.mean()
        self.std = train.std if train != None else C.std()
        transform = transforms.Normalize((self.mean,),(self.std,))
        X = torch.as_tensor(C[:, np.newaxis, ...], dtype=torch.float)
        self.data = torch.stack(map(transform, X), dim=0)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, i):
        return self.data[i], self.labels[i]

class StackedDataset(Dataset):
    def __init__(self, data, train=None, k=3, prime=2):
        super(Dataset, self).__init__()
        raw, self.labels = data
        # C = np.stack(map(lambda x: get_cycles(x, k), tqdm(raw)), axis=0)
        # self.mean = train.mean if train != None else tuple(x.mean() for x in C)
        # self.std = train.std if train != None else tuple(x.std() for x in C)
        # transform = transforms.Normalize(self.mean, self.std)
        # X = torch.as_tensor(C[:, np.newaxis, ...], dtype=torch.float)
        # self.data = torch.stack(map(transform, X), dim=0)
        C = np.stack(map(lambda x: get_cycles(x, k), tqdm(raw)), axis=0)
        self.mean = train.mean if train != None else tuple(x.mean() for x in C)
        self.std = train.std if train != None else tuple(x.std() for x in C)
        transform = transforms.Normalize(self.mean, self.std)
        X = torch.as_tensor(C, dtype=torch.float)
        self.data = torch.stack(map(transform, X), dim=0)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, i):
        return self.data[i], self.labels[i]

class RawDataset(Dataset):
    def __init__(self, data, train=None):
        super(Dataset, self).__init__()
        raw, self.labels = data
        self.mean = train.mean if train != None else raw.mean()
        self.std = train.std if train != None else raw.std()
        transform = transforms.Normalize((self.mean,),(self.std,))
        X = torch.FloatTensor(raw.numpy()[:, np.newaxis, ...])
        self.data = torch.stack(map(transform, X), dim=0)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, i):
        return self.data[i], self.labels[i]

DATASETS = {'raw' : RawDataset, 'stack' : StackedDataset, 'cycle' : CycleDataset}

# ''' DEFAULT '''
# class MaskTensor(object):
#     def __init__(self, masks, shape):
#         super(MaskTensor, self).__init__()
#         self.n = len(masks)
#         if self.n > 0:
#             self.masks = [torch.from_numpy(m).float() for m in masks]
#         else:
#             self.masks = None
#         self.shape = shape
#     def __call__(self, x):
#         if self.masks == None:
#             return x
#         X = torch.stack([m * x for m in self.masks], 0)
#         return X.view(self.shape[0] * self.n, self.shape[1], self.shape[2])
