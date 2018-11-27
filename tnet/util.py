from torchvision import datasets, transforms
from multiprocessing import Pool
from collections import Iterable
from functools import partial
import numpy.linalg as la
from ripser import ripser
import dionysus as dio
from tqdm import tqdm
import pickle as pkl
import numpy as np
import math, torch

SHAPE = {'mnist' : (28, 28),
        'cifar' : (32, 32, 3)}

DATA = {'mnist' : datasets.MNIST,
        'cifar' : datasets.CIFAR10}

def pmap(fun, x, *args):
    pool = Pool()
    f = partial(fun, *args)
    try:
        y = pool.map(f, x)
    except KeyboardInterrupt as e:
        print(e)
        pool.close()
        pool.join()
        sys.exit()
    pool.close()
    pool.join()
    return y

def circle(n=20, r=1., uniform=False, noise=0.1):
    t = np.linspace(0, 1, n, False) if uniform else np.random.rand(n)
    e = r * (1 + noise * (2 * np.random.rand(n) - 1)) if noise else r
    return np.array([e*np.cos(2 * np.pi * t),
                    e*np.sin(2*np.pi*t)]).T.astype(np.float32)

def double_circle(n=50, r=(1., 0.7), *args, **kwargs):
    p1 = circle(int(n * r[0] / sum(r)), r[0], *args, **kwargs)
    p2 = circle(int(n * r[1] / sum(r)), r[1], *args, **kwargs)
    return np.vstack([p1 - np.array([r[0], 0.]),
                    p2 + np.array([r[1], 0.])])

def torus(n=1000, R=1., r=0.3):
    t = 2*math.pi * np.random.rand(2, n)
    x = (R + r * np.cos(t[1])) * np.cos(t[0])
    y = (R + r * np.cos(t[1])) * np.sin(t[0])
    z = r * np.sin(t[1])
    return np.vstack([x, y, z]).T

def load_data(dset='mnist', train=True):
    data = DATA[dset]('../data', train=train, transform=transforms.ToTensor())
    X = torch.as_tensor(data.train_data if train else data.test_data, dtype=torch.float)
    y = data.train_labels if train else data.test_labels

    # if dset != 'cifar':
    #     X, y = X.numpy(), y.numpy()
    return X / X.max(), y
    # return X.reshape(-1, *SHAPE[dset]) / float(X.max()), y

def group_data(X, y, n=-1, k=1):
    CLASS = np.unique(y)
    D = {c : X[filter(lambda i: y[i] == c, range(len(y)))][:n] for c in CLASS}
    return {c : np.array_split(D[c], k) for c in CLASS} if k != 1 else D
    # return D

def data_coords(x, dset='mnist'):
    s = SHAPE[dset]
    return get_coords(x.T.reshape(dset[1], dset[2], dset[0]))

def persist(M, prime=11, thresh=2.):
    R = ripser(M, 1, thresh, prime, True, True)
    d, c = R['dgms'][1], R['cocycles'][1]
    I = sorted(range(len(d)), key=lambda i: d[i,1] - d[i,0], reverse=True)
    R['dgms'][1], R['cocycles'][1] = d[I], [c[i] for i in I]
    return R

def normalize(x):
    return (x - x.min()) / ((x.max() - x.min()) if x.max() > 0 else 1.)

def nbrs(x):
    X = x.reshape(-1, 3)
    d = [[la.norm(X[i] - X[j]) for j in range(len(X))] for i in range(len(X))]
    return np.array([sorted(range(len(X)), key=lambda j: d[i][j]) for i in range(len(X))])

def dgm_array(dgm):
    return np.array([[pt.birth, pt.death] for pt in dgm])

def plot_dgm(ax, D):
    ax.cla()
    dgms = map(dgm_array, D)
    t = max(map(lambda x: x.max() if len(x) > 0 and x.max() < np.Inf else 0, dgms))
    ax.plot([0,t], [0,t], c='black', alpha=0.3)
    for dgm in dgms:
        if len(dgm) > 0:
            ax.scatter(dgm[:,0], dgm[:,1], s=5, alpha=0.5)
    return dgms

def save_pkl(fname, x):
    with open(fname, 'w') as f:
        pkl.dump(x, f)
    return fname

def load_pkl(fname):
    with open(fname, 'r') as f:
        x = pkl.load(f)
    return x

# def load_mnist(k=10, c=0):
#     X, y = load_data()
#     # return group_data(X, y, k)[c].T
#     # return group_data(X, y, k)[c][0].T
#     S = group_data(X, y, k)
#     return S[c][0].T
#     # P = S[c][0].T
#     # I = filter(lambda x: sum(P[x]) > 0, range(len(P)))
#     # return P[I], I
#
# def distance_matrix(X, axis=2, **kwargs):
#     return la.norm(X[np.newaxis] - X[:, np.newaxis], axis=axis, **kwargs)
#
def get_cycle(R, H, p):
    c = np.array([list(R[s.index]) for s in H[H.pair(p.data)]])
    M = np.zeros(28*28, dtype=int)
    M[np.unique(c)] = p.death - p.birth
    return M.reshape(28, 28)

def get_cycles(img):
    x = img.reshape(28, 28)
    R = dio.fill_freudenthal(x)
    H = dio.homology_persistence(R)
    dgms = dio.init_diagrams(H, R)
    n = len(dgms[1])
    if n > 0:
        return sum(get_cycle(R, H, p) for p in dgms[1])
    return np.zeros((28,28), dtype=int)
#
# def build_filt(img):
#     x = img.reshape(28, 28)
#     F = dio.fill_freudenthal(x)
#     return F
