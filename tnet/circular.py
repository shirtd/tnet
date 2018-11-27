from scipy.sparse.linalg import lsqr
from scipy.sparse import csc_matrix
import numpy as np
from util import *

def get_bdy(f, z, prime=11):
    data, row, col = [], [], []
    for i,s in enumerate(f):
        if s.dimension() == 1:
            for isb,sb in enumerate(s.boundary()):
                data.append(1. if isb % 2 == 0 else -1.)
                row.append(i)
                col.append(f.index(sb))
    row_max = max(row) # x.index <= row_max condition below projects the cocycle to the filtration
    z_data = [x.element if x.element < prime/2 else x.element - prime for x in z if x.index <= row_max]
    z_row  = [x.index for x in z if x.index <= row_max]
    z_col  = [0 for x in z if x.index <= row_max]
    dim = max(row_max,max(col)) + 1
    D = csc_matrix((np.array(data), (np.array(row), np.array(col))), shape=(dim, dim))
    z = csc_matrix((z_data, (z_row, z_col)), shape=(dim, 1)).toarray()
    return D, z

def get_coords(R, D, z, prime=11, tol=1e-10):
    solution = lsqr(D, z, atol=tol, btol=tol, show=False)
    v = [s[0] for s in R if s.dimension() == 0]
    max_vrt, min_vrt = map(lambda f: f(v), (max, min))
    vertex_values = {i : 0. for i in range(min_vrt, max_vrt + 1)}
    # vertex_values = [0. for _ in range(min_vrt, max_vrt + 1)]
    for i, x in enumerate(solution[0]):
        if R[i].dimension() == 0:
            vertex_values[R[i][0]] = x
    return np.array([vertex_values[i] for i in range(min_vrt, max_vrt + 1)])

def smooth(R, f, z, prime=11, tol=1e-10):
    D, z = get_bdy(f, z, prime)
    return get_coords(R, D, z)

# def get_cocycle(R, H, pt):
#     cocycle = H.cocycle(pt.data)
#     f_restricted = dio.Filtration([s for s in R if s.data <= (pt.death + pt.birth)/2])
#     return f_restricted, cocycle
#
# def get_cocycles(R, H, dgms, cycles=0, dim=1):
#     pts = sorted(dgms[dim], key=lambda pt: pt.death - pt.birth, reverse=True)
#     if isinstance(cycles, int):
#         return get_cocycle(R, H, pts[cycles])
#     if len(pts) < len(cycles):
#         cycles = range(len(pts))
#     return zip(*[get_cocycle(R, H, pts[c]) for c in cycles])
#
# def get_coords(x, prime=11, cycles=0, shape=SHAPE['mnist']):
#     R = dio.fill_freudenthal(x)
#     H = dio.cohomology_persistence(R, prime, True)
#     D = dio.init_diagrams(H, R)
#     F, C = get_cocycles(R, H, D, cycles)
#     # return R, F, C, D
#     fun1 = lambda a: a.reshape(-1, *reversed(shape)).sum(axis=0)
#     fun2 = lambda a: np.swapaxes(np.stack(a, axis=0).T, 0, 1)
#     fun = lambda a: fun2(fun1(a))
#     if isinstance(cycles, int):
#         return normalize(fun(coords.smooth(R, F, C, prime))), D
#     return normalize(sum(fun(coords.smooth(R, f, c, prime)) for f, c in zip(F, C))), D

def cocycle_reduce(R, f, cocycle, prime=11):
    S = {tuple([v for v in s]) : i for i, s in enumerate(R) if all(v < 784 for v in s)}
    cmap = {tuple(v for v in R[a.index]) : a.element for a in cocycle if all(v < 784 for v in R[a.index])}
    for a in cocycle:
        s = tuple(v % 784 for v in R[a.index])
        rs = tuple(reversed(s))
        if s in cmap:
            cmap[s] = (cmap[s] + a.element) % prime
        elif rs in cmap:
            cmap[rs] = (cmap[s] - a.element) % prime
        else:
            cmap[s] = a.element
    # fmap = {i : S[tuple(np.unique([v % 784 for v in s]))] for i, s in enumerate(R)}
    # c0 = [make_entry(fmap[a.index], a.element) for a in cocycle]
    c0 = [make_entry(k, v) for k,v in cmap.iteritems()]
    f0 = dio.Filtration([s for s in f if all(v < 784 for v in s)])
    v = smooth(f0, c0, prime)
    return v, f0, c0, cmap

# plt.imshow(M)
# G = np.array([[i, j] for j in range(28) for i in range(28)])
# colors = cm.rainbow(np.linspace(0, 1, prime))
# for k, v in cmap.iteritems():
#     e = G[list(k)]
#     ax.plot(e[:,0], e[:,1], c=colors[v], alpha=0.5)
# f_restricted = dio.Filtration([s for s in R if s.data <= (pt.death + pt.birth)/2])
