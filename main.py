# from tnet.tnet import tnet, circular
# import matplotlib.pyplot as plt
# import tnet.circular as coords
from tnet.args import parser
from tnet.util import *
# import dionysus as dio
from tnet import tnet

# plt.ion()
# fig, ax = plt.subplots(4, 8, sharex=True, sharey=True, figsize=(12, 7))
# plt.tight_layout(); plt.subplots_adjust(wspace=0, hspace=0)

if __name__ == '__main__':
    dset = 'mnist'
    args = parser.parse_args()
    shape = SHAPE[dset]
    train = load_data(dset, train=True)
    test = load_data(dset, train=False)
    # train = train[0][:1000], train[1][:1000]
    # test = test[0][:100], test[1][:100]
    res = tnet(args, train, test)

    # from tnet.data import *
    # k = 3
    # raw, label = train[0][:100], train[1][:100]
    # C = np.stack(map(lambda x: get_cycles(x, k), tqdm(raw)), axis=0)
    # mean, std = tuple(x.mean() for x in C), tuple(x.std() for x in C)
    # transform = transforms.Normalize(mean, std)
    # X = torch.as_tensor(C, dtype=torch.float)
    # data = torch.stack(map(transform, X), dim=0)

    # for j in range(8):
    #     for i in range(k+1):
    #         ax[i, j].imshow(X[j,i].numpy())

    # x = train[0][0]
    # ax[0].imshow(x)
    # x = x.numpy()
    # x = x / float(x.max())
    # shape = x.shape
    # R = dio.fill_freudenthal(x)
    # H = dio.homology_persistence(R, 11)
    # D = dio.init_diagrams(H, R)
    # p = max(D[1], key=lambda pt: pt.death - pt.birth)
    # M = get_cycle(R, H, p)
    # ax[1].imshow(M)
    # c = np.array([list(R[s.index]) for s in H[H.pair(p.data)]])
    # C = H.cocycle(p.data)

    # F = dio.Filtration([s for s in R if s.data <= (p.death + p.birth)/2])
