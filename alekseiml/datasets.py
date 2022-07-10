import numpy as np


def getKFolds(N, k):
    fold_size = N//k
    idx = np.random.permutation(np.arange(fold_size*k))

    splits = np.split(idx, k)
    folds = []
    for i in range(k):
        te = splits[i]
        tr_si = np.setdiff1d(np.arange(k), i)
        tr = np.concatenate([splits[si] for si in tr_si])
        folds.append((tr.astype(np.int), te.astype(np.int)))
    return folds


def getBootstrapFolds(N, k, train_fraction=0.8):
    folds = []
    for i in range(k):
        idx = np.random.permutation(np.arange(N))
        #m = int(N * train_fraction)
        tr = np.random.choice(idx, size=N, replace=True) # When we draw with replacement, the test set will be larger
        te = np.setdiff1d(idx, tr)
        folds.append((tr.astype(np.int), te.astype(np.int)))
    return folds