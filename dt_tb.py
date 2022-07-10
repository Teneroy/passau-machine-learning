from alekseiml.dtree import *
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import numpy.linalg as la
from alekseiml.dtree import _entropy, _misclass, _gini
from alekseiml.datasets import *
from alekseiml.metrics import acc, err_mis


np.set_printoptions(precision=3)

# csv-file has no header, so we define it manually
col_names = ['price_buy', 'price_main', 'n_doors', 'n_persons', 'lug_boot', 'safety', 'recommendation']
df = pd.read_csv("./data/car/data/car.data", header=None, names=col_names)

# All attributes are categorical - a mix of strings and integers.
# We simply map the categorical values of each attribute to a set of distinct integers
ai2an_map = col_names
ai2aiv2aivn_map = []
enc_cols = []
for col in df.columns:
    df[col] = df[col].astype('category')
    a = np.array(df[col].cat.codes.values).reshape((-1, 1))
    enc_cols.append(a)
    ai2aiv2aivn_map.append(list(df[col].cat.categories.values))

# Get the data as numpy 2d-matrix (n_samples, n_features)
dataset = np.hstack(enc_cols)
X, y = dataset[:, :6], dataset[:, 6]
print(X.shape, y.shape)

tree = DecisionTreeID3()
tree.fit(X, y, verbose=0)
tree.print_tree(ai2an_map, ai2aiv2aivn_map)


print ("According to the attributes %s"%(col_names[:-1]))
print ("Should i buy the car %s?"%(dataset[52,0:6]))
print ("The car is %s (in truth it is %s)"%(tree.predict(dataset[[52],0:6]),dataset[52,6]))


impurity_measures = [_gini, _entropy, _misclass]
k = 10

folds = getKFolds(X.shape[0], k)
# folds = getBootstrapFolds(X.shape[0], k, train_fraction=0.9)

for imp in impurity_measures:
    err_tr = 0.
    err_te = 0.
    for i in range(k):
        idx_tr, idx_te = folds[i]

        _X_tr = X[idx_tr]
        _y_tr = y[idx_tr]
        _X_te = X[idx_te]
        _y_te = y[idx_te]

        DecisionTreeID3(criterion=imp)
        tree.fit(_X_tr, _y_tr, verbose=0)

        y_tr_p = tree.predict(_X_tr)
        y_te_p = tree.predict(_X_te)
        err_tr += err_mis(_y_tr, y_tr_p)
        err_te += err_mis(_y_te, y_te_p)

    print("%s: Average training error %f;Average test error %f" % (imp, err_tr / k, err_te / k))