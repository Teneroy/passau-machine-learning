from alekseiml.classification import SVM
import numpy as np
import pandas as pd
import random as rd

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00229/Skin_NonSkin.txt"
data = pd.read_csv(url, header = None, sep='\t')

np.random.seed(1)

msk = np.random.rand(len(data)) < 0.8
data[msk].to_csv('train.csv', header=False, index=False)
data[~msk].to_csv('test.csv', header=False, index=False)

colnames = ['x0', 'x1', 'x2', 't']
train = pd.read_csv('train.csv', names=colnames)
test = pd.read_csv('test.csv', names=colnames)

prepare_target = lambda a : 1. if a == 1 else -1.

train = train.sample(frac=1)
train_X = train.values[:, 0:3]
train_y = [prepare_target(record) for record in train.values[:, 3:4]]

test_X = test.values[:, 0:3]
test_y = [prepare_target(record) for record in test.values[:, 3:4]]

svm = SVM(1, 3)
svm2 = SVM(1, 3)


np.random.seed(0)

l, acc, tp, tn, w = svm.fit()
print("l: ", l, ", acc: ", acc, ", tp: ", tp, ", tn: ", tn)

svm2.learn(train_X, train_y)
l, acc, tp, tn, w = svm2.test(test_X, test_y)
print("l: ", l, ", acc: ", acc, ", tp: ", tp, ", tn: ", tn)
