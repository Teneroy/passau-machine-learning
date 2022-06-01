from numpy import np


def r2_score(y, y_pred):
    return 1 - np.mean((y - y_pred) ** 2) / y.var()

