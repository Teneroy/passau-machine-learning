def standard_deviation(X):
    cur_sum = 0
    cur_sq_sum = 0
    num_vals = 0
    for x in X:
        cur_sum += x
        cur_sq_sum += x**2
        num_vals += 1
    return np.sqrt(cur_sq_sum/num_vals-(cur_sum/num_vals)**2)


def sample_standard_deviation(X):
    return np.sqrt(np.sum(X**2)/len(X)-(np.sum(X)/len(X))**2)


# root mean squared error
def rmse(X_1, X_2):
    return np.sqrt(np.sum((X_1-X_2)**2))