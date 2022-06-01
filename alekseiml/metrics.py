from numpy import np


def r2_score(y, y_pred):
    return 1 - np.mean((y - y_pred) ** 2) / y.var()


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


def euclidean_distance(A, B):
    """
    Description:
    computes euclidean distance between matrices A and B.
    E. g. C[2,15] is distance between point 2 from A (A[2]) matrix and point 15 from matrix B (B[15])

    Args:
    A (numpy.ndarray): Matrix size N1:D
    B (numpy.ndarray): Matrix size N2:D

    Returns:
    numpy.ndarray: Matrix size N1:N2
    """

    A_square = np.reshape(np.sum(A * A, axis=1), (A.shape[0], 1))
    B_square = np.reshape(np.sum(B * B, axis=1), (1, B.shape[0]))
    AB = A @ B.T

    C = -2 * AB + B_square + A_square

    return np.sqrt(C)