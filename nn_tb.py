import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from alekseiml.deepl import *
from alekseiml.metrics import *

target = np.random.normal(0, 10, 200)
predicted = target - np.arange(-10, 10, 0.1)
xs = np.arange(-10, 10, 0.1)
loss = lambda pred, target: (target-pred)**2
plt.plot(xs, loss(predicted, target))
# plt.show()

del_loss = lambda pred, target: -2*(target-pred)
plt.plot(xs, del_loss(predicted, target))
# plt.show()


def derivative(fn, a, method='central', h=0.001):
    """Compute the difference formula for f'(a) with step size h.
    Parameters
    ----------
    fn : function
        Vectorized function of multiple variables
    a : number
        Compute derivative at x = a
    method : string
        Difference formula: 'forward', 'backward' or 'central'
    h : number
        Step size in difference formula
    Returns
    -------
    float
        Difference formula:
            central: f(a+h) - f(a-h))/2h
            forward: f(a+h) - f(a))/h
            backward: f(a) - f(a-h))/h
    """
    if hasattr(a, 'ndim'):
        if a.ndim > 2:
            raise ValueError(
                'Input for derivative() computation might have at max two dimensions (first batch, then input dimensions to objective).')
        if a.ndim == 2:
            return np.array([derivative(fn, p, method, h) for p in a])

    diffs = np.eye(len(a))*h
    params = np.stack([a]*len(a))
    if method == 'central':
        #return (fn(a + h) - fn(a - h))/(2*h)
        return np.array([(fn(p+d) - fn(p-d)) / (2 * h) for p, d in zip(params, diffs)])
    elif method == 'forward':
        #return (fn(a + h) - fn(a))/h
        return np.array([(fn(p+d) - fn(p)) / h for p, d in zip(params, diffs)])
    elif method == 'backward':
        #return (fn(a) - fn(a - h))/h
        return np.array([(fn(p) - fn(p-d)) / h for p, d in zip(params, diffs)])
    else:
        raise ValueError("Method must be 'central', 'forward' or 'backward'.")


# ys = np.arange(1, 10)
# for y in ys:
#     y_hat = y+np.random.normal(0, 1)
#     print("L(%s, %s)" % (round(y, 2), round(y_hat, 2)), round(loss(y, y_hat), 6), "dL/d[x]", derivative(lambda x: loss(x[0], x[1]), [y, y_hat]), "symbolic dL/d y_hat", round(del_loss(y_hat, y), 6))

x = np.array([150.13, 152.37, 184.44, 92.62, 273.99])
y = np.array([0.5470, 0.5420, 0.6135, 0.3812, 0.8988])

sigmoid = lambda x: 1/(1+np.exp(-x))
z = lambda w, a, b: w*a+b
act = lambda w, a, b: sigmoid(z(w, a, b))

print(act(1, 150.13, 0))
print(act(1, x[0], -150))
print(act(1, x[1], -150))

# forward
def model(x, W, B):
    out = x
    list_z = []
    list_a = []
    for w, b in zip(W, B):
        net_in = z(w, out, b)
        out = sigmoid(net_in)
        list_z.append(net_in)
        list_a.append(out)
    return out, list_z, list_a


W = np.array([1, 1, 1])
B = np.array([0, 0, 0])

y_hat, list_z, list_a = model(x, W, B)
print(y_hat)

y2_hat, _, _ = model(x, np.array([0.1, 1, 1]), np.array([-20, -0.5, -0.5]))
print(y2_hat)


def cost(y_hat, y):
    return np.sum((y_hat-y)**2)/2


print(cost(y_hat, y))


# Backward
eta = 0.5
print(y_hat-y)
print(y_hat*(1-y_hat))
print(list_z[-1])


W_new = W-eta*np.array([1/len(z)*np.sum(-(y-y_hat)*(y_hat*(1-y_hat))*z) for z in list_z])

# data gen

path_data = "data_algorithm_performances_regression.csv"  # Path to store our generated data in
n_samples = 2000  # Number of samples to generate

performance_intercept = 72  #
performance_slope = 4.9
training_time = np.random.uniform(100, 300, n_samples)
noisy_training_time = training_time + np.random.normal(0, 10, n_samples)
performances = performance_intercept + performance_slope * training_time + np.random.normal(0, training_time/100)
def normalize(X, upper_bound=1):
    X /= np.max(np.abs(X), axis=0)
    X *= (upper_bound/X.max())
    return X
performances = normalize(performances)
data = pd.DataFrame(np.stack([noisy_training_time, performances], axis=1), columns=["training_time", "performance"])
data.to_csv(path_data, index=False)
data.head()

data_regression = data.to_numpy()
ix_split = 500
train_x = data_regression[:ix_split,0].reshape(-1, 1)
train_y = data_regression[:ix_split,1].reshape(-1, 1)
test_x = data_regression[ix_split:,0].reshape(-1, 1)
test_y = data_regression[ix_split:,1].reshape(-1, 1)

# plt.plot(train_x, train_y)
# plt.show()

print(train_x[:5])
print(train_y[:5])


x_train = train_x
y_train = train_y
B_new = np.copy(B)
W_new = np.copy(W)
for it in range(500):
    y_hat, list_z, list_a = model(x_train, W_new, B_new)
    W_new = W_new-eta*np.array([1/len(z)*np.sum(-(y_train-y_hat)*(y_hat*(1-y_hat))*z) for z in list_z])
    print("cost", cost(y_hat, y_train))
    print(y_hat[:3], y_train[:3])
    print("updated W", W_new)


class MyModel(Module):
    def __init__(self):  # Overall: R^1 -> R^5 -> R^3 -> R^1
        self._layer1 = LinearLayer(1, 5)  # A linear function from R^1 -> R^5
        self._act1 = SigmoidLayer(5)  # A non-linear function acting on R^5
        self._layer2 = LinearLayer(5, 3)
        self._act2 = SigmoidLayer(3)
        self._layer3 = LinearLayer(3, 1)

    def forward(self, input):
        h = self._act1(self._layer1.forward(input))  # h is now in R^5
        h = self._act2(self._layer2.forward(h))  # h is now in R^3
        return self._layer3(h)

    def layers(self):
        return [self._layer1, self._layer2, self._layer3]


#mm = lambda x: lay2(np.maximum(lay1(x), 0))
mm = MyModel()
print(mm.forward(train_x[:5]))
delta = mm(train_x) - train_y


learning_rate = 0.01
optimizer = SGDOptimizer(mm.layers(), lr=learning_rate)
for epoch in range(1000):
    out = mm(train_x)
    cost, dY_hat = compute_cost(train_y, out)
    # print("cost, dY_hat", cost, dY_hat[:3])

    dlayer3 = mm._layer3.backward(dY_hat)
    dact2 = mm._act2.backward(dlayer3)
    dlayer2 = mm._layer2.backward(dact2)
    dact1 = mm._act1.backward(dlayer2)
    dlayer1 = mm._layer1.backward(dact1)

    # Optimizer: update based on all local gradients
    optimizer.step()

