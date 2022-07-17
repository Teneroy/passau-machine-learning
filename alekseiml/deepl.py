import pickle

import numpy as np


class Module:
    def forward(self, input):
        return input

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Tensor:
    def __init__(self, np_val: np.array):
        self._value = np_val
        self._grad = None

    def np(self):
        return self._value


class LinearLayer(Module):
    def __init__(self, size_input: int, size_output: int):
        self.weight = Tensor(np.random.normal(0, 1, (size_input, size_output)))
        self.bias = Tensor(np.random.normal(0, 1, (size_output,)))
        self._grad = None

    def forward(self, input):
        self._prev_input = input
        self._act_input = input.dot(self.weight.np()) + self.bias.np()
        return self._act_input

    def backward(self, upstream_grad):
        # derivative of Cost w.r.t W
        # (a1.T).dot(delta3)
        self.d_weight = self._prev_input.T.dot(upstream_grad)
        self.weight._grad = self.d_weight

        # derivative of Cost w.r.t b, sum across rows
        self.d_bias = np.sum(upstream_grad, axis=0)
        self.bias._grad = self.d_bias

        # derivative of Cost w.r.t _prev_input
        self.d_prev_input = upstream_grad.dot(self.weight.np().T)
        self._grad = self.d_prev_input

        return self.d_prev_input


class SigmoidLayer(Module):
    def __init__(self, shape):
        self._result = np.zeros(shape)

    def forward(self, input):
        self._result = 1 / (1 + np.exp(-input))
        return self._result

    def backward(self, upstream_grad):
        self.d_result = upstream_grad * self._result * (1 - self._result)
        self._grad = self.d_result
        return self.d_result


class SGDOptimizer:
    def __init__(self, layers: [], lr: float):
        self._layers = layers
        self._learning_rate = lr

    def step(self):
        # self._mm._layer1.weight._value += -learning_rate * mm._layer1.weight._grad
        # self._mm._layer1.bias._value += -learning_rate * mm._layer1.bias._grad
        # self._mm._layer2.weight._value += -learning_rate * mm._layer2.weight._grad
        # self._mm._layer2.bias._value += -learning_rate * mm._layer2.bias._grad
        # self._mm._layer3.weight._value += -learning_rate * mm._layer3.weight._grad
        # self._mm._layer3.bias._value += -learning_rate * mm._layer3.bias._grad
        for layer in self._layers:
            layer.weight._value += -self._learning_rate * layer.weight._grad
            layer.bias._value += -self._learning_rate * layer.bias._grad
