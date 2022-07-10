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


def padArray(var, pad1, pad2=None):
    '''Pad array with 0s
    Args:
        var (ndarray): 2d or 3d ndarray. Padding is done on the first 2 dimensions.
        pad1 (int): number of columns/rows to pad at left/top edges.
    Keyword Args:
        pad2 (int): number of columns/rows to pad at right/bottom edges.
            If None, same as <pad1>.
    Returns:
        var_pad (ndarray): 2d or 3d ndarray with 0s padded along the first 2
            dimensions.
    '''
    if pad2 is None:
        pad2 = pad1
    if pad1+pad2 == 0:
        return var
    var_pad = np.zeros(tuple(pad1+pad2+np.array(var.shape[:2])) + var.shape[2:])
    var_pad[pad1:-pad2, pad1:-pad2] = var
    return var_pad


def asStride(arr, sub_shape, stride):
    '''Get a strided sub-matrices view of an ndarray.
    Args:
        arr (ndarray): input array of rank 2 or 3, with shape (m1, n1) or (m1, n1, c).
        sub_shape (tuple): window size: (m2, n2).
        stride (int): stride of windows in both y- and x- dimensions.
    Returns:
        subs (view): strided window view.
    See also skimage.util.shape.view_as_windows()
    '''
    s0, s1 = arr.strides[:2]
    m1, n1 = arr.shape[:2]
    m2, n2 = sub_shape[:2]
    view_shape = (1+(m1-m2)//stride, 1+(n1-n2)//stride, m2, n2)+arr.shape[2:]
    strides = (stride*s0, stride*s1, s0, s1)+arr.strides[2:]
    subs = np.lib.stride_tricks.as_strided(
        arr, view_shape, strides=strides, writeable=False)
    return subs


def conv3D3(var, kernel, stride=1, pad=0):
    '''3D convolution by strided view.
    Args:
        var (ndarray): 2d or 3d array to convolve along the first 2 dimensions.
        kernel (ndarray): 2d or 3d kernel to convolve. If <var> is 3d and <kernel>
            is 2d, create a dummy dimension to be the 3rd dimension in kernel.
    Keyword Args:
        stride (int): stride along the 1st 2 dimensions. Default to 1.
        pad (int): number of columns/rows to pad at edges.
    Returns:
        conv (ndarray): convolution result.
    '''
    kernel = checkShape(var, kernel)
    if pad > 0:
        var_pad = padArray(var, pad, pad)
    else:
        var_pad = var
    view = asStride(var_pad, kernel.shape, stride)
    if np.ndim(kernel) == 2:
        conv = np.sum(view*kernel, axis=(2, 3))
    else:
        conv = np.sum(view*kernel, axis=(2, 3, 4))
    return conv


def poolingOverlap(mat, f, stride=None, method='max', pad=False,
                   return_max_pos=False):
    '''Overlapping pooling on 2D or 3D data.

    Args:
        mat (ndarray): input array to do pooling on the first 2 dimensions.
        f (int): pooling kernel size.
    Keyword Args:
        stride (int or None): stride in row/column. If None, same as <f>,
            i.e. non-overlapping pooling.
        method (str): 'max for max-pooling,
                      'mean' for average-pooling.
        pad (bool): pad <mat> or not. If true, pad <mat> at the end in
               y-axis with (f-n%f) number of nans, if not evenly divisible,
               similar for the x-axis.
        return_max_pos (bool): whether to return an array recording the locations
            of the maxima if <method>=='max'. This could be used to back-propagate
            the errors in a network.
    Returns:
        result (ndarray): pooled array.

    See also unpooling().
    '''
    m, n = mat.shape[:2]
    if stride is None:
        stride = f
    _ceil = lambda x, y: x//y + 1

    if pad:
        ny = _ceil(m, stride)
        nx = _ceil(n, stride)
        size = ((ny-1)*stride+f, (nx-1)*stride+f) + mat.shape[2:]
        mat_pad = np.full(size, 0)
        mat_pad[:m, :n, ...] = mat
    else:
        mat_pad = mat[:(m-f)//stride*stride+f, :(n-f)//stride*stride+f, ...]

    view = asStride(mat_pad, (f, f), stride)
    if method == 'max':
        result = np.nanmax(view, axis=(2, 3), keepdims=return_max_pos)
    else:
        result = np.nanmean(view, axis=(2, 3), keepdims=return_max_pos)

    if return_max_pos:
        pos = np.where(result == view, 1, 0)
        result = np.squeeze(result)
        return result, pos
    else:
        return result


def ReLU(x):
    return max(0.0, x)


class ConvLayer(object):
    def __init__(self, f, pad, stride, nc_in, nc, learning_rate, af=None, lam=0.01,
                 clipvalue=0.5):
        '''Convolutional layer

        Args:
            f (int): kernel size for height and width.
            pad (int): padding on each edge.
            stride (int): convolution stride.
            nc_in (int): number of channels from input layer.
            nc (int): number of channels this layer.
            learning_rate (float): initial learning rate.
        Keyword Args:
            af (callable): activation function. Default to ReLU.
            lam (float): regularization parameter.
            clipvalue (float): clip gradients within [-clipvalue, clipvalue]
                during back-propagation.

        The layer has <nc> channels/filters. Each filter has shape (f, f, nc_in).
        The <nc> filters are saved in a list `self.filters`, therefore `self.filters[0]`
        corresponds to the 1st filter.

        Bias is saved in `self.biases`, which is a 1d array of length <nc>.
        '''

        self.type = 'conv'
        self.f = f
        self.pad = pad
        self.stride = stride
        self.lr = learning_rate
        self.nc_in = nc_in
        self.nc = nc
        if af is None:
            self.af = ReLU
        else:
            self.af = af
        self.lam = lam
        self.clipvalue = clipvalue

        self.init()

    def init(self):
        '''Initialize weights

        Default to use HE initialization:
            w ~ N(0, std)
            std = \sqrt{2 / n}
        where n is the number of inputs
        '''
        np.random.seed(100)
        std = np.sqrt(2 / self.f ** 2 / self.nc_in)
        self.filters = np.array([
            np.random.normal(0, scale=std, size=[self.f, self.f, self.nc_in])
            for i in range(self.nc)])
        self.biases = np.random.normal(0, std, size=self.nc)

    @property
    def n_params(self):
        '''Number of parameters in layer'''
        n_filters = self.filters.size
        n_biases = self.nc
        return n_filters + n_biases

    def forward(self, x):
        '''Forward pass of a single image input'''

        x = force3D(x)
        x = padArray(x, self.pad, self.pad)
        # input size:
        ny, nx, nc = x.shape
        # output size:
        oy = (ny + 2 * self.pad - self.f) // self.stride + 1
        ox = (nx + 2 * self.pad - self.f) // self.stride + 1
        oc = self.nc
        weight_sums = np.zeros([oy, ox, oc])

        # loop through filters
        for ii in range(oc):
            slabii = conv3D3(x, self.filters[ii], stride=self.stride, pad=0)
            weight_sums[:, :, ii] = slabii[:, :]

        # add bias
        weight_sums = weight_sums + self.biases
        # activate func
        activations = self.af(weight_sums)

        return weight_sums, activations

    def backPropError(self, delta_in, z):
        '''Back-propagate errors

        Args:
            delta_in (ndarray): delta from the next layer in the network.
            z (ndarray): weighted sum of the current layer.
        Returns:
            result (ndarray): delta of the current layer.

        The theoretical equation for error back-propagation is:

            \delta^{(l)} = \delta^{(l+1)} \bigotimes_f Rot(W^{(l+1)}) \bigodot f'(z^{(l)})

        where:
            \delta^{(l)} : error of layer l, defined as \partial J / \partial z^{(l)}.
            \bigotimes_f : convolution in full mode.
            Rot() : is rotating the filter by 180 degrees, i.e. a kernel flip.
            W^{(l+1)} : weights of layer l+1.
            \bigodot : Hadamard (elementwise) product.
            f() : activation function of layer l.
            z^{(l)} : weighted sum in layer l.

        Computation in practice is more complicated than the above equation.
        '''

        # number of channels of input to layer l weights
        nc_pre = z.shape[-1]
        # number of channels of output from layer l weights
        nc_next = delta_in.shape[-1]

        result = np.zeros_like(z)
        # loop through channels in layer l
        for ii in range(nc_next):
            # flip the kernel
            kii = self.filters[ii, ::-1, ::-1, ...]
            deltaii = delta_in[:, :, ii]
            # loop through channels of input
            for jj in range(nc_pre):
                slabij = fullConv3D(deltaii, kii[:, :, jj], self.stride)
                result[:, :, jj] += slabij

        result = result * dReLU(z)

        return result

    def computeGradients(self, delta, act):
        '''Compute gradients of cost wrt filter weights

        Args:
            delta (ndarray): errors in filter ouputs.
            act (ndarray): activations fed into filter.
        Returns:
            grads (ndarray): gradients of filter weights.
            grads_bias (ndarray): 1d array, gradients of biases.

        The theoretical equation of gradients of filter weights is:

            \partial J / \partial W^{(l)} = a^{(l-1)} \bigotimes \delta^{(l)}

        where:
            J : cost function of network.
            W^{(l)} : weights in filter.
            a^{(l-1)} : activations fed into filter.
            \bigotimes : convolution in valid mode.
            \delta^{(l)} : errors in the outputs from the filter.

        Computation in practice is more complicated than the above equation.
        '''

        nc_out = delta.shape[-1]  # number of channels in outputs
        nc_in = act.shape[-1]  # number of channels in inputs

        grads = np.zeros_like(self.filters)

        for ii in range(nc_out):
            deltaii = np.take(delta, ii, axis=-1)
            gii = grads[ii]
            for jj in range(nc_in):
                actjj = act[:, :, jj]
                gij = conv3D3(actjj, deltaii, stride=1, pad=0)
                gii[:, :, jj] += gij
            grads[ii] = gii

        # gradient clip
        gii = np.clip(gii, -self.clipvalue, self.clipvalue)
        grads_bias = np.sum(delta, axis=(0, 1))  # 1d

        return grads, grads_bias

    def gradientDescent(self, grads, grads_bias, m):
        '''Gradient descent weight and bias update'''

        self.filters = self.filters * (1 - self.lr * self.lam / m) - self.lr * grads / m
        self.biases = self.biases - self.lr * grads_bias / m

        return


class FlattenLayer(object):
    def __init__(self, input_shape):
        '''Flatten layer'''

        self.type = 'flatten'
        self.input_shape = input_shape

    @property
    def n_params(self):
        return 0

    def forward(self, x):
        '''Forward pass'''

        x = x.flatten()
        return x, x

    def backPropError(self, delta_in, z):
        '''Back-propagate errors
        '''

        result = np.reshape(delta_in, tuple(self.input_shape))
        return result


class FCLayer(object):
    def __init__(self, n_inputs, n_outputs, learning_rate, af=None, lam=0.01,
                 clipvalue=0.5):
        '''Fully-connected layer

        Args:
            n_inputs (int): number of inputs.
            n_outputs (int): number of layer outputs.
            learning_rate (float): initial learning rate.
        Keyword Args:
            af (callable): activation function. Default to ReLU.
            lam (float): regularization parameter.
            clipvalue (float): clip gradients within [-clipvalue, clipvalue]
                during back-propagation.
        '''

        self.type = 'fc'
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.lr = learning_rate
        self.clipvalue = clipvalue

        if af is None:
            self.af = ReLU
        else:
            self.af = af

        self.lam = lam  # regularization parameter
        self.init()

    @property
    def n_params(self):
        return self.n_inputs * self.n_outputs + self.n_outputs

    def init(self):
        '''Initialize weights

        Default to use HE initialization:
            w ~ N(0, std)
            std = \sqrt{2 / n}
        where n is the number of inputs
        '''
        std = np.sqrt(2 / self.n_inputs)
        np.random.seed(100)
        self.weights = np.random.normal(0, std, size=[self.n_outputs, self.n_inputs])
        self.biases = np.random.normal(0, std, size=self.n_outputs)

    def forward(self, x):
        '''Forward pass'''

        z = np.dot(self.weights, x) + self.biases
        a = self.af(z)

        return z, a

    def backPropError(self, delta_in, z):
        '''Back-propagate errors

        Args:
            delta_in (ndarray): delta from the next layer in the network.
            z (ndarray): weighted sum of the current layer.
        Returns:
            result (ndarray): delta of the current layer.

        The theoretical equation for error back-propagation is:

            \delta^{(l)} = W^{(l+1)}^{T} \cdot \delta^{(l+1)} \bigodot f'(z^{(l)})

        where:
            \delta^{(l)} : error of layer l, defined as \partial J / \partial z^{(l)}.
            W^{(l+1)} : weights of layer l+1.
            \bigodot : Hadamard (elementwise) product.
            f() : activation function of layer l.
            z^{(l)} : weighted sum in layer l.
        '''

        result = np.dot(self.weights.T, delta_in) * dReLU(z)

        return result

    def computeGradients(self, delta, act):
        '''Compute gradients of cost wrt weights

        Args:
            delta (ndarray): errors in ouputs.
            act (ndarray): activations fed into weights.
        Returns:
            grads (ndarray): gradients of weights.
            grads_bias (ndarray): 1d array, gradients of biases.

        The theoretical equation of gradients of filter weights is:

            \partial J / \partial W^{(l)} = \delta^{(l)} \cdot a^{(l-1)}^{T}

        where:
            J : cost function of network.
            W^{(l)} : weights in layer.
            a^{(l-1)} : activations fed into the weights.
            \delta^{(l)} : errors in the outputs from the weights.

        When implemented, had some trouble getting the shape correct. Therefore
        used einsum().
        '''
        # grads = np.einsum('ij,kj->ik', delta, act)
        grads = np.outer(delta, act)
        # gradient-clip
        grads = np.clip(grads, -self.clipvalue, self.clipvalue)
        grads_bias = np.sum(delta, axis=-1)

        return grads, grads_bias

    def gradientDescent(self, grads, grads_bias, m):
        '''Gradient descent weight and bias update'''

        self.weights = self.weights * (1 - self.lr * self.lam / m) - self.lr * grads / m
        self.biases = self.biases - self.lr * grads_bias / m

        return
