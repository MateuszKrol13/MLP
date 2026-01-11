import numpy as np
from typing import Callable
from numpy import matrix

from activations import *
from losses import *
from utils import Grad

class Layer:
    _instance_cntr = 0

    def __init__(
            self,
            weights=None,
            bias=None,
            activation: str=None
    ):
        self.name = f"layer_{Layer._instance_cntr}" # this is only bounded to runtime
        Layer._instance_cntr +=1
        self.weights = weights
        self.bias = bias
        self.activation = get_activation(activation) if activation is not None else None
        self._grad = Grad()

    def forward(self, input):
        # in[batch x features] * weights[features x neurons] + bias[neurons] = out[batch x neurons]
        output = input @ self.weights + self.bias
        act_derr = None

        # out[b x n] = activation(out[b x n])
        if self.activation:
            act_derr, output = self.activation(output)

        # update grad
        self._grad.update(
            l_derr=self.weights,
            out_derr=input, # d_out / d_w
            act_derr=act_derr
        )

        return output

    def _from_ndarray(self, w_arr, b_arr):
        # used for debug
        if len(w_arr.shape) != 2:
            raise ValueError(f"Weights array is not 2D, given shape {w_arr.shape}")
        if b_arr.shape[0] > b_arr.shape[-1]:
            raise ValueError(f"Bias vector must be a row matrix, given shape {b_arr.shape}")

        self.weights = w_arr
        self.bias=b_arr
        return self

class MultiLayerPerceptron:
    def __init__(self, opt, err):
        self._optimizer=opt
        self._loss=err
        self._layers = []

    def add(self, l: Layer):
        self._layers.append(l)

    def pop(self):
        if not self._layers:
            raise ValueError("Layer list is empty!")
        del self._layers[-1]

    def fit(self, x, y, epochs: int):

        print("begin network training!")
        for epoch in range(epochs):
            print(f"Running epoch {epoch} of {epochs}")
            for i in range(len(x)):
                # forward
                inp = x[i]
                for layer in self._layers:
                    inp = layer.forward(inp)

                # error func
                err_derr, loss = self._loss.calc_loss(inp, y[i])
                print(f"Loss: {np.sum(loss)}", end="\r")

                # backward
                layer_err = err_derr
                for layer in reversed(self._layers):
                    layer_err = self._optimizer.apply_grad(layer, passdown_err=layer_err)

    def predict(self, x):
        predictions = []
        for i in range(len(x)):
            inp = x[i]
            for layer in self._layers:
                inp = layer.forward(inp)

            predictions.append(inp)

        return predictions

if "__main__" == __name__:
    # tests
    w = np.asarray([ [-1, 0, 1, 2]] *3, dtype=np.float32) * 3
    b = np.asarray([1] * 4, dtype=np.float32)
    inp = np.asarray([-2, 3, 0], dtype=np.float32)
    tar = np.asarray([0, 1], dtype=np.float32)

    L = Layer(weights=w, bias=b)
    o = L.forward(inp)

    L2 = Layer(weights=w, bias=b, activation="relu")
    L2.forward(inp)
    print(L2._cache)

    #runs
    L_one = Layer(
        weights=np.asarray([[0.3, 0.3, 0.1, 0.2], [0.3, 0.4, 0.3, 0. ], [0.3, 0.7, 0.6, 0.3]], dtype=np.float32),
        bias=np.asarray([0.9, 0.9, 0.4, 0.5], dtype=np.float32),
        activation="relu"
    )

    L_two = Layer(
        weights=np.asarray([[0.5, 0.7], [0.5, 0.6], [0.8, 0.3], [0.4, 0.7]], dtype=np.float32),
        bias=np.asarray([0.3, 0.2], dtype=np.float32),
        activation="relu"
    )

    L_three = Layer(
        weights=np.asarray([[0.8, 0.6], [0.5, 0.3]], dtype=np.float32),
        bias=np.asarray([0.5, 0.4], dtype=np.float32),
        activation="softmax"
    )

    #forward
    inp=np.asarray([[1.0, 0.5, -1.0]])
    target = np.asarray([1, 0])

    i1 = L_one.forward(inp)
    i2 = L_two.forward(i1)
    i3 = L_three.forward(i2)

    #loss
    mse = (i3 - target) ** 2 / tar.size
    d_mse = 2 * (i3-target) / tar.size

    #backward
    b3 = L_three.backwards(d_mse, 0.1)
    b2 = L_two.backwards(b3,0.1)
    b1 = L_one.backwards(b2, 0.1)