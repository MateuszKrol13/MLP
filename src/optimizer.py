from abc import abstractmethod
from src.mlp import Layer
import numpy as np

class Optimizer:
    def __init__(self, lr):
        self._learning_rate = lr

    @abstractmethod
    def apply_grad(self, owner, passdown_err):
        pass


class SGD(Optimizer):
    def __init__(self, lr):
        super().__init__(lr)
        pass

    def apply_grad(self, owner: Layer, passdown_err, batch:int=1):
        grad = owner._grad

        # workaround for activations bounded with loss functions / dependent on passdown gradient
        if grad.activation_derivative is None:
            activation_error = passdown_err
        else:
            activation_error = passdown_err * grad.activation_derivative  # element-wise mult

        # update weights d_weights = passdown_err * d_activation * d_output
        d_weights = activation_error.T @ grad.output_derivative / batch
        d_bias = activation_error.T / batch

        # layer_derivative
        layer_derr = activation_error @ owner.weights.T

        # update values
        owner.weights -= d_weights.T * self._learning_rate
        owner.bias -= np.sum(d_bias.T, axis=0, keepdims=True) * self._learning_rate
        grad.clear()

        return layer_derr
