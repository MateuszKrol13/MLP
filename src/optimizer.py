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

        # update weights d_weights = passdown_err * d_activation * d_output
        activation_error = passdown_err * grad.activation_derivative  # element-wise mult
        d_weights = grad.output_derivative.T @ activation_error / batch
        d_bias = np.sum(activation_error, axis=0) / batch

        # layer_derivative
        layer_derr = activation_error @ owner.weights.T

        # update values
        owner.weights -= d_weights * self._learning_rate
        owner.bias -= d_bias * self._learning_rate
        grad.clear()

        return layer_derr
