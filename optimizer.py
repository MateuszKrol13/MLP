from abc import abstractmethod
from mlp import Layer
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

    def apply_grad(self, owner: Layer, passdown_err):
        if passdown_err.shape[-1] != passdown_err.size:
            raise ValueError(f"Passdown error must be a row vector, instead got matrix of shape {passdown_err.shape}")

        grad = owner._grad
        activation_error = passdown_err * grad.activation_derivative  # element-wise mult
        output_error = grad.output_derivative

        d_weights = activation_error.T @ grad.output_derivative
        d_bias = activation_error
        owner.weights -= d_weights.T * self._learning_rate
        owner.bias -= d_bias * self._learning_rate

        grad.clear()
        return output_error
