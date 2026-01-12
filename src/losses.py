from abc import abstractmethod
import numpy as np
from numpy.typing import NDArray
from utils.typing import LossValue, LossDerivative

class Loss:
    @classmethod
    @abstractmethod
    def calc_loss(cls, y_pred, y_true) -> tuple[LossValue, LossDerivative]:
        pass


class MeanSquareError(Loss):

    @classmethod
    def calc_loss(cls, y_pred: NDArray, y_true: NDArray) -> tuple[LossValue, LossDerivative]:
        if not isinstance(y_pred, np.ndarray):
            raise TypeError("Prediction matrix is not a numpy array!")

        if not isinstance(y_true, np.ndarray):
            raise TypeError("Target matrix is not a numpy array!")
        #assert y_pred.shape == y_true.shape

        loss = ((y_pred - y_true) ** 2 / y_pred.size)
        loss_derivative = (2 * abs(y_pred - y_true) / y_pred.size)

        return loss, loss_derivative

class CategoricalCrossEntropy(Loss):

    @classmethod
    def calc_loss(cls, y_pred, y_true) -> tuple[LossValue, LossDerivative]:
        y_pred = np.clip(y_pred, 1e-12, 1 - 1e-12)
        loss = -np.sum(y_true * np.log(y_pred), axis=1)
        loss_derivative = y_pred - y_true

        return loss, loss_derivative