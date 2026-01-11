from abc import abstractmethod
from numpy import ndarray
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
        if not isinstance(y_pred, ndarray):
            raise TypeError("Prediction matrix is not a numpy array!")

        if not isinstance(y_true, ndarray):
            raise TypeError("Target matrix is not a numpy array!")
        #assert y_pred.shape == y_true.shape

        loss = ((y_pred - y_true) ** 2 / y_pred.size).reshape((1, y_pred.size))
        loss_derivative = (2 * abs(y_pred - y_true) / y_pred.size).reshape((1, y_pred.size))

        return loss, loss_derivative