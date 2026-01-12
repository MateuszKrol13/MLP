from typing import Type, Dict
from abc import abstractmethod
import numpy as np

from autograd import Variable, DifferentiableVariable

ACTIVATION_REGISTRY: Dict[str, Type["Activation"]] = {}

def get_activation(name: str) -> Type["Activation"]:
    if name not in ACTIVATION_REGISTRY:
        raise ValueError(f"Activation '{name}' is not registered.")
    return ACTIVATION_REGISTRY[name]


def register_activation(name: str):
    def wrapper(cls: Type["Activation"]):
        ACTIVATION_REGISTRY[name] = cls
        return cls
    return wrapper

class Activation:

    @abstractmethod
    def __call__(self, input: np.ndarray) -> np.ndarray:
        pass

@register_activation("relu")
class ReLU(Activation, Variable):
    _class_counter=0

    def __init__(self):
        super().__init__(
            value=None,
            name=f"ReLU-{self._class_counter}",
        )
        self._class_counter+=1

    def __call__(self, input: np.ndarray) -> np.ndarray:
        self.value = np.where(input > 0, input, 0)
        return self.value

    def pass_gradient(self, grad):
        local_grad = np.where(self.value > 0, 1, 0)
        self.value = None
        return grad * local_grad

@register_activation("sigmoid")
class Sigmoid(Activation, Variable):
    _class_counter=0

    def __init__(self):
        super().__init__(
            value=None,
            name=f"Sigmoid-{self._class_counter}",
        )
        self._class_counter+=1

    def __call__(self, input: np.ndarray) -> np.ndarray:
        self.value = 1.0 / (1.0 + np.exp(-input))
        return self.value

    def pass_gradient(self, grad):
        local_grad = self.value * (1-self.value)
        self.value=None
        return grad * local_grad

@register_activation("linear")
class Linear(Activation, Variable):
    _class_counter = 0

    def __init__(self):
        super().__init__(
            value=None,
            name=f"Linear-{self._class_counter}",
        )
        self._class_counter += 1

    def __call__(self, input: np.ndarray) -> np.ndarray:
        self.value = input
        return self.value

    def pass_gradient(self, grad) -> np.ndarray:
        local_grad = np.ones_like(self.value, dtype=np.float32)
        self.value=None
        return grad * local_grad


class Softmax(Activation, Variable):
    _class_counter = 0

    def __init__(self):
        super().__init__(
            value=None,
            name=f"Softmax-{self._class_counter}",
        )
        self._class_counter += 1

    def __call__(self, input: np.ndarray) -> np.ndarray:
        scaled_logits = input - np.max(input, axis=-1)
        e_logits = np.exp(scaled_logits)
        self.value = e_logits / np.sum(e_logits, axis=-1)
        return self.value

    def pass_gradient(self, grad) -> np.ndarray:
        local_grad = self.value * (grad - np.sum(grad * self.value, axis=-1))
        self.value = None
        return local_grad

'''
@register_activation("softmax")
def softmax(logits, loss_error):
    # for single batch only?

    scaled_logits = logits - np.max(logits, axis=0)
    e_logits = np.exp(scaled_logits)
    activated = e_logits / np.sum(e_logits.flatten())

    dot = np.sum(loss_error * activated, axis=1, keepdims=True)
    derived = activated * (loss_error - dot)

    return derived, activated
'''