from typing import Callable, Dict
import numpy as np

ACTIVATION_REGISTRY: Dict[str, Callable] = {None: None}

def get_activation(name: str) -> Callable:
    if name not in ACTIVATION_REGISTRY:
        raise ValueError(f"Function {name} is not a registered activation function.")

    return ACTIVATION_REGISTRY[name]

def register_activation(name: str):
    def wrapper(func: Callable):
        ACTIVATION_REGISTRY[name] = func
        return func
    return wrapper

@register_activation("relu")
def relu(input):
    activated = np.where(input > 0, input, 0)
    derived = np.where(activated > 0, 1, 0)

    return derived, activated

@register_activation("softmax")
def softmax(input):
    activated =  1 / (1 + np.exp(-1 * input))
    derived = activated * (1 - activated)

    return derived, activated
