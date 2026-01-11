from typing import TypeVar
_this = TypeVar("_this", bound="Grad")

class Grad:
    def __init__(self):
        self.layer_derivative = None
        self.output_derivative = None
        self.activation_derivative = None

    def __iter__(self):
        return iter((self.layer_derivative, self.output_derivative, self.activation_derivative))

    def update(self, l_derr=None, out_derr=None, act_derr=None):
        self.layer_derivative = l_derr
        self.output_derivative = out_derr
        self.activation_derivative = act_derr

    def clear(self):
        self.layer_derivative = None
        self.output_derivative = None
        self.activation_derivative = None

    def _set(self: _this, l_derr, o_derr, act_derr) -> _this:
        self.layer_derivative = l_derr
        self.output_derivative = o_derr
        self.activation_derivative = act_derr
        return self

    def get_grad(self):
        return self.__dict__
