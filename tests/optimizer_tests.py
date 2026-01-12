import unittest
import numpy as np

from src.optimizer import SGD
from src.mlp import Layer
from utils import Grad

class SGDTests(unittest.TestCase):
    passdown_err = np.asarray([[23, -10]])

    l_derr=np.asarray([[1, -1], [-1, 1]])
    o_derr = np.asarray([[7, 7]])
    act_derr = np.ones((1, 2))

    test_apply_grad_weights=np.asarray([[-15.1, 8.], [-15.1, 8.]])

    @classmethod
    def setUpClass(cls):
        cls.opt = SGD(lr=0.1)
        cls.test_layer=Layer(activation="relu")._from_ndarray(
            w_arr=np.ones((2, 2)),
            b_arr=np.ones((1, 2)) * -2
        )

    def test_apply_grad(self):
        self.test_layer._grad = Grad()._set(
            l_derr=self.l_derr,
            o_derr=self.o_derr,
            act_derr=self.act_derr
        )

        l_err = self.opt.apply_grad(self.test_layer, passdown_err=self.passdown_err)

        np.testing.assert_almost_equal(
            self.test_layer.weights,
            self.test_apply_grad_weights,
            err_msg=f"Weights after update are different than expected {self.test_apply_grad_weights},"
                    f" given {self.test_layer.weights}",
            decimal=10
        )

    def test_apply_grad_clear(self):
        self.test_layer._grad = Grad()._set(
            l_derr=1,
            o_derr=1,
            act_derr=1
        )
