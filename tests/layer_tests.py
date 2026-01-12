import unittest
import numpy as np

from src.mlp import Layer
from src.activations import get_activation

class ConstructionTests(unittest.TestCase):
    def setUp(self):
        self.layer = Layer()

    def test_naming(self):
        Layer._instance_cntr = 0
        for i in range(5):
            l = Layer()
        self.assertEqual(l.name, "layer_4", msg="Layer name different than expected")

    def test_default_init(self):
        fields = self.layer.__dict__


class PassTests(unittest.TestCase):
    input=np.asarray([[1, 0, 0]])
    weights=np.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    bias=np.asarray([-1, 0, 1])

    test_forward_pass_expected = np.asarray([[0, 2, 4]])
    test_grad_update_vals_expected = {
        "layer_grad" : np.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
        "output_grad" : np.asarray([[1, 0, 0]]),
        "activation_grad" : np.asarray([[0, 1, 1]])
    }

    @classmethod
    def setUpClass(cls):
        cls.layer=Layer(activation='relu')._from_ndarray(
            w_arr=cls.weights,
            b_arr=cls.bias
        )
        cls.forward_output = cls.layer.forward(cls.input)

    def test_forward_pass(self):
        self.assertTrue(
            np.array_equal(self.forward_output, self.test_forward_pass_expected),
            msg=f"forward_output {self.forward_output} is not equal to expected output {self.test_forward_pass_expected}"
        )

    # TODO: test activation derr
    def test_grad_update_auto(self):
        l_derr, o_derr, a_derr = self.layer._grad
        activation_derr, _ = get_activation('relu')(self.input)

        self.assertTrue(
            np.array_equal(l_derr, self.layer.weights),
            msg=f"Layer derivative given {l_derr}, expected {self.layer.weights}"
        )
        self.assertTrue(
            np.array_equal(o_derr, self.input),
            msg=f"Output derivative given {o_derr}, expected {self.input}"
        )
        #self.assertTrue(
        #    np.array_equal(a_derr, activation_derr),
        #    msg=f"Activation derivative given {a_derr}, expected {activation_derr}"
        #)

    def test_grad_update_vals(self):
        for k, v in zip(self.layer._grad, self.test_grad_update_vals_expected.values()):
            with self.subTest(msg="", param_derived=k, param_expected=v):
                self.assertTrue(
                    np.array_equal(k, v)
                )

    def test_preserve_shape(self):
        weights_random = np.random.randn(32, 172)
        bias_random = np.random.randn(1, 172)

        l = Layer(activation="linear")._from_ndarray(
            w_arr=weights_random,
            b_arr=bias_random
        )
        inp = np.random.randn(100, 32)
        outp = l.forward(inp)

        self.assertEqual(outp.shape, (100, 172))