import unittest
import numpy as np

from src.mlp import Layer

class NetworkExample(unittest.TestCase):
    def setUp(self):
        self.L1 = Layer(
            weights=np.asarray([[0.3, 0.3, 0.1, 0.2], [0.3, 0.4, 0.3, 0.], [0.3, 0.7, 0.6, 0.3]], dtype=np.float32),
            bias=np.asarray([0.9, 0.9, 0.4, 0.5], dtype=np.float32),
            activation="relu"
        )

        self.L2 = Layer(
            weights=np.asarray([[0.5, 0.7], [0.5, 0.6], [0.8, 0.3], [0.4, 0.7]], dtype=np.float32),
            bias=np.asarray([0.3, 0.2], dtype=np.float32),
            activation="relu"
        )

        self.L3 = Layer(
            weights=np.asarray([[0.8, 0.6], [0.5, 0.3]], dtype=np.float32),
            bias=np.asarray([0.5, 0.4], dtype=np.float32),
            activation="softmax"
        )

    def test_forward_pass(self):
        pass

    def test_backward_pass(self):
        pass

    def test_loss_calc(self):
        pass