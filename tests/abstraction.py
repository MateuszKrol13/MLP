import unittest
import numpy as np

from src.activations import get_activation

def _relu(x):
    return np.where(x > 0, x, 0)

def _softmax(x):
    scaled_x = x - np.max(x, axis=0)
    e_x = np.exp(scaled_x)
    activated = e_x / np.sum(e_x.flatten())
    return activated

def _load_dataset():
    examples, targets = [], []
    with open(r"F:\dev\python\NN-scratch\data\digit-recognizer\train.csv", "r") as f:
        for line in f:
            y, *x = line.split(',')
            examples.append(x)
            targets.append(y)

    x = np.asarray(examples[1:], dtype=np.float32) / 255
    y = np.eye(10, dtype=np.float32)[np.asarray(targets[1:], dtype=int)]
    return x, y

"""Testing classes against bare bone implementation of mlp network"""
class AbstractionTest(unittest.TestCase):
    prediction=np.asarray([0.5, 0.3])
    target=np.asarray([1., 0.])

    test_calc_loss_expected_val=np.asarray([0.25, 0.09]) / 2
    test_calc_loss_expected_derr=np.asarray([0.5, 0.3])



    @classmethod
    def setUpClass(cls):
        """Single pass to compare outputs"""
        BATCH = 10
        cls.LEARNING_RATE = 0.1
        NO_CLASSES = 10
        EPOCHS = 100
        print(f"BATCH : {BATCH}")

        x, y = _load_dataset()

        # init weights
        cls.w_1 = np.random.randn(784, 120)
        cls.b_1 = np.random.randn(1, 120)

        cls.w_2 = np.random.randn(120, 10)
        cls.b_2 = np.random.randn(1, 10)

        ### FORWARD PASS ###
        cls.x_batch, cls.y_batch = x[0:BATCH], y[0:BATCH]

        # LAYER 1
        cls.i1 = cls.x_batch
        cls.o1 = cls.i1 @ cls.w_1 + cls.b_1
        cls.a1 = _relu(cls.o1)

        # LAYER 2
        cls.i2 = cls.a1
        cls.o2 = cls.i2 @ cls.w_2 + cls.b_2

        # softmax in-place
        up = np.exp(cls.o2 - np.max(cls.o2, axis=-1, keepdims=True))
        down = np.sum(up, axis=-1, keepdims=True)
        cls.a2 = up / down

        cls.y_pred = cls.a2

        ### ERROR ###
        cls.cathegorical_crossentropy = -np.sum(cls.y_batch * np.log(cls.a2 + 1e-30), axis=-1)
        cls.cathegorical_crossentropy_derivative_and_softmax = cls.a2 - cls.y_batch
        cls.mse = ((cls.y_pred - cls.y_batch) ** 2 / NO_CLASSES)
        cls.mse_derivative = (2 * (cls.y_pred - cls.y_batch) / NO_CLASSES)

        ### BACK PASS ###
        # LAYER 2
        cls.softmax_derr_crossentropy = cls.cathegorical_crossentropy_derivative_and_softmax
        cls.softmax_derr_mse = cls.a2 * (cls.mse_derivative - np.sum(cls.mse_derivative * cls.a2, axis=-1, keepdims=True))

        cls.dw_2 = cls.softmax_derr_crossentropy.T @ cls.i2 / BATCH
        cls.db_2 = cls.softmax_derr_crossentropy.T / BATCH

        cls.l2_derivative = cls.softmax_derr_crossentropy @ cls.w_2.T

        # DONT update
        # w_2 -= dw_2.T * LEARNING_RATE
        # b_2 -= np.sum(db_2.T, axis=0) * LEARNING_RATE

        # LAYER 1
        cls.relu_derr = cls.l2_derivative * np.where(cls.o1 > 0, 1, 0)
        cls.dw_1 = cls.relu_derr.T @ cls.i1 / BATCH
        cls.db_1 = cls.relu_derr.T / BATCH

        # DONT update
        # w_1 -= dw_1.T * LEARNING_RATE
        # b_1 -= np.sum(db_1.T, axis=0) * LEARNING_RATE

    def test_relu_activation(self):
        d_relu, a_relu = get_activation("relu")(self.o1)
        np.testing.assert_almost_equal(
            a_relu,
            self.a1,
            err_msg=f"ReLU activation value different than expected!"
        )

        np.testing.assert_almost_equal(
            self.l2_derivative * d_relu,
            self.relu_derr,
            err_msg=f"ReLU partial derivative value different than expected!"
        )

    def test_softmax_activation(self):
        _, out = get_activation("softmax")(self.o2)
        np.testing.assert_almost_equal(
            out,
            self.a2,
            err_msg=f"Softmax activation value different than expected!"
        )

    def test_update_weights(self):
        from src.optimizer import SGD
        from src.mlp import Layer
        from utils.grad import Grad
        from src.losses import CategoricalCrossEntropy

        opt = SGD(lr=self.LEARNING_RATE)

        w1_updated = self.w_1 - self.dw_1.T * self.LEARNING_RATE
        b1_updated = self.b_1 - np.sum(self.db_1.T, axis=0)  * self.LEARNING_RATE

        w2_updated = self.w_2 - self.dw_2.T * self.LEARNING_RATE
        b2_updated = self.b_2 - np.sum(self.db_2.T, axis=0) * self.LEARNING_RATE

        l1 = Layer(activation="relu")._from_ndarray(
            w_arr=np.copy(self.w_1),
            b_arr=np.copy(self.b_1)
        )
        l1_act_derr, _ = get_activation("relu")(self.o1)
        l1._grad=Grad()._set(
            l_derr=self.w_1,
            o_derr=self.i1,
            act_derr=l1_act_derr
        )
        opt.apply_grad(owner=l1, passdown_err=self.l2_derivative,  batch=10)

        # L1
        np.testing.assert_almost_equal(
            w1_updated,
            l1.weights
        )
        np.testing.assert_almost_equal(
            b1_updated,
            l1.bias
        )

        # L2
        l2 = Layer(activation="softmax")._from_ndarray(
            w_arr=np.copy(self.w_2),
            b_arr=np.copy(self.b_2)

        )
        l2._grad=Grad()._set(
            l_derr=self.w_2,
            o_derr=self.i2,
            act_derr=None  # softmax special case
        )
        opt.apply_grad(l2, passdown_err=self.cathegorical_crossentropy_derivative_and_softmax, batch=10)
        np.testing.assert_almost_equal(
            w2_updated,
            l2.weights
        )
        np.testing.assert_almost_equal(
            b2_updated,
            l2.bias
        )