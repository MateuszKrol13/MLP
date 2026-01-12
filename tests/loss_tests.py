import unittest
import numpy as np

from src.losses import MeanSquareError

class MSETests(unittest.TestCase):
    prediction=np.asarray([0.5, 0.3])
    target=np.asarray([1., 0.])

    test_calc_loss_expected_val=np.asarray([0.25, 0.09]) / 2
    test_calc_loss_expected_derr=np.asarray([0.5, 0.3])

    @classmethod
    def setUpClass(cls):
        cls.cls_reference = MeanSquareError
        cls.obj = MeanSquareError()

    def test_call_object(self):
        arr = np.empty((3, 2))
        try:
            _ = self.obj.calc_loss(arr, arr)
        except Exception:
            self.assertTrue(False, msg="Calling loss object failed")

    def test_call_reference(self):
        arr = np.empty((3, 2))
        try:
            _ = self.cls_reference.calc_loss(arr, arr)
        except Exception:
            self.assertTrue(False, msg="Calling loss object failed")

    def test_calc_loss(self):
        loss, derivative = self.obj.calc_loss(
            y_pred=self.prediction,
            y_true=self.target
        )

        self.assertTrue(
            np.array_equal(loss, self.test_calc_loss_expected_val),
            msg=f"Loss value given {loss}, expected {self.test_calc_loss_expected_val}"
        )
        self.assertTrue(
            np.array_equal(derivative, self.test_calc_loss_expected_derr),
            msg=f"Loss derivative given {derivative}, expected {self.test_calc_loss_expected_derr}"
        )

    def test_output_shape(self):
        loss, derivative = self.obj.calc_loss(
            y_pred=self.prediction,
            y_true=self.target
        )

        self.assertEqual(len(loss.shape), 2)
        self.assertEqual(len(derivative.shape), 2)