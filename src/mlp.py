import numpy as np

from src.activations import *
from utils import Grad

class Layer:
    _instance_cntr = 0

    def __init__(
            self,
            weights=None,
            bias=None,
            activation: str=None
    ):
        self.name = f"layer_{Layer._instance_cntr}" # this is only bounded to runtime
        Layer._instance_cntr +=1
        self.weights = weights
        self.bias = bias
        self.activation = get_activation(activation) if activation is not None else None
        self.built = False
        self._grad = Grad()

    def forward(self, input):
        # in[batch x features] * weights[features x neurons] + bias[neurons] = out[batch x neurons]
        output = input @ self.weights + self.bias
        act_derr = None

        # out[b x n] = activation(out[b x n])
        if self.activation:
            act_derr, output = self.activation(output)

        # update grad
        self._grad.update(
            l_derr=self.weights,
            out_derr=input, # d_out / d_w
            act_derr=act_derr
        )
        return output

    def _from_ndarray(self, w_arr, b_arr):
        # used for debug
        if len(w_arr.shape) != 2:
            raise ValueError(f"Weights array is not 2D, given shape {w_arr.shape}")
        if b_arr.shape[0] > b_arr.shape[-1]:
            raise ValueError(f"Bias vector must be a row matrix, given shape {b_arr.shape}")

        self.weights = w_arr
        self.bias=b_arr
        return self

class MultiLayerPerceptron:
    def __init__(self, opt, err):
        self._optimizer=opt
        self._loss=err
        self._layers = []

    def add(self, l: Layer):
        self._layers.append(l)

    def pop(self):
        if not self._layers:
            raise ValueError("Layer list is empty!")
        del self._layers[-1]

    def fit(self, x, y, epochs: int, batch: int):

        print("begin network training!")
        for epoch in range(epochs):
            losses = []
            for i in range(0, len(x) // batch, 1):
                ### DATA ###
                x_batch, y_batch = x[i*batch:(i+1)*batch], y[i*batch:(i+1)*batch]

                ### FORWARD ###
                for layer in self._layers:
                    x_batch = layer.forward(x_batch)

                ### ERROR ###
                loss, err_derr = self._loss.calc_loss(y_pred=x_batch, y_true=y_batch)
                losses.append(loss)
                print(f"Epoch {epoch}, loss: {np.sum(loss)}", end="\r")

                ### BACKWARD ###
                err=err_derr
                for layer in reversed(self._layers):
                    err = self._optimizer.apply_grad(layer, passdown_err=err, batch=batch)

            print(f"Epoch {epoch} loss combined: {np.mean(losses)}")

    def save_weights(self):
        import pickle
        save_dict = {}
        for layer in self._layers:
            layer_save = {
                "weights": layer.weights,
                "bias": layer.bias,
                "activation": layer.activation
            }
            save_dict[layer.name] = layer_save

        with open("mlp.pkl", "wb") as f:
            pickle.dump(save_dict, f)

    def load_weigths(self):
        import pickle
        with open("mlp.pkl", "rb") as f:
            save_dict = pickle.load(f)

        self._layers = []
        for k, v in save_dict.items():
            l = Layer(activation=v["activation"])._from_ndarray(
                w_arr=v["weights"],
                v_arr=v["bias"]
            )
            self._layers.append(l)

    def predict(self, x, batch=1):
        predictions = []
        for i in range(0, len(x) // batch, 1):
            inp = x[i*batch:(i+1)*batch]
            for layer in self._layers:
                inp = layer.forward(inp)

            predictions.append(inp)

        return predictions