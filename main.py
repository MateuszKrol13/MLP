import numpy as np

from src.mlp import MultiLayerPerceptron, Layer
from src.optimizer import SGD
from src.losses import MeanSquareError, CategoricalCrossEntropy
from utils import load_data

if __name__ == "__main__":
    x, y = load_data()

    nn = MultiLayerPerceptron(
        opt=SGD(lr=0.1),
        err=CategoricalCrossEntropy
    )

    nn.add(
        Layer(weights=np.random.randn(784, 120), bias=np.random.randn(1, 120), activation="relu"),
    )
    nn.add(
        Layer(weights=np.random.randn(120, 10), bias=np.random.randn(1, 10), activation="softmax")
    )

    nn.fit(x=x, y=y, epochs=50, batch=10)