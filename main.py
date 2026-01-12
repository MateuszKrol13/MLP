import numpy as np

from src.mlp import MultiLayerPerceptron, Layer
from src.optimizer import SGD
from src.losses import MeanSquareError, CategoricalCrossEntropy

if __name__ == "__main__":
    examples, targets = [], []
    with open(r"F:\dev\python\NN-scratch\data\digit-recognizer\train.csv", "r") as f:
        for line in f:
            y, *x = line.split(',')
            examples.append(x)
            targets.append(y)

    x = np.asarray(examples[1:], dtype=np.float32) / 255
    y = np.eye(10, dtype=np.float32)[np.asarray(targets[1:], dtype=int)]

    nn = MultiLayerPerceptron(
        opt=SGD(lr=0.5),
        err=CategoricalCrossEntropy
    )

    nn.add(
        Layer(weights=np.random.randn(784, 128), bias=np.random.randn(1, 128), activation="relu"),
    )
    nn.add(
        Layer(weights=np.random.randn(128, 64), bias=np.random.randn(1, 64), activation="relu"),
    )
    nn.add(
        Layer(weights=np.random.randn(64, 32), bias=np.random.randn(1, 32), activation="relu"),
    )
    nn.add(
        Layer(weights=np.random.randn(32, 10), bias=np.random.randn(1, 10), activation="sigmoid")
    )

    nn.fit(x=x, y=y, epochs=50, batch=100)