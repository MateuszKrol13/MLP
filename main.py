import numpy as np

from mlp import MultiLayerPerceptron, Layer
from optimizer import SGD
from losses import MeanSquareError

if __name__ == "__main__":
    examples, targets = [], []
    with open(r"F:\dev\python\NN-scratch\data\digit-recognizer\train.csv", "r") as f:
        for line in f:
            y, *x = line.split(',')
            examples.append(x)
            targets.append(y)

    x = [np.asarray([ex], dtype=np.float32) for ex in examples[1:]]
    y = np.eye(10, dtype=np.float32)[np.asarray(targets[1:], dtype=int)]


    nn = MultiLayerPerceptron(
        opt=SGD(lr=0.1),
        err=MeanSquareError
    )

    nn.add(
        Layer(weights=np.random.randn(784, 120), bias=np.random.randn(1, 120), activation="relu"),
    )
    nn.add(
        Layer(weights=np.random.randn(120, 10), bias=np.random.randn(1, 10), activation="softmax")
    )

    nn.fit(x=x, y=y, epochs=3)