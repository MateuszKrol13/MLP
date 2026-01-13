import numpy as np
from numpy.typing import NDArray
from typing import Tuple

x_data = NDArray
y_data = NDArray

def load_data() -> Tuple[x_data, y_data]:
    examples, targets = [], []
    with open(r"F:\dev\python\NN-scratch\data\digit-recognizer\train.csv", "r") as f:
        for line in f:
            y, *x = line.split(',')
            examples.append(x)
            targets.append(y)

    x = np.asarray(examples[1:], dtype=np.float32) / 255
    y = np.eye(10, dtype=np.float32)[np.asarray(targets[1:], dtype=int)]

    return x, y