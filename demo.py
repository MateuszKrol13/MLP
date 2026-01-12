import numpy as np

def relu(x):
    return np.where(x > 0, x, 0)

def softmax(x):
    scaled_x = x - np.max(x, axis=0)
    e_x = np.exp(scaled_x)
    activated = e_x / np.sum(e_x.flatten())
    return activated

if __name__ =="__main__":
    BATCH = 10
    LEARNING_RATE = 0.1
    NO_CLASSES = 10
    EPOCHS = 100
    print(f"BATCH : {BATCH}")

    examples, targets = [], []
    with open(r"F:\dev\python\NN-scratch\data\digit-recognizer\train.csv", "r") as f:
        for line in f:
            y, *x = line.split(',')
            examples.append(x)
            targets.append(y)

    x = np.asarray(examples[1:], dtype=np.float32) / 255
    y = np.eye(10, dtype=np.float32)[np.asarray(targets[1:], dtype=int)]
    y_cat = np.asarray(targets[1:], dtype=np.uint8)

    # init weights
    w_1 = np.random.randn(784, 120)
    b_1 = np.random.randn(1, 120)

    w_2 = np.random.randn(120, 10)
    b_2 = np.random.randn(1, 10)

    #epochs
    for e in range(EPOCHS):
        losses = []
        for i in range(0, len(x) // BATCH):
            ### FORWARD PASS ###
            x_batch, y_batch = x[i*BATCH:(i+1)*BATCH], y[i*BATCH:(i+1)*BATCH]

            # LAYER 1
            i1 = x_batch
            o1 = i1 @ w_1 + b_1
            a1 = relu(o1)

            # LAYER 2
            i2 = a1
            o2 = i2 @ w_2 + b_2
            #a2 = softmax(o2)
            #softmax in-place
            up = np.exp(o2- np.max(o2, axis=-1, keepdims=True))
            down = np.sum(up, axis=-1, keepdims=True)
            a2 = up / down

            y_pred = a2

            ### ERROR ###
            cathegorical_crossentropy = -np.sum(y_batch * np.log(a2 + 1e-30), axis=-1)

            #mse = ((y_pred - y_batch) ** 2 / NO_CLASSES)
            #mse_derivative = (2 * (y_pred - y_batch) / NO_CLASSES)
            print(f"Epoch {e}, loss: {np.sum(cathegorical_crossentropy)}", end="\r")
            losses.append(np.sum(cathegorical_crossentropy))

            ### BACK PASS ###
            # LAYER 2
            softmax_derr = a2 - y_batch
            #softmax_derr = a2 * (mse_derivative - np.sum(mse_derivative * a2, axis=-1, keepdims=True))

            dw_2 = softmax_derr.T @ i2 / BATCH
            db_2 = softmax_derr.T / BATCH

            l2_derivative = softmax_derr @ w_2.T

            #update
            w_2 -= dw_2.T * LEARNING_RATE
            b_2 -= np.sum(db_2.T, axis=0) * LEARNING_RATE

            # LAYER 1
            relu_derr = l2_derivative * np.where(o1 > 0, 1, 0)
            dw_1 = relu_derr.T @ i1 / BATCH
            db_1 = relu_derr.T / BATCH

            # update
            w_1 -= dw_1.T * LEARNING_RATE
            b_1 -= np.sum(db_1.T, axis=0) * LEARNING_RATE

        print(f"Epoch {e} loss combined: {np.mean(losses)}")

    # INFERENCE
    predictions = []
    for i in range(0, len(x) // BATCH):
        ### FORWARD PASS ###
        x_batch, y_batch = x[i * BATCH:(i + 1) * BATCH], y[i * BATCH:(i + 1) * BATCH]

        # LAYER 1
        i1 = x_batch
        o1 = i1 @ w_1 + b_1
        a1 = relu(o1)

        # LAYER 2
        i2 = a1
        o2 = i2 @ w_2 + b_2
        # a2 = softmax(o2)
        # softmax in-place
        up = np.exp(o2 - np.max(o2, axis=-1, keepdims=True))
        down = np.sum(up, axis=-1, keepdims=True)
        a2 = up / down

        y_pred = a2
        predictions.append(y_pred)