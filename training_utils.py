import numpy as np

def feed_forward(model, X_b):
    Z1 = X_b @ model.W1 + model.b1
    y_hat = Z1 @ model.W2 + model.b2
    return y_hat

def l2_loss(y_true, y_hat):
    N = y_true.shape[0]
    return np.sum((y_true - y_hat) ** 2) / (2 * N)

def compute_gradient(model, X_b, y_b, y_hat):

    N = y_b.shape[0]

    dY = (y_hat - y_b) / N

    Z1 = X_b @ model.W1 + model.b1

    dW2 = Z1.T @ dY
    db2 = np.sum(dY, axis=0, keepdims=True)

    dZ1 = dY @ model.W2.T

    dW1 = X_b.T @ dZ1
    db1 = np.sum(dZ1, axis=0, keepdims=True)

    # ✅ Gradient Clipping
    clip_value = 5
    dW1 = np.clip(dW1, -clip_value, clip_value)
    dW2 = np.clip(dW2, -clip_value, clip_value)
    db1 = np.clip(db1, -clip_value, clip_value)
    db2 = np.clip(db2, -clip_value, clip_value)

    return dW1, db1, dW2, db2

def update_parameters(model, grads, lr):

    dW1, db1, dW2, db2 = grads

    model.W1 -= lr * dW1
    model.b1 -= lr * db1
    model.W2 -= lr * dW2
    model.b2 -= lr * db2
