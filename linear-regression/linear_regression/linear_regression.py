import numpy as np

# -------------------------
# Loss Functions
# -------------------------
def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

# -------------------------
# Prediction Function
# -------------------------
def predict(X, w, b):
    return X * w + b

# -------------------------
# Batch Gradient Descent
# -------------------------
def batch_gradient_descent(X, y, lr=0.01, epochs=100):
    w, b = 0.0, 0.0
    n = len(X)
    for _ in range(epochs):
        y_pred = predict(X, w, b)
        dw = -(2/n) * np.sum(X * (y - y_pred))
        db = -(2/n) * np.sum(y - y_pred)
        w -= lr * dw
        b -= lr * db
    return w, b

# -------------------------
# Stochastic Gradient Descent
# -------------------------
def stochastic_gradient_descent(X, y, lr=0.01, epochs=100):
    w, b = 0.0, 0.0
    n = len(X)
    for _ in range(epochs):
        for i in range(n):
            xi, yi = X[i], y[i]
            y_pred = predict(xi, w, b)
            dw = -2 * xi * (yi - y_pred)
            db = -2 * (yi - y_pred)
            w -= lr * dw
            b -= lr * db
    return w, b

# -------------------------
# Mini-Batch Gradient Descent
# -------------------------
def mini_batch_gradient_descent(X, y, lr=0.01, epochs=100, batch_size=16):
    w, b = 0.0, 0.0
    n = len(X)
    for _ in range(epochs):
        indices = np.random.permutation(n)
        X_shuffled, y_shuffled = X[indices], y[indices]
        for i in range(0, n, batch_size):
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]
            y_pred = predict(X_batch, w, b)
            dw = -(2/len(X_batch)) * np.sum(X_batch * (y_batch - y_pred))
            db = -(2/len(X_batch)) * np.sum(y_batch - y_pred)
            w -= lr * dw
            b -= lr * db
    return w, b

