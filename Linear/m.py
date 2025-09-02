import numpy as np
import matplotlib.pyplot as plt


# ------------------ Utility Functions ------------------
def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def plot_regression(X, y, y_pred, title="Linear Regression"):
    plt.scatter(X, y, color="blue", label="Data Points")
    plt.plot(X, y_pred, color="red", label="Regression Line")
    plt.xlabel("X")
    plt.ylabel("y")
    plt.legend()
    plt.title(title)
    plt.show()


def plot_loss_curve(loss_history, title="Loss Curve"):
    plt.plot(range(len(loss_history)), loss_history, color="green")
    plt.xlabel("Epochs")
    plt.ylabel("Loss (MSE)")
    plt.title(title)
    plt.show()


# ------------------ Linear Regression Class ------------------
class LinearRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000, method="batch", batch_size=32):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.method = method
        self.batch_size = batch_size
        self.weights = None
        self.bias = None
        self.loss_history = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        if self.method == "batch":
            self._batch_gradient_descent(X, y)
        elif self.method == "stochastic":
            self._stochastic_gradient_descent(X, y)
        elif self.method == "mini-batch":
            self._mini_batch_gradient_descent(X, y)
        else:
            raise ValueError("Invalid method. Choose 'batch', 'stochastic', or 'mini-batch'.")

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

    def get_weights(self):
        return self.weights

    def get_bias(self):
        return self.bias

    # ------------------ Gradient Descent Variants ------------------
    def _batch_gradient_descent(self, X, y):
        n_samples = X.shape[0]
        for _ in range(self.n_iterations):
            y_pred = self.predict(X)
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            loss = mean_squared_error(y, y_pred)
            self.loss_history.append(loss)

    def _stochastic_gradient_descent(self, X, y):
        n_samples = X.shape[0]
        for _ in range(self.n_iterations):
            for i in range(n_samples):
                xi = X[i, :].reshape(1, -1)
                yi = y[i]
                y_pred = self.predict(xi)

                dw = np.dot(xi.T, (y_pred - yi))
                db = y_pred - yi

                self.weights -= self.learning_rate * dw.flatten()
                self.bias -= self.learning_rate * db

            y_full_pred = self.predict(X)
            loss = mean_squared_error(y, y_full_pred)
            self.loss_history.append(loss)

    def _mini_batch_gradient_descent(self, X, y):
        n_samples = X.shape[0]
        for _ in range(self.n_iterations):
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            for i in range(0, n_samples, self.batch_size):
                Xi = X_shuffled[i:i + self.batch_size]
                yi = y_shuffled[i:i + self.batch_size]

                y_pred = self.predict(Xi)

                dw = (1 / len(Xi)) * np.dot(Xi.T, (y_pred - yi))
                db = (1 / len(Xi)) * np.sum(y_pred - yi)

                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db

            y_full_pred = self.predict(X)
            loss = mean_squared_error(y, y_full_pred)
            self.loss_history.append(loss)


# ------------------ Main Execution ------------------
if __name__ == "__main__":
    # Dummy dataset: y = 2x + 1
   # Dataset: y = x^2 + 2x + 3
    X = np.array([[7], [8], [9], [40], [50], [60],[70]])
    y = np.array([3,2,3,9,10,11,40])


    # ---- Batch GD ----
    print("---- Batch Gradient Descent ----")
    model_batch = LinearRegression(learning_rate=0.01, n_iterations=1000, method="batch")
    model_batch.fit(X, y)
    pred_batch = model_batch.predict(X)
    print("Weights:", model_batch.get_weights())
    print("Bias:", model_batch.get_bias())
    print("MSE:", mean_squared_error(y, pred_batch))
    plot_regression(X, y, pred_batch, title="Batch Gradient Descent")
    plot_loss_curve(model_batch.loss_history, title="Batch GD Loss Curve")

    # ---- Stochastic GD ----
    print("\n---- Stochastic Gradient Descent ----")
    model_sgd = LinearRegression(learning_rate=0.01, n_iterations=100, method="stochastic")
    model_sgd.fit(X, y)
    pred_sgd = model_sgd.predict(X)
    print("Weights:", model_sgd.get_weights())
    print("Bias:", model_sgd.get_bias())
    print("MSE:", mean_squared_error(y, pred_sgd))
    plot_regression(X, y, pred_sgd, title="Stochastic Gradient Descent")
    plot_loss_curve(model_sgd.loss_history, title="SGD Loss Curve")

    # ---- Mini-Batch GD ----
    print("\n---- Mini-Batch Gradient Descent ----")
    model_mbgd = LinearRegression(learning_rate=0.01, n_iterations=1000, method="mini-batch", batch_size=2)
    model_mbgd.fit(X, y)
    pred_mbgd = model_mbgd.predict(X)
    print("Weights:", model_mbgd.get_weights())
    print("Bias:", model_mbgd.get_bias())
    print("MSE:", mean_squared_error(y, pred_mbgd))
    plot_regression(X, y, pred_mbgd, title="Mini-Batch Gradient Descent")
    plot_loss_curve(model_mbgd.loss_history, title="Mini-Batch GD Loss Curve")
