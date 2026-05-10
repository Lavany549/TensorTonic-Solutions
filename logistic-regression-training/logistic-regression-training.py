import numpy as np
from sklearn.metrics import accuracy_score
def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    # Write code here
    m = X.shape[0]
    N = X.shape[1]
    w = np.zeros(N)
    b = 0.0

    for i in range(steps):
        z = X @ w + b
        y_pred = _sigmoid(z)
        y_pred = (y_pred >= 0.5).astype(int)

        dw = (1 / m) * (X.T @ (y_pred - y))
        db = (1 / m) * np.sum(y_pred - y)

        w = w - lr * dw
        b = b - lr * db

        if i % 100 == 0:
            pred_labels = (y_pred >= 0.5).astype(int)
            print("accuracy score:", accuracy_score(y, pred_labels))

    return (w, b)