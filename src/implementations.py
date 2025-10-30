"""
implementations.py

This module implements core machine learning functions for regression and classification tasks.

Functions:
- mae, mse: Loss functions for regression.
- grid_search: Brute-force search for optimal linear regression weights.
- mean_squared_error_gd, mean_squared_error_sgd: Linear regression using gradient descent or stochastic gradient descent.
- least_squares, ridge_regression: Linear regression using normal equations.
- sigmoid: Sigmoid function for logistic regression.
- logistic_negative_log_likelihood, logistic_regression: Logistic regression and its loss.
- reg_logistic_regression, weighted_reg_logistic_regression: Regularized logistic regression variants.
- predict_labels_logistic: Predict class labels for logistic regression.
- build_poly: Polynomial feature expansion for input data.
"""

import numpy as np
import matplotlib.pyplot as plt


# --------------------- Loss functions --------------------- #

def mae(y, tx, w):
    """Compute the Mean Absolute Error (MAE)."""
    return np.mean(np.abs(tx @ w - y))


def mse(y, tx, w):
    """Compute the Mean Squared Error (MSE) with factor 1/2."""
    return np.mean((tx @ w - y) ** 2) / 2


# --------------------- Grid search --------------------- #

def grid_search(y, tx, grid_w0, grid_w1):
    """Grid search for optimal weights in linear regression."""
    best_loss = float("inf")
    best_w = None
    for w0 in grid_w0:
        for w1 in grid_w1:
            w = np.array([w0, w1])
            loss = mse(y, tx, w)
            if loss < best_loss:
                best_loss = loss
                best_w = w
    return best_w, best_loss


# --------------------- Linear regression --------------------- #

def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma, return_history=False):
    """Linear regression using gradient descent."""
    weights = [initial_w]
    losses = [mse(y, tx, initial_w)]
    for _ in range(max_iters):
        # Gradient step
        gradient = tx.T @ (tx @ weights[-1] - y) / len(y)
        weights.append(weights[-1] - gamma * gradient)
        losses.append(mse(y, tx, weights[-1]))
    return (weights, losses) if return_history else (weights[-1], losses[-1])


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma, return_history=False):
    """Linear regression using stochastic gradient descent."""
    weights = [initial_w]
    losses = [mse(y, tx, initial_w)]
    for _ in range(max_iters):
        i = np.random.randint(len(y))
        gradient = tx[i] * (tx[i] @ weights[-1] - y[i])
        weights.append(weights[-1] - gamma * gradient)
        losses.append(mse(y, tx, weights[-1]))
    return (weights, losses) if return_history else (weights[-1], losses[-1])


def least_squares(y, tx):
    """Linear regression using normal equations."""
    w = np.linalg.solve(tx.T @ tx, tx.T @ y)
    return w, mse(y, tx, w)


def ridge_regression(y, tx, lambda_):
    """Ridge regression using normal equations."""
    N, D = tx.shape
    w = np.linalg.solve(tx.T @ tx + 2 * N * lambda_ * np.eye(D), tx.T @ y)
    return w, mse(y, tx, w)


# --------------------- Logistic regression --------------------- #

def sigmoid(t):
    """Compute sigmoid function."""
    return 1 / (1 + np.exp(-t))


def logistic_negative_log_likelihood(y, tx, w):
    """Compute negative log likelihood loss for logistic regression."""
    return np.mean(-y * (tx @ w) + np.log(1 + np.exp(tx @ w)))


def logistic_regression(y, tx, initial_w, max_iters, gamma, return_history=False):
    """Logistic regression using gradient descent."""
    weights = [initial_w]
    losses = [logistic_negative_log_likelihood(y, tx, initial_w)]
    for _ in range(max_iters):
        pred = sigmoid(tx @ weights[-1])
        gradient = tx.T @ (pred - y) / len(y)
        weights.append(weights[-1] - gamma * gradient)
        losses.append(logistic_negative_log_likelihood(y, tx, weights[-1]))
    return (weights, losses) if return_history else (weights[-1], losses[-1])


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma, return_history=False):
    """Regularized logistic regression (L2) using gradient descent."""
    weights = [initial_w]
    losses = [logistic_negative_log_likelihood(y, tx, initial_w)]
    for _ in range(max_iters):
        pred = sigmoid(tx @ weights[-1])
        gradient = tx.T @ (pred - y) / len(y) + 2 * lambda_ * weights[-1]
        weights.append(weights[-1] - gamma * gradient)
        losses.append(logistic_negative_log_likelihood(y, tx, weights[-1]))
    return (weights, losses) if return_history else (weights[-1], losses[-1])


def weighted_reg_logistic_regression(y, tx, lambda_, sample_weights, initial_w, max_iters, gamma, return_history=False):
    """Weighted and regularized logistic regression using gradient descent."""
    # Scale sample weights to sum to N
    sample_weights = sample_weights * len(y) / np.sum(sample_weights)
    weights = [initial_w]
    losses = [logistic_negative_log_likelihood(y, tx, initial_w)]
    for _ in range(max_iters):
        pred = sigmoid(tx @ weights[-1])
        gradient = tx.T @ (sample_weights * (pred - y)) / len(y) + 2 * lambda_ * weights[-1]
        weights.append(weights[-1] - gamma * gradient)
        losses.append(logistic_negative_log_likelihood(y, tx, weights[-1]))
    return (weights, losses) if return_history else (weights[-1], losses[-1])


# --------------------- Prediction and feature expansion --------------------- #

def predict_labels_logistic(tx, w, threshold=0.5):
    """Generate predicted class labels (0/1) for logistic regression."""
    pred = sigmoid(tx @ w)
    return (pred >= threshold).astype(int)


def build_poly(x, degree, to_expand=None):
    """
    Polynomial feature expansion.

    Args:
        x (np.array): Input data, shape=(N,D)
        degree (int): Maximum polynomial degree
        to_expand (np.array of bool, optional): Which features to expand. Default all.

    Returns:
        np.array: Expanded feature matrix
    """
    assert degree >= 1, "Degree must be at least 1"
    if to_expand is None:
        to_expand = np.full(x.shape[1], True, dtype=bool)

    N = x.shape[0]
    poly = [np.ones((N, 1)), x]  # bias + degree 1
    for d in range(2, degree + 1):
        for j in range(x.shape[1]):
            if to_expand[j]:
                poly.append((x[:, j] ** d).reshape(-1, 1))
    return np.concatenate(poly, axis=1)
