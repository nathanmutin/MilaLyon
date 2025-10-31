import numpy as np
import matplotlib.pyplot as plt


def mae(y, tx, w):
    """Compute the Mean Absolute Error (MAE)

    Args:
    y (np.array): shape=(N,) target values
    tx (np.array): shape=(N,D) feature matrix
    w (np.array): shape=(D,) weights

    Returns:
        float: MAE loss value
    """
    return np.mean(np.abs(tx @ w - y))


def mse(y, tx, w):
    """Compute the Mean Squared Error (MSE) with a factor 1/2.

    Args:
        y (np.array): shape=(N,) target values
        tx (np.array): shape=(N,D) feature matrix
        w (np.array): shape=(D,) weights

    Returns:
        float: MSE loss value
    """
    return np.mean((tx @ w - y) ** 2) / 2


def grid_search(y, tx, grid_w0, grid_w1):
    """Grid search for optimal weights in linear regression.

    Args:
        y (np.array): shape=(N,) target values
        tx (np.array): shape=(N,D) feature matrix
        grid_w0 (np.array): 1D array of candidate values for weight w0
        grid_w1 (np.array): 1D array of candidate values for weight w1

    Returns:
        tuple: (best_w, best_loss)
            best_w (np.array): shape=(D,) optimal weights
            best_loss (float): minimal loss value
    """
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


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma, return_history=False):
    """Linear regression using gradient descent.

    Args:
        y (np.array): shape=(N,) target values
        tx (np.array): shape=(N,D) feature matrix
        initial_w (np.array): shape=(D,) initial weights
        max_iters (int): maximum number of gradient descent iterations
        gamma (float): learning rate
        return_history (bool): if True, returns the history of weights and losses

    Returns:
        w (np.array): shape=(D,) final weights
        loss (float): final loss value

        if return_history:
            weights (list of np.array): history of weights
            losses (list of float): history of loss values
    """
    weights = [initial_w]
    losses = [mse(y, tx, initial_w)]
    for _ in range(max_iters):
        # gradient descent step
        # d(mse)/dw = 1/N * X^T (Xw - y)
        weights.append(weights[-1] - gamma * tx.T @ (tx @ weights[-1] - y) / len(y))
        losses.append(mse(y, tx, weights[-1]))

    if return_history:
        return weights, losses

    return weights[-1], losses[-1]


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma, return_history=False):
    """Linear regression using stochastic gradient descent.

    Args:
        y (np.array): shape=(N,) target values
        tx (np.array): shape=(N,D) feature matrix
        initial_w (np.array): shape=(D,) initial weights
        max_iters (int): maximum number of stochastic gradient descent iterations
        gamma (float): learning rate
        return_history (bool): if True, returns the history of weights and losses

    Returns:
        w (np.array): shape=(D,) final weights
        loss (float): final loss value

        if return_history:
            weights (list of np.array): history of weights
            losses (list of float): history of loss values
    """
    weights = [initial_w]
    losses = [mse(y, tx, initial_w)]
    for _ in range(max_iters):
        # pick a random sample
        random_index = np.random.randint(len(y))

        # stochastic gradient descent step
        # d(mse)/dw =  x_i (x_i w - y_i)
        weights.append(
            weights[-1]
            - gamma
            * tx[random_index]
            * (tx[random_index] @ weights[-1] - y[random_index])
        )
        losses.append(mse(y, tx, weights[-1]))

    if return_history:
        return weights, losses

    return weights[-1], losses[-1]


def least_squares(y, tx):
    """Least squares regression using normal equations.

    Args:
        y (np.array): shape=(N,) target values
        tx (np.array): shape=(N,D) feature matrix

    Returns:
        w (np.array): shape=(D,) optimal weights
        loss (float): minimal loss value
    """
    # Compute w such that
    # X^T X w = X^T y
    w = np.linalg.solve(tx.T @ tx, tx.T @ y)

    return w, mse(y, tx, w)


def ridge_regression(y, tx, lambda_):
    """Ridge regression using normal equations.

    Args:
        y (np.array): shape=(N,) target values
        tx (np.array): shape=(N,D) feature matrix
        lambda_ (float): regularization parameter

    Returns:
        w (np.array): shape=(D,) optimal weights
        loss (float): minimal loss value
    """
    # Compute w such that
    # (X^T X + 2NλI) w = X^T y
    N, D = tx.shape
    w = np.linalg.solve(tx.T @ tx + 2 * N * lambda_ * np.eye(D), tx.T @ y)
    return w, mse(y, tx, w)


def sigmoid(t):
    """Apply the sigmoid function on t.

    Args:
        t (np.array): input data

    Returns:
        np.array: sigmoid(t)
    """
    return 1 / (1 + np.exp(-t))


def logistic_negative_log_likelihood(y, tx, w):
    """Compute the negative log likelihood for logistic regression.

    Args:
        y (np.array): shape=(N,) target values
        tx (np.array): shape=(N,D) feature matrix
        w (np.array): shape=(D,) weights
    Returns:
        float: negative log likelihood loss value
    """
    # - mean(y log(pred) + (1-y) log(1-pred))
    # with pred = sigmoid(tx @ w)
    # More efficient implementation after a few algebraic manipulations
    return np.mean(-y * (tx @ w) + np.log(1 + np.exp(tx @ w)))


def logistic_regression(y, tx, initial_w, max_iters, gamma, return_history=False):
    """Logistic regression using gradient descent (y ∈ {0,1}).

    Args:
        y (np.array): shape=(N,) target values
        tx (np.array): shape=(N,D) feature matrix
        initial_w (np.array): shape=(D,) initial weights
        max_iters (int): maximum number of gradient descent iterations
        gamma (float): learning rate
        return_history (bool): if True, returns the history of weights and losses

    Returns:
        w (np.array): shape=(D,) final weights
        loss (float): final loss value

        if return_history:
            weights (list of np.array): history of weights
            losses (list of float): history of loss values
    """
    weights = [initial_w]
    losses = [logistic_negative_log_likelihood(y, tx, initial_w)]

    # Minimize the negative log likelihood
    for _ in range(max_iters):
        # Compute gradient
        pred = sigmoid(tx @ weights[-1])
        gradient = tx.T @ (pred - y) / len(y)  # gradient of the loss
        # Gradient descent step
        weights.append(weights[-1] - gamma * gradient)
        losses.append(logistic_negative_log_likelihood(y, tx, weights[-1]))

    if return_history:
        return weights, losses

    return weights[-1], losses[-1]


def reg_logistic_regression(
    y, tx, lambda_, initial_w, max_iters, gamma, return_history=False
):
    """Regularized logistic regression using gradient descent
        (y ∈ {0,1}, with regularization term λ∥w∥2)

    Args:
        y (np.array): shape=(N,) target values
        tx (np.array): shape=(N,D) feature matrix
        lambda_ (float): regularization parameter
        initial_w (np.array): shape=(D,) initial weights
        max_iters (int): maximum number of gradient descent iterations
        gamma (float): learning rate
        return_history (bool): if True, returns the history of weights and losses

    Returns:
        w (np.array): shape=(D,) final weights
        loss (float): final loss value

        if return_history:
            weights (list of np.array): history of weights
            losses (list of float): history of loss values
    """
    # Minimize the regularized negative log likelihood
    weights = [initial_w]
    losses = [logistic_negative_log_likelihood(y, tx, initial_w)]
    for _ in range(max_iters):
        # gradient descent step
        pred = sigmoid(tx @ weights[-1])  # sigmoid function
        gradient = (
            tx.T @ (pred - y) / len(y) + 2 * lambda_ * weights[-1]
        )  # gradient of the regularized loss
        weights.append(weights[-1] - gamma * gradient)
        losses.append(logistic_negative_log_likelihood(y, tx, weights[-1]))

    if return_history:
        return weights, losses

    return weights[-1], losses[-1]


def weighted_reg_logistic_regression(
    y, tx, lambda_, sample_weights, initial_w, max_iters, gamma, return_history=False
):
    """Weighted and regularized logistic regression using gradient descent
       Same as reg_logistic_regression but with the learning rate is scaled by
       sample weights

    Args:
        y (np.array): shape=(N,) target values
        tx (np.array): shape=(N,D) feature matrix
        lambda_ (float): regularization parameter
        sample_weights (np.array): shape=(N,) sample weights
        initial_w (np.array): shape=(D,) initial weights
        max_iters (int): maximum number of gradient descent iterations
        gamma (float): learning rate
        return_history (bool): if True, returns the history of weights and losses

    Returns:
        w (np.array): shape=(D,) final weights
        loss (float): final loss value

        if return_history:
            weights (list of np.array): history of weights
            losses (list of float): history of loss values
    """
    # Normalize sample weights such that they sum to the number of samples
    sample_weights = sample_weights * len(y) / np.sum(sample_weights)

    # Minimize the regularized negative log likelihood
    weights = [initial_w]
    losses = [logistic_negative_log_likelihood(y, tx, initial_w)]
    for _ in range(max_iters):
        # gradient descent step
        pred = sigmoid(tx @ weights[-1])  # sigmoid function
        gradient = (
            tx.T @ (sample_weights * (pred - y)) / len(y) + 2 * lambda_ * weights[-1]
        )  # gradient of the regularized loss
        weights.append(weights[-1] - gamma * gradient)
        losses.append(logistic_negative_log_likelihood(y, tx, weights[-1]))

    if return_history:
        return weights, losses

    return weights[-1], losses[-1]


def predict_labels_logistic(tx, w, threshold=0.5):
    """Generate class predictions for logistic regression.

    Args:
        tx (np.array): feature matrix
        w (np.array): weights
        threshold (float): classification threshold

    Returns:
        np.array: predicted class labels (0/1)
    """
    pred = sigmoid(tx @ w)
    return (pred >= threshold).astype(int)


def build_poly(x, degree, to_expand=None):
    """Polynomial basis functions for input data x, for j=0 up to j=degree.

    Args:
        x (np.array): shape=(N,) input data
        degree (int): polynomial degree
        to_expand (np.array of bool): shape=(D,) indicating which features to expand, if None all features are expanded

    Returns:
        np.array: shape=(N,degree+1) matrix of polynomial features
    """
    assert degree >= 1, "Degree must be at least 1"

    # Handle to_expand default
    if to_expand is None:
        to_expand = np.full(x.shape[1], True, dtype=bool)

    N = x.shape[0]
    # degree 0 (bias term)
    poly = [np.ones((N, 1))]
    # degree 1 (original features)
    poly.append(x)
    # degree 2 to degree max
    for d in range(2, degree + 1):
        # Only expand features that are marked in to_expand
        # <=> only add x[:, j] ** d if to_expand[j] is True
        for j in range(x.shape[1]):
            if to_expand[j]:
                poly.append((x[:, j] ** d).reshape(-1, 1))
    return np.concatenate(poly, axis=1)
