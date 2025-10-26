import numpy as np

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
    best_loss = float('inf')
    best_w = None

    for w0 in grid_w0:
        for w1 in grid_w1:
            w = np.array([w0, w1])
            loss = mse(y, tx, w)
            if loss < best_loss:
                best_loss = loss
                best_w = w

    return best_w, best_loss

def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """Linear regression using gradient descent.

    Args:
        y (np.array): shape=(N,) target values
        tx (np.array): shape=(N,D) feature matrix
        initial_w (np.array): shape=(D,) initial weights
        max_iters (int): maximum number of gradient descent iterations
        gamma (float): learning rate

    Returns:
        w (np.array): shape=(D,) final weights
        loss (float): final loss value
    """
    w = initial_w
    for _ in range(max_iters):
        # gradient descent step
        # d(mse)/dw = 1/N * X^T (Xw - y)
        w -= gamma * tx.T @ (tx @ w - y) / len(y)

    return w, mse(y, tx, w)

def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """Linear regression using stochastic gradient descent.

    Args:
        y (np.array): shape=(N,) target values
        tx (np.array): shape=(N,D) feature matrix
        initial_w (np.array): shape=(D,) initial weights
        max_iters (int): maximum number of stochastic gradient descent iterations
        gamma (float): learning rate

    Returns:
        w (np.array): shape=(D,) final weights
        loss (float): final loss value
    """
    w = initial_w
    for _ in range(max_iters):
        # pick a random sample
        random_index = np.random.randint(len(y))

        # stochastic gradient descent step
        # d(mse)/dw =  x_i (x_i w - y_i)
        w -= gamma * tx[random_index] * (tx[random_index] @ w - y[random_index])

    return w, mse(y, tx, w)

def build_poly(x, degree):
    """Polynomial basis functions for input data x, for j=0 up to j=degree.

    Args:
        x (np.array): shape=(N,) input data
        degree (int): polynomial degree
        
    Returns:
        np.array: shape=(N,degree+1) matrix of polynomial features
    """
    
    N = x.shape[0]
    poly = np.ones((N, degree + 1))
    for d in range(1, degree + 1):
        poly[:, d] = x ** d
    return poly 

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
    return np.mean(- y * (tx @ w) + np.log(1 + np.exp(tx @ w)))

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Logistic regression using gradient descent (y ∈ {0,1}).

    Args:
        y (np.array): shape=(N,) target values
        tx (np.array): shape=(N,D) feature matrix
        initial_w (np.array): shape=(D,) initial weights
        max_iters (int): maximum number of gradient descent iterations
        gamma (float): learning rate

    Returns:
        w (np.array): shape=(D,) final weights
        loss (float): final loss value
    """
    # Minimize the negative log likelihood
    w = initial_w
    for _ in range(max_iters):
        # gradient descent step
        pred = sigmoid(tx @ w)  # sigmoid function
        gradient = tx.T @ (pred - y) / len(y)  # gradient of the loss
        w -= gamma * gradient

    return w, logistic_negative_log_likelihood(y, tx, w)

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Regularized logistic regression using gradient descent
        (y ∈ {0,1}, with regularization term λ∥w∥2)

    Args:
        y (np.array): shape=(N,) target values
        tx (np.array): shape=(N,D) feature matrix
        lambda_ (float): regularization parameter
        initial_w (np.array): shape=(D,) initial weights
        max_iters (int): maximum number of gradient descent iterations
        gamma (float): learning rate

    Returns:
        w (np.array): shape=(D,) final weights
        loss (float): final loss value
    """
    # Minimize the regularized negative log likelihood
    w = initial_w
    for _ in range(max_iters):
        # gradient descent step
        pred = sigmoid(tx @ w)  # sigmoid function
        gradient = tx.T @ (pred - y) / len(y) + 2 * lambda_ * w  # gradient of the regularized loss
        w -= gamma * gradient

    return w, logistic_negative_log_likelihood(y, tx, w)

def weighted_reg_logistic_regression(y, tx, lambda_, sample_weights, initial_w, max_iters, gamma):
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

    Returns:
        w (np.array): shape=(D,) final weights
        loss (float): final loss value
    """
    # Minimize the regularized negative log likelihood
    w = initial_w
    for _ in range(max_iters):
        # gradient descent step
        pred = sigmoid(tx @ w)  # sigmoid function
        gradient = tx.T @ (sample_weights * (pred - y)) / len(y) + 2 * lambda_ * w  # gradient of the regularized loss
        w -= gamma * gradient

    return w, logistic_negative_log_likelihood(y, tx, w)

def predict_labels_logistic(tx, w, threshold=0.5):
    pred = sigmoid(tx @ w)
    return (pred >= threshold).astype(int)

def compute_scores(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score
    }