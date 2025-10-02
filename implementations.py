import numpy as np

def normalize(x, x_test=None):
    """Normalizes the data set.

    Args:
        x (np.array): shape=(N,D) feature matrix

    Returns:
        np.array: shape=(N,D) normalized feature matrix
    """
    mean_x = np.mean(x, axis=0)
    std_x = np.std(x, axis=0)
    std_x[std_x == 0] = 1.0  # avoid division by zero    
    
    x = (x - mean_x) / std_x
    
    if x_test is not None:
        x_test = (x_test - mean_x) / std_x
        return x, x_test
    return x

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
    raise NotImplementedError

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
    raise NotImplementedError


def mean_imputation(x_train, x_test, train_columns):
    """Impute missing values with the mean of each feature. 
    Drops columns that are entirely NaN.

    Args:
        x_train (np.array): shape=(N,D) training feature matrix with NaNs for missing values
        x_test (np.array): shape=(M,D) test feature matrix with NaNs for missing values

    Returns:
        tuple: (x_train_imputed, x_test_imputed)
            x_train_imputed (np.array): training feature matrix with imputed values
            x_test_imputed (np.array): test feature matrix with imputed values
    """
    # Mask for columns that are not entirely NaN
    valid_mask = ~np.isnan(x_train).all(axis=0)
        
    # Print indices of dropped columns
    dropped_indices = np.where(~valid_mask)[0]
    print("Dropped columns (all NaN):", [ train_columns[i] for i in dropped_indices ])
    
    
    # Keep only valid columns in train and test
    x_train = x_train[:, valid_mask]
    x_test = x_test[:, valid_mask]

    # Compute means on the training set (ignoring NaNs)
    mean_x = np.nanmean(x_train, axis=0)

    # Impute training set
    inds_train = np.where(np.isnan(x_train))
    x_train[inds_train] = np.take(mean_x, inds_train[1])

    # Impute test set using train means
    inds_test = np.where(np.isnan(x_test))
    x_test[inds_test] = np.take(mean_x, inds_test[1])
    
    print("New shape after mean imputation:", x_train.shape)

    return x_train, x_test
