import numpy as np
from helpers import batch_iter

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

def compute_loss(y, tx, w):
    """Calculate the loss using MSE.

    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        w: shape=(2,). The vector of model parameters.

    Returns:
        the value of the loss (a scalar), corresponding to the input parameters w.
    """
    # ***************************************************
    return 1/tx.shape[0]*np.sum((y-tx@w)**2)
    # ***************************************************


def compute_gradient(y, tx, w):
    """Computes the gradient at w.

    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        w: shape=(2, ). The vector of model parameters.

    Returns:
        An array of shape (2, ) (same shape as w), containing the gradient of the loss at w.
    """
    # ***************************************************
    e = y-tx@w
    loss = -1/tx.shape[0]*tx.T@e
    return loss
    # ***************************************************

def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient at w from a data sample batch of size B, where B < N, and their corresponding labels.

    Args:
        y: numpy array of shape=(B, )
        tx: numpy array of shape=(B,2)
        w: numpy array of shape=(2, ). The vector of model parameters.

    Returns:
        A numpy array of shape (2, ) (same shape as w), containing the stochastic gradient of the loss at w.
    """

    # ***************************************************
    return compute_gradient(y,tx,w)
    # ***************************************************

def least_squares(y, tx):
    """Calculate the least squares solution.
       returns mse, and optimal weights.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        mse: scalar.

    >>> least_squares(np.array([0.1,0.2]), np.array([[2.3, 3.2], [1., 0.1]]))
    (array([ 0.21212121, -0.12121212]), 8.666684749742561e-33)
    """
    # ***************************************************
    A = tx.T@tx
    b= tx.T@y
    w = np.linalg.solve(A,b)
    e = y-tx@w
    loss = (e@e)/2/tx.shape[0]
    return w,float(loss)
    # ***************************************************

def mean_squared_error(y, tx, initial_w, max_iters, gamma, verbose=False):
    """The Gradient Descent (GD) algorithm.

    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        initial_w: shape=(2, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize

    Returns:
        losses: a list of length max_iters containing the loss value (scalar) for each iteration of GD
        ws: a list of length max_iters containing the model parameters as numpy arrays of shape (2, ), for each iteration of GD
    """
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        # ***************************************************
        gradient = compute_gradient(y,tx,w)
        loss = compute_loss(y,tx,w)
        # ***************************************************
        # ***************************************************
        w = w - gamma*gradient
        # ***************************************************
        # store w and loss
        ws.append(w)
        losses.append(loss)
        if verbose:
            print(
                "GD iter. {bi}/{ti}: loss={l}, w0={w0}, w1={w1}".format(
                    bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]
                )
            )

    return losses, ws

    
def mean_squared_error_sgd(y, tx, initial_w, batch_size, max_iters, gamma, verbose=False):
    """The Stochastic Gradient Descent algorithm (SGD).

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        initial_w: numpy array of shape=(2, ). The initial guess (or the initialization) for the model parameters
        batch_size: a scalar denoting the number of data points in a mini-batch used for computing the stochastic gradient
        max_iters: a scalar denoting the total number of iterations of SGD
        gamma: a scalar denoting the stepsize

    Returns:
        losses: a list of length max_iters containing the loss value (scalar) for each iteration of SGD
        ws: a list of length max_iters containing the model parameters as numpy arrays of shape (2, ), for each iteration of SGD
    """

    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w

    for n_iter in range(max_iters):
        # ***************************************************
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size):
            loss = compute_loss(minibatch_y,minibatch_tx,w)
            gradient = compute_stoch_gradient(minibatch_y,minibatch_tx,w)
            losses.append(loss)
            w = w - gamma*gradient
            ws.append(w)
        # ***************************************************
        if verbose:
            print(
                "SGD iter. {bi}/{ti}: loss={l}, w0={w0}, w1={w1}".format(
                    bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]
                )
            )
    return losses, ws

def ridge_regression(y, tx, lambda_):
    """implement ridge regression.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        lambda_: scalar.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        loss: scalar.

    >>> ridge_regression(np.array([0.1,0.2]), np.array([[2.3, 3.2], [1., 0.1]]), 0)
    array([ 0.21212121, -0.12121212])
    >>> ridge_regression(np.array([0.1,0.2]), np.array([[2.3, 3.2], [1., 0.1]]), 1)
    array([0.03947092, 0.00319628])
    """
    # ***************************************************
    D = tx.shape[1]
    lambdaI = 2*tx.shape[0]*lambda_*np.eye(D)
    A = tx.T@tx + lambdaI
    b= tx.T@y
    w = np.linalg.solve(A,b)
    return w,compute_loss(y,tx,w)
    # ***************************************************

#Example run
if __name__ == "__main__":
    y = np.array([0.1,0.2])
    tx = np.array([[2.3, 3.2], [1., 0.1]])
    initial_w = np.array([0.,0.])
    max_iters = 10
    batch_size = 1
    gamma = 0.01
    print("Least squares:")
    print(least_squares(y,tx))
    print("Gradient Descent:")
    print(mean_squared_error(y,tx,initial_w,max_iters,gamma))
    print("Stochastic Gradient Descent:")
    print(mean_squared_error_sgd(y,tx,initial_w, batch_size, max_iters,gamma))
    print("Ridge Regression:")
    print(ridge_regression(y,tx,1))