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
        weights.append(weights[-1] - gamma * tx[random_index] * (tx[random_index] @ weights[-1] - y[random_index]))
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
    return np.mean(- y * (tx @ w) + np.log(1 + np.exp(tx @ w)))

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

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma, return_history=False):
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
        gradient = tx.T @ (pred - y) / len(y) + 2 * lambda_ * weights[-1]  # gradient of the regularized loss
        weights.append(weights[-1] - gamma * gradient)
        losses.append(logistic_negative_log_likelihood(y, tx, weights[-1]))

    if return_history:
        return weights, losses

    return weights[-1], losses[-1]

def weighted_reg_logistic_regression(y, tx, lambda_, sample_weights, initial_w, max_iters, gamma, return_history=False):
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
        gradient = tx.T @ (sample_weights * (pred - y)) / len(y) + 2 * lambda_ * weights[-1]  # gradient of the regularized loss
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

def compute_scores(y_true, y_pred):
    """Compute accuracy, precision, recall, and F1-score.
    
    Args:
        y_true (np.array): true labels (0/1)
        y_pred (np.array): predicted labels (0/1)
    
    Returns:
        dict: dictionary containing 'accuracy', 'precision', 'recall', and 'f1-score'
    """
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
    
def best_threshold(y_true, tx, w, thresholds=np.linspace(0., 1., 100)):
    """Find threshold maximizing F1-score.
    
    Args:
        y_true (np.array): true labels (0/1)
        tx (np.array): feature matrix
        w (np.array): weights
        thresholds (np.array): array of thresholds to evaluate
        
    Returns:
        best_t (float): threshold yielding highest F1-score
        best_f1 (float): highest F1-score corresponding to best_t
    """
    best_t, best_f1 = 0.5, 0
    for t in thresholds:
        y_pred = predict_labels_logistic(tx,w,t)
        scores = compute_scores(y_true, y_pred)
        f1 = scores['f1_score']
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t, best_f1
    
def k_fold_indices(N, k, seed=42):
    """Generate indices for k-fold cross-validation.
    
    Args:
        N (int): number of samples in the dataset
        k (int): number of folds
        seed (int): random seed for reproducibility
    
    Returns:
        list of np.array: list containing k arrays of indices for each fold
    """
    np.random.seed(seed)
    indices = np.random.permutation(N)
    fold_sizes = np.full(k, N // k, dtype=int)
    fold_sizes[:N % k] += 1
    current = 0
    folds = []
    for fold_size in fold_sizes:
        start, stop = current, current + fold_size
        folds.append(indices[start:stop])
        current = stop
    return folds

def plot_training_validation_performance(x_train, y_train, x_val, y_val, weights, losses):
    """
    Trains logistic regression, evaluates F1 and accuracy across thresholds,
    and plots both threshold and loss curves for training and validation.

    Args:
        x_train (np.array): training features
        y_train (np.array): training labels (0/1)
        x_val (np.array): validation features
        y_val (np.array): validation labels (0/1)
        weights (np.array): weights history of the logistic regression model
        losses (list): training loss history
    """
    # ---- Threshold sweep ----
    thresholds = np.arange(0.0, 1.0 + 0.01, 0.01)
    f1_train, acc_train = [], []
    f1_val, acc_val = [], []

    for t in thresholds:
        y_pred_train = predict_labels_logistic(x_train, weights[-1], t)
        y_pred_val = predict_labels_logistic(x_val, weights[-1], t)

        scores_train = compute_scores(y_train, y_pred_train)
        scores_val = compute_scores(y_val, y_pred_val)

        f1_train.append(scores_train['f1_score'])
        acc_train.append(scores_train['accuracy'])
        f1_val.append(scores_val['f1_score'])
        acc_val.append(scores_val['accuracy'])

    # ---- Best threshold ----
    best_idx = np.argmax(f1_val)
    best_threshold = thresholds[best_idx]
    print(f"✅ Best threshold: {best_threshold:.2f} | F1_val = {f1_val[best_idx]:.3f} | Acc_val = {acc_val[best_idx]:.3f}")

    # ---- Plot F1 & Accuracy vs Threshold ----
    plt.figure(figsize=(8,5))
    plt.plot(thresholds, acc_train, label='Train Accuracy',  marker='o',  markersize=3, color='blue')
    plt.plot(thresholds, acc_val, label='Val Accuracy',  marker='x', markersize=3, color='green')
    plt.plot(thresholds, f1_train, label='Train F1', marker='o',markersize = 3, color='orange')
    plt.plot(thresholds, f1_val, label='Val F1',  marker='x', markersize=3, color='red')
    plt.axvline(best_threshold, color='red', linestyle=':', label=f'Best Threshold ({best_threshold:.2f})')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title('F1 & Accuracy vs Threshold (Train & Validation)')
    plt.legend()
    plt.grid(alpha=0.5)
    plt.show()
    
    # ---- Plot F1 & Loss vs Iterations ----
    f1_scores = []
    for i in weights:
        y_train_pred = predict_labels_logistic(x_train, i, best_threshold)
        scores = compute_scores(y_train, y_train_pred)
        f1_scores.append(scores['f1_score'])
          
    fig, ax1 = plt.subplots(figsize=(7, 5))

    ax1.plot(losses, color='tab:red', label='Loss')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Loss', color='tab:red')
    ax1.tick_params(axis='y', labelcolor='tab:red')

    ax2 = ax1.twinx()
    ax2.plot(f1_scores, color='tab:blue', label='F1 Score')
    ax2.set_ylabel('F1 Score', color='tab:blue')
    ax2.tick_params(axis='y', labelcolor='tab:blue')
    plt.title('Training Loss and F1 Score over Iterations')

    plt.tight_layout()
    plt.show()

def cross_validate_hyperparameter(y, tx, train_func, hyperparams, k=5):
    """Generic k-fold CV to select the best hyperparameter using F1 score.
    
    Args:
        y (np.array): target vector
        tx (np.array): feature matrix
        train_func (callable): training function that accepts (y_train, x_train, hyperparams) and returns (w, loss)
        hyperparams (np.ndarray): hyperparameters to evaluate
        k (int): number of folds
        plot (bool): whether to plot the results
        
    Returns:
        best_hyperparam: hyperparameter yielding highest mean F1-score
        results (dict): dictionary mapping each hyperparameter to its mean F1-score
    """
    folds = k_fold_indices(len(y), k)
    results = {}

    for param in hyperparams:
        f1_scores = []
        thresholds = []
        for i in range(k):
            val_idx = folds[i]
            train_idx = np.hstack([folds[j] for j in range(k) if j != i])
            x_tr, y_tr = tx[train_idx], y[train_idx]
            x_val, y_val = tx[val_idx], y[val_idx]

            w, _ = train_func(y_tr, x_tr, param)
           
            best_t, best_f1 = best_threshold(y_val, x_val, w)
            f1_scores.append(best_f1)
            thresholds.append(best_t)

        results[param] = np.mean(f1_scores)
        print(f"Param={param} | Mean F1={np.mean(f1_scores):.4f}")

    best_param = max(results, key=results.get)
    print(f"\n✅ Best param: {best_param} (F1={results[best_param]:.4f})")
    return best_param, results

def build_poly(x, degree):
    """Polynomial basis functions for input data x, for j=0 up to j=degree.

    Args:
        x (np.array): shape=(N,) input data
        degree (int): polynomial degree
        
    Returns:
        np.array: shape=(N,degree+1) matrix of polynomial features
    """
    
    N = x.shape[0]
    poly = [np.ones((N,1))]
    for d in range(1, degree + 1):
        poly.append(x ** d)
    return np.concatenate(poly, axis=1) 

def cross_validate_degrees(x,y, degrees, k=5, max_iters=1000, gamma =0.5):
    """Perform k-fold CV to select the best polynomial degree using F1 score."""
    folds = k_fold_indices(len(y), k)
    results = {}

    for degree in degrees:
        f1_scores = []
        thresholds = []
        x_poly = build_poly(x, degree)
        for i in range(k):
            val_idx = folds[i]
            train_idx = np.hstack([folds[j] for j in range(k) if j != i])
            x_tr, y_tr = x_poly[train_idx], y[train_idx]
            x_val, y_val = x_poly[val_idx], y[val_idx]

            initial_w = np.zeros(x_tr.shape[1])
            w, _ = logistic_regression(y_tr, x_tr, initial_w, max_iters, gamma)
           
            best_t, best_f1 = best_threshold(y_val, x_val, w)
            y_pred = predict_labels_logistic(x_val, w, best_t)
            f1_scores.append(best_f1)
            thresholds.append(best_t)

        results[degree] = np.mean(f1_scores)
        print(f"Degree={degree} | Mean F1={np.mean(f1_scores):.4f}")

    best_degree = max(results, key=results.get)
    print(f"\n✅ Best degree: {best_degree} (F1={results[best_degree]:.4f})")
    return best_degree, results

def oversample_data(x, y, ratio=1.0, seed=42):
    """
    Randomly oversample the minority class to reach the desired ratio.

    Args:
        x (np.array): shape (N, D), features
        y (np.array): shape (N,), binary labels (0 or 1)
        ratio (float): desired minority/majority ratio after resampling.
                       e.g., ratio=1.0 => fully balanced (equal classes)
                             ratio=0.5 => minority has half as many as majority
        seed (int): random seed for reproducibility

    Returns:
        x_resampled, y_resampled: oversampled dataset
    """
    np.random.seed(seed)

    # Separate classes
    x_min, x_maj = x[y == 1], x[y == 0]
    n_min, n_maj = len(x_min), len(x_maj)

    # Determine how many minority samples we need
    target_min = int(ratio * n_maj)
    if target_min <= n_min:
        return x, y  # already balanced enough

    # Sample with replacement from minority class
    idx = np.random.choice(n_min, target_min - n_min, replace=True)
    x_extra = x_min[idx]
    y_extra = np.ones(len(idx))

    # Combine
    x_res = np.vstack((x, x_extra))
    y_res = np.hstack((y, y_extra))

    # Shuffle
    perm = np.random.permutation(len(y_res))
    return x_res[perm], y_res[perm]