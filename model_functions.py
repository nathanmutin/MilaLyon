import numpy as np
import matplotlib.pyplot as plt

def oversample_minority(x_train, y_train, target_ratio=0.5, random_seed=0):
    """
    Duplicate minority class until minority fraction ≈ target_ratio (0<target_ratio<1).
    Returns x_res, y_res (shuffled).
    """
    rng = np.random.default_rng(random_seed)
    pos_idx = np.where(y_train==1)[0]
    neg_idx = np.where(y_train==0)[0]
    n_pos = len(pos_idx)
    n_neg = len(neg_idx)

    if n_pos == 0:
        return x_train.copy(), y_train.copy()

    # desired minority count
    desired_pos = int(target_ratio * (n_pos + n_neg) / (1 - target_ratio))
    if desired_pos <= n_pos:
        # already enough or target small -> no change
        return x_train.copy(), y_train.copy()

    # sample with replacement from minority
    sample_idx = rng.choice(pos_idx, size=(desired_pos - n_pos), replace=True)
    x_extra = x_train[sample_idx]
    y_extra = y_train[sample_idx]

    x_res = np.vstack([x_train, x_extra])
    y_res = np.concatenate([y_train, y_extra])

    # shuffle
    perm = rng.permutation(len(y_res))
    return x_res[perm], y_res[perm]

def sigmoid_stable(z):
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid(z):
    
    return 1.0 / (1.0 + np.exp(-z))

def train_logreg_weighted(x, y, lr=0.01, epochs=1000, lambda_=0.01):
    N, D = x.shape
    w = np.zeros(D)
    b = 0.0

    n_pos = np.sum(y==1)
    n_neg = np.sum(y==0)
    w_pos = N / (2 * max(n_pos,1))
    w_neg = N / (2 * max(n_neg,1))
    sample_weights = np.where(y==1, w_pos, w_neg)  # shape (N,)

    for _ in range(epochs):
        z = x @ w + b
        p = sigmoid_stable(z)
        err = (p - y) * sample_weights  # shape (N,)
        dw = (x.T @ err) / N + lambda_ * w
        db = np.sum(err) / N
        w -= lr * dw
        b -= lr * db

    return w, b

def f1_score(y_true, y_pred):
    tp = np.sum((y_true==1) & (y_pred==1))
    fp = np.sum((y_true==0) & (y_pred==1))
    fn = np.sum((y_true==1) & (y_pred==0))
    if tp+fp+fn == 0:
        return 0.0
    p = tp / (tp + fp + 1e-12)
    r = tp / (tp + fn + 1e-12)
    return 2*p*r / (p+r+1e-12)

def logistic_regression_stable(y, tx, initial_w, max_iters, gamma):
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
    losses = []
    for _ in range(max_iters):
        # gradient descent step
        pred = sigmoid_stable(tx @ w)  # sigmoid function
        gradient = tx.T @ (pred - y) / len(y)  # gradient of the loss
        w -= gamma * gradient
      
        loss = logistic_negative_log_likelihood(y, tx, w)
        losses.append(loss)

        if iter % 10 == 0 or iter == max_iters - 1:
            print(f"Iter {iter}/{max_iters} - Loss: {loss:.5f}")

    # --- Plot loss over iterations ---
    plt.figure(figsize=(6, 4))
    plt.plot(range(max_iters), losses, label='Training Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Negative Log-Likelihood Loss')
    plt.title('Logistic Regression Convergence')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.show()

    return w, losses[-1]


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Logistic regression using gradient descent with loss plot."""
    w = initial_w.copy()
    losses = []

    for iter in range(max_iters):
        pred = sigmoid(tx @ w)
        gradient = tx.T @ (pred - y) / len(y)
        w -= gamma * gradient

        loss = logistic_negative_log_likelihood(y, tx, w)
        losses.append(loss)

        if iter % 10 == 0 or iter == max_iters - 1:
            print(f"Iter {iter}/{max_iters} - Loss: {loss:.5f}")

    # --- Plot loss over iterations ---
    plt.figure(figsize=(6, 4))
    plt.plot(range(max_iters), losses, label='Training Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Negative Log-Likelihood Loss')
    plt.title('Logistic Regression Convergence')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.show()

    return w, losses[-1]

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

def logistic_regression_predict(x, w, b, threshold=0.5):
    """Predict binary class labels."""
    probs = sigmoid(np.dot(x, w) + b)
    return (probs >= threshold).astype(int), probs

#Logistic regression model training and evaluation
def predict_labels(tx, w, threshold=0.5):
    pred = sigmoid(tx @ w)
    return (pred >= threshold).astype(int)

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

def evaluate_classification(y_true, y_pred, verbose=False):
    """Compute accuracy, F1 score, and confusion matrix."""
    import numpy as np

    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    accuracy = (tp + tn) / len(y_true)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    if verbose:
        print("Confusion Matrix:")
        print(f"TP: {tp}, FP: {fp}")
        print(f"FN: {fn}, TN: {tn}")

    return accuracy, f1, (tp, fp, fn, tn)

def plot_threshold_performance(x_val, y_val, w, b=0.0, step=0.01):
    """
    Evaluate model performance for various thresholds and plot F1 & Accuracy.

    Args:
        x_val (np.array): validation features
        y_val (np.array): true labels (0/1)
        w (np.array): trained weights
        b (float): bias term (default 0.0)
        step (float): step size for threshold sweep (default 0.01)
    """
    thresholds = np.arange(0.0, 1.0 + step, step)
    f1_scores, accuracies = [], []

    for threshold in thresholds:
        y_pred, _ = logistic_regression_predict(x_val, w, b, threshold)
        acc, f1,_ = evaluate_classification(y_val, y_pred)
        f1_scores.append(f1)
        accuracies.append(acc)

    # --- Plot F1 and Accuracy vs Threshold ---
    plt.figure(figsize=(7, 4))
    plt.plot(thresholds, accuracies, label='Accuracy', marker='o', markersize=3)
    plt.plot(thresholds, f1_scores, label='F1 Score', marker='x', markersize=3)
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title('Model Performance vs Threshold')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.show()

    # --- Print best threshold for F1 ---
    best_idx = np.argmax(f1_scores)
    print(f"Best F1 threshold: {thresholds[best_idx]:.2f} | F1 = {f1_scores[best_idx]:.3f}, Accuracy = {accuracies[best_idx]:.3f}")

    return thresholds, accuracies, f1_scores



def grid_search_lambda_thresholds(y_tr, x_tr, x_val, y_val, lambda_values, max_iters=10000, gamma=0.5):
    """
    Perform grid search over regularization parameter lambda
    and plot Accuracy and F1 vs threshold for each lambda.

    Args:
        y_tr (np.array): training labels
        x_tr (np.array): training features
        x_val (np.array): validation features
        y_val (np.array): validation labels
        lambda_values (list or np.array): values of lambda to test
        max_iters (int): number of iterations for training
        gamma (float): learning rate
    """
    plt.figure(figsize=(10, 5))

    thresholds = np.arange(0.0, 1.0, 0.02)
    colors = plt.cm.viridis(np.linspace(0, 1, len(lambda_values)))

    for idx, lambda_ in enumerate(lambda_values):
        # Train model with current lambda
        w_r, loss_r = reg_logistic_regression(y_tr, x_tr, lambda_, np.zeros(x_tr.shape[1]), max_iters, gamma)
        
        f1_scores, accuracies = [], []
        for threshold in thresholds:
            y_pred, _ = logistic_regression_predict(x_val, w_r, 0.0, threshold)
            acc, f1 ,_= evaluate_classification(y_val, y_pred)
            f1_scores.append(f1)
            accuracies.append(acc)

        # Plot F1 and Accuracy for this lambda
        plt.plot(thresholds, f1_scores, label=f'F1 (λ={lambda_})', color=colors[idx], linestyle='-')
        plt.plot(thresholds, accuracies, label=f'Acc (λ={lambda_})', color=colors[idx], linestyle='--')

    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title('F1 and Accuracy vs Threshold for Different λ values')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()

def grid_search_lambda_thresholds_weighted(y_tr, x_tr, x_val, y_val, lambda_values, max_iters=10000, gamma=0.5):
    """
    Perform grid search over regularization parameter lambda
    and plot Accuracy and F1 vs threshold for each lambda.

    Args:
        y_tr (np.array): training labels
        x_tr (np.array): training features
        x_val (np.array): validation features
        y_val (np.array): validation labels
        lambda_values (list or np.array): values of lambda to test
        max_iters (int): number of iterations for training
        gamma (float): learning rate
    """
    plt.figure(figsize=(10, 5))

    thresholds = np.arange(0.0, 1.0, 0.02)
    colors = plt.cm.viridis(np.linspace(0, 1, len(lambda_values)))

    for idx, lambda_ in enumerate(lambda_values):
        # Train model with current lambda
        w_w, loss_w = train_logreg_weighted(x_tr, y_tr, gamma, max_iters, lambda_)
        
        f1_scores, accuracies = [], []
        for threshold in thresholds:
            y_pred, _ = logistic_regression_predict(x_val, w_w, 0.0, threshold)
            acc, f1 ,_= evaluate_classification(y_val, y_pred)
            f1_scores.append(f1)
            accuracies.append(acc)

        # Plot F1 and Accuracy for this lambda
        plt.plot(thresholds, f1_scores, label=f'F1 (λ={lambda_})', color=colors[idx], linestyle='-')
        plt.plot(thresholds, accuracies, label=f'Acc (λ={lambda_})', color=colors[idx], linestyle='--')

    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title('F1 and Accuracy vs Threshold for Different λ values')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()