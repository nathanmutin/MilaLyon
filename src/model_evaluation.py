import numpy as np
import matplotlib.pyplot as plt
from src.implementations import *
from src.preprocessing import *


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
    f1_score = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
    }


def best_threshold(y_true, tx, w, thresholds=np.linspace(0.0, 1.0, 100)):
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
        y_pred = predict_labels_logistic(tx, w, t)
        scores = compute_scores(y_true, y_pred)
        f1 = scores["f1_score"]
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t, best_f1


def plot_training_validation_performance(
    x_train, y_train, x_val, y_val, weights, losses
):
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

        f1_train.append(scores_train["f1_score"])
        acc_train.append(scores_train["accuracy"])
        f1_val.append(scores_val["f1_score"])
        acc_val.append(scores_val["accuracy"])

    # ---- Best threshold ----
    best_idx = np.argmax(f1_val)
    best_threshold = thresholds[best_idx]
    print(
        f"✅ Best threshold: {best_threshold:.2f} | F1_val = {f1_val[best_idx]:.3f} | Acc_val = {acc_val[best_idx]:.3f}"
    )

    # ---- Plot F1 & Accuracy vs Threshold ----
    plt.figure(figsize=(8, 5))
    plt.plot(
        thresholds,
        acc_train,
        label="Train Accuracy",
        marker="o",
        markersize=3,
        color="blue",
    )
    plt.plot(
        thresholds,
        acc_val,
        label="Val Accuracy",
        marker="x",
        markersize=3,
        color="green",
    )
    plt.plot(
        thresholds, f1_train, label="Train F1", marker="o", markersize=3, color="orange"
    )
    plt.plot(thresholds, f1_val, label="Val F1", marker="x", markersize=3, color="red")
    plt.axvline(
        best_threshold,
        color="red",
        linestyle=":",
        label=f"Best Threshold ({best_threshold:.2f})",
    )
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title("F1 & Accuracy vs Threshold (Train & Validation)")
    plt.legend()
    plt.grid(alpha=0.5)
    plt.show()
    return best_threshold
