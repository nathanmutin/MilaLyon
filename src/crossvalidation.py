"""
crossvalidation.py

This module provides functions for k-fold cross-validation to select optimal hyperparameters
or polynomial degrees for machine learning models. It supports generic training functions 
and uses F1-score as the evaluation metric. 

Functions:
- k_fold_indices: Generate randomized indices for k-fold cross-validation.
- cross_validate_hyperparameter: Select the best hyperparameter using k-fold CV and F1-score.
- cross_validate_degrees: Select the best polynomial degree using k-fold CV and F1-score.
"""

import numpy as np
import matplotlib.pyplot as plt
from src.implementations import *
from src.preprocessing import *
from src.model_evaluation import *


def k_fold_indices(N, k, seed=42):
    """
    Generate indices for k-fold cross-validation.

    Args:
        N (int): Number of samples in the dataset.
        k (int): Number of folds.
        seed (int): Random seed for reproducibility.

    Returns:
        list of np.array: List containing k arrays of indices for each fold.
    """
    np.random.seed(seed)
    indices = np.random.permutation(N)
    fold_sizes = np.full(k, N // k, dtype=int)
    fold_sizes[: N % k] += 1
    current = 0
    folds = []
    for fold_size in fold_sizes:
        start, stop = current, current + fold_size
        folds.append(indices[start:stop])
        current = stop
    return folds


def cross_validate_hyperparameter(y, tx, train_func, hyperparams, k=5):
    """
    Generic k-fold cross-validation to select the best hyperparameter using F1 score.

    Args:
        y (np.array): Target vector.
        tx (np.array): Feature matrix.
        train_func (callable): Training function accepting (y_train, x_train, hyperparam) 
                               and returning (weights, loss).
        hyperparams (np.ndarray): Array of hyperparameters to evaluate.
        k (int): Number of folds.

    Returns:
        best_hyperparam: Hyperparameter yielding the highest mean F1-score.
        results (dict): Mapping of each hyperparameter to its mean F1-score.
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


def cross_validate_degrees(x, y, degrees, to_expand, k=5, max_iters=1000, gamma=0.5):
    """
    Perform k-fold cross-validation to select the best polynomial degree using F1 score.

    Args:
        x (np.array): Feature matrix.
        y (np.array): Target vector.
        degrees (list or np.array): Polynomial degrees to evaluate.
        to_expand (list): List of feature indices to apply polynomial expansion to.
        k (int): Number of folds.
        max_iters (int): Maximum iterations for logistic regression.
        gamma (float): Learning rate for logistic regression.

    Returns:
        best_degree: Polynomial degree yielding the highest mean F1-score.
        results (dict): Mapping of each degree to its mean F1-score.
    """
    folds = k_fold_indices(len(y), k)
    results = {}

    for degree in degrees:
        f1_scores = []
        thresholds = []
        x_poly = build_poly(x, degree, to_expand=to_expand)
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
