"""
preprocessing.py

This module contains data preprocessing functions for machine learning pipelines,
including normalization, imputation, feature selection, outlier handling, PCA,
one-hot encoding, and oversampling.
"""

import numpy as np
from src.implementations import *


# --------------------- Normalization --------------------- #

def normalize(x, x_test=None):
    """
    Standardize features to zero mean and unit variance.

    Args:
        x (np.array): training features (N, D)
        x_test (np.array, optional): test features (M, D)

    Returns:
        np.array: normalized training features
        np.array: normalized test features (if provided)
    """
    mean_x = np.nanmean(x, axis=0)
    std_x = np.nanstd(x, axis=0)
    std_x[std_x == 0] = 1.0
    x = (x - mean_x) / std_x
    if x_test is not None:
        x_test = (x_test - mean_x) / std_x
        return x, x_test
    return x


def min_max_normalize(x, x_test=None):
    """
    Rescale features to [0, 1] range.

    Args:
        x (np.array): training features (N, D)
        x_test (np.array, optional): test features (M, D)

    Returns:
        np.array: normalized training features
        np.array: normalized test features (if provided)
    """
    min_x = np.nanmin(x, axis=0)
    max_x = np.nanmax(x, axis=0)
    range_x = max_x - min_x
    range_x[range_x == 0] = 1.0
    x = (x - min_x) / range_x
    if x_test is not None:
        x_test = (x_test - min_x) / range_x
        return x, x_test
    return x


# --------------------- Missing values --------------------- #

def mean_imputation(x_train, x_test):
    """
    Replace NaNs with feature means from training set.
    Modifies x_train and x_test in place.
    """
    mean_x = np.nanmean(x_train, axis=0)
    x_train[np.isnan(x_train)] = np.take(mean_x, np.where(np.isnan(x_train))[1])
    x_test[np.isnan(x_test)] = np.take(mean_x, np.where(np.isnan(x_test))[1])


def replace_default_with_nan(x_train, x_test, default_values):
    """Replace dataset default values with NaN."""
    for i, defaults in enumerate(default_values):
        for value in defaults:
            x_train[x_train[:, i] == value, i] = np.nan
            x_test[x_test[:, i] == value, i] = np.nan


def replace_by_zero(x_train, x_test, zero_values):
    """Replace specified values (including NaN) with zero."""
    for i, val in enumerate(zero_values):
        if val is None or np.isnan(val):
            x_train[np.isnan(x_train[:, i]), i] = 0
            x_test[np.isnan(x_test[:, i]), i] = 0
        else:
            x_train[x_train[:, i] == val, i] = 0
            x_test[x_test[:, i] == val, i] = 0


def replace_nan(x_train, x_test, continuous_flag):
    """
    Replace remaining NaNs: mean for continuous, mode for categorical.
    Returns modified x_train, x_test.
    """
    for i in range(x_train.shape[1]):
        non_nan = x_train[~np.isnan(x_train[:, i]), i]
        if continuous_flag[i] == 0:
            mode = np.bincount(non_nan.astype(int)).argmax()
            x_train[np.isnan(x_train[:, i]), i] = mode
            x_test[np.isnan(x_test[:, i]), i] = mode
        else:
            mean = np.mean(non_nan)
            x_train[np.isnan(x_train[:, i]), i] = mean
            x_test[np.isnan(x_test[:, i]), i] = mean
    return x_train, x_test


# --------------------- Feature engineering --------------------- #

def convert_to_times_per_week(x, feature_flags):
    """Convert specified features to 'times per week'."""
    for i, flag in enumerate(feature_flags):
        if flag != 1:
            continue
        col = x[:, i].astype(float)
        col[col == 888] = 0  # Never
        col[col == 777] = np.nan  # Don't know
        col[col == 999] = np.nan  # Refused
        col[(col >= 201) & (col <= 299)] /= 4.33  # month → week
        x[:, i] = col
    return x


def one_hot_encode(data_dict):
    """One-hot encode categorical features in the data dictionary."""
    one_hot_features = data_dict["feature_names"][data_dict["one_hot"] == 1]
    binary_features = data_dict["feature_names"][data_dict["binary"] == 1]

    for feature in one_hot_features:
        if feature in binary_features:
            continue
        idx = np.where(data_dict["feature_names"] == feature)[0][0]
        unique_vals = np.unique(data_dict["x_train"][:, idx])
        for val in unique_vals:
            new_train_col = (data_dict["x_train"][:, idx] == val).astype(int)
            new_test_col = (data_dict["x_test"][:, idx] == val).astype(int)
            data_dict["x_train"] = np.column_stack((data_dict["x_train"], new_train_col))
            data_dict["x_test"] = np.column_stack((data_dict["x_test"], new_test_col))
            # Append meta-data
            for key in ["feature_names", "useless", "health_related", "better_elsewhere",
                        "bad_format_no_better", "binary", "one_hot", "zero_values",
                        "default_values", "ordinal", "continuous"]:
                arr = data_dict[key]
                data_dict[key] = np.append(arr, arr[idx] if key != "binary" else 1)

        # Drop original feature
        for key in ["x_train", "x_test", "feature_names", "useless", "health_related",
                    "better_elsewhere", "bad_format_no_better", "binary", "one_hot",
                    "zero_values", "default_values", "ordinal", "continuous"]:
            data_dict[key] = np.delete(data_dict[key], idx, axis=1 if key.startswith("x_") else 0)


def drop_features_from_dictionary(data_dict, feature_names_to_drop):
    """Drop specified features from data dictionary."""
    for fname in feature_names_to_drop:
        if fname in data_dict["feature_names"]:
            idx = np.where(data_dict["feature_names"] == fname)[0][0]
            for key in ["x_train", "x_test", "feature_names", "useless", "health_related",
                        "better_elsewhere", "bad_format_no_better", "binary", "one_hot",
                        "zero_values", "default_values", "ordinal", "continuous"]:
                data_dict[key] = np.delete(data_dict[key], idx, axis=1 if key.startswith("x_") else 0)


# --------------------- Data splitting --------------------- #

def split_train_val(x_train, y_train, val_size=0.1, random_seed=42):
    """Random train/validation split."""
    np.random.seed(random_seed)
    N = x_train.shape[0]
    indices = np.random.permutation(N)
    val_idx = indices[:int(N * val_size)]
    train_idx = indices[int(N * val_size):]
    return x_train[train_idx], y_train[train_idx], x_train[val_idx], y_train[val_idx]


# --------------------- Outliers & PCA --------------------- #

def clip_outliers(x_train, x_test=None, n_std=3):
    """Clip feature values to mean ± n_std * std."""
    mean = np.mean(x_train, axis=0)
    std = np.std(x_train, axis=0)
    clip_min = mean - n_std * std
    clip_max = mean + n_std * std

    n_clipped = np.sum((x_train < clip_min) | (x_train > clip_max))
    x_train_clipped = np.clip(x_train, clip_min, clip_max)

    if x_test is not None:
        x_test_clipped = np.clip(x_test, clip_min, clip_max)
        return x_train_clipped, x_test_clipped, n_clipped
    return x_train_clipped, n_clipped


def pca_reduce(x_train, x_test=None, variance_threshold=0.95):
    """PCA dimensionality reduction keeping given variance."""
    cov = np.cov(x_train, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[idx], eigvecs[:, idx]

    explained_variance = eigvals / np.sum(eigvals)
    k = np.searchsorted(np.cumsum(explained_variance), variance_threshold) + 1

    x_train_pca = x_train @ eigvecs[:, :k]
    if x_test is not None:
        x_test_pca = x_test @ eigvecs[:, :k]
        return x_train_pca, x_test_pca, eigvecs[:, :k], explained_variance[:k]
    return x_train_pca, eigvecs[:, :k], explained_variance[:k]


# --------------------- Oversampling --------------------- #

def oversample_data(x, y, ratio=1.0, seed=42):
    """Randomly oversample minority class to reach specified ratio."""
    np.random.seed(seed)
    x_min, x_maj = x[y == 1], x[y == 0]
    n_min, n_maj = len(x_min), len(x_maj)
    target_min = int(ratio * n_maj)
    if target_min <= n_min:
        return x, y
    idx = np.random.choice(n_min, target_min - n_min, replace=True)
    x_extra = x_min[idx]
    y_extra = np.ones(len(idx))
    x_res = np.vstack((x, x_extra))
    y_res = np.hstack((y, y_extra))
    perm = np.random.permutation(len(y_res))
    return x_res[perm], y_res[perm]
