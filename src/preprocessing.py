"""
preprocessing.py

This module contains utility functions for preprocessing datasets in machine learning pipelines.
It includes normalization, imputation, feature engineering, feature selection, dimensionality reduction,
outlier handling, one-hot encoding, data splitting, and oversampling.

Author: Your Name
Date: YYYY-MM-DD
"""

import numpy as np
from src.implementations import *


# ---------------------- Normalization ----------------------
def normalize(x, x_test=None):
    """Normalize data to zero mean and unit variance."""
    mean_x = np.nanmean(x, axis=0)
    std_x = np.nanstd(x, axis=0)
    std_x[std_x == 0] = 1.0  # avoid division by zero
    x = (x - mean_x) / std_x
    if x_test is not None:
        x_test = (x_test - mean_x) / std_x
        return x, x_test
    return x


def min_max_normalize(x, x_test=None):
    """Min-max normalize data to range [0, 1]."""
    min_x = np.nanmin(x, axis=0)
    max_x = np.nanmax(x, axis=0)
    range_x = max_x - min_x
    range_x[range_x == 0] = 1.0
    x = (x - min_x) / range_x
    if x_test is not None:
        x_test = (x_test - min_x) / range_x
        return x, x_test
    return x


# ---------------------- Imputation ----------------------
def mean_imputation(x_train, x_test):
    """Impute missing values with feature-wise mean."""
    mean_x = np.nanmean(x_train, axis=0)
    inds_train = np.where(np.isnan(x_train))
    x_train[inds_train] = mean_x[inds_train[1]]
    inds_test = np.where(np.isnan(x_test))
    x_test[inds_test] = mean_x[inds_test[1]]


def replace_default_with_nan(x_train, x_test, default_values):
    """Replace default values with NaN."""
    for i, defaults in enumerate(default_values):
        for val in defaults:
            x_train[x_train[:, i] == val, i] = np.nan
            x_test[x_test[:, i] == val, i] = np.nan


def replace_by_zero(x_train, x_test, zero_values):
    """Replace specified values with zero."""
    for i in range(len(zero_values)):
        x_train[x_train[:, i] == zero_values[i], i] = 0
        x_test[x_test[:, i] == zero_values[i], i] = 0
        if zero_values[i] is not None and np.isnan(zero_values[i]):
            x_train[np.isnan(x_train[:, i]), i] = 0
            x_test[np.isnan(x_test[:, i]), i] = 0


def replace_nan(x_train, x_test, continuous_flag):
    """Replace NaN values with mean for continuous or mode for categorical features."""
    for i in range(x_train.shape[1]):
        non_nan_values = x_train[~np.isnan(x_train[:, i]), i]
        if continuous_flag[i] == 0:
            mode = np.bincount(non_nan_values.astype(int)).argmax()
            x_train[np.isnan(x_train[:, i]), i] = mode
            x_test[np.isnan(x_test[:, i]), i] = mode
        else:
            mean = np.mean(non_nan_values)
            x_train[np.isnan(x_train[:, i]), i] = mean
            x_test[np.isnan(x_test[:, i]), i] = mean
    return x_train, x_test


# ---------------------- Feature Engineering ----------------------
def convert_to_times_per_week(x, feature_flags):
    """Convert flagged features to 'times per week'."""
    for i, flag in enumerate(feature_flags):
        if flag == 1:
            col = x[:, i].astype(float)
            col[col == 888] = 0  # Never
            col[col == 777] = np.nan  # Don't know
            col[col == 999] = np.nan  # Refused
            col[(col >= 201) & (col <= 299)] /= 4.33  # Month → week
            x[:, i] = col
    return x


def split_train_val(x_train, y_train, val_size=0.1, random_seed=43):
    """Split training data into train and validation sets."""
    np.random.seed(random_seed)
    N = x_train.shape[0]
    indices = np.random.permutation(N)
    val_size_int = int(N * val_size)
    val_indices = indices[:val_size_int]
    train_indices = indices[val_size_int:]
    x_val = x_train[val_indices]
    y_val = y_train[val_indices]
    x_train_new = x_train[train_indices]
    y_train_new = y_train[train_indices]
    return x_train_new, y_train_new, x_val, y_val


# ---------------------- Feature Selection ----------------------
def drop_too_many_missing(x_train, x_test, train_columns, threshold=0.2):
    """Drop features with too many missing values."""
    nan_ratio = np.isnan(x_train).sum(axis=0) / x_train.shape[0]
    cols_to_keep = nan_ratio <= threshold
    dropped_cols = np.where(~cols_to_keep)[0]
    dropped_names = [train_columns[i] for i in dropped_cols]
    print(f"Dropped {len(dropped_cols)} features ({np.mean(~cols_to_keep)*100:.1f}%)")
    if len(dropped_names) > 0:
        print("Dropped feature names:", dropped_names)
    return x_train[:, cols_to_keep], x_test[:, cols_to_keep], cols_to_keep


def identify_too_many_missing(x_train, feature_names, threshold=0.2):
    """Return feature names with too many missing values."""
    nan_ratio = np.isnan(x_train).sum(axis=0) / x_train.shape[0]
    return [feature_names[i] for i in np.where(nan_ratio > threshold)[0]]


def identify_low_correlation(x_train, y_train, feature_names, threshold=0.1):
    """Return features with low correlation to target."""
    correlations = np.array([np.corrcoef(x_train[:, i], y_train)[0, 1] for i in range(x_train.shape[1])])
    low_corr_mask = np.abs(correlations) < threshold
    return [feature_names[i] for i in np.where(low_corr_mask)[0]], correlations


def drop_highly_correlated(x_train, feature_names, threshold=0.9):
    """Identify highly correlated features for dropping."""
    corr_matrix = np.corrcoef(x_train, rowvar=False)
    nan_counts = np.isnan(x_train).sum(axis=0)
    drop_cols = set()
    D = corr_matrix.shape[0]
    for i in range(D):
        for j in range(i + 1, D):
            if abs(corr_matrix[i, j]) > threshold:
                drop_cols.add(i if nan_counts[i] > nan_counts[j] else j)
    return [feature_names[i] for i in drop_cols], corr_matrix


def drop_features_from_dictionnary(data_dict, feature_names_to_drop):
    """Drop features from a data dictionary."""
    for feature_name in feature_names_to_drop:
        if feature_name in data_dict["feature_names"]:
            idx = np.where(data_dict["feature_names"] == feature_name)[0][0]
            for key in data_dict.keys():
                if isinstance(data_dict[key], np.ndarray) and data_dict[key].shape[1] == len(data_dict["feature_names"]):
                    data_dict[key] = np.delete(data_dict[key], idx, axis=1)
            for key in ["feature_names", "useless", "health_related", "better_elsewhere",
                        "bad_format_no_better", "binary", "one_hot", "zero_values",
                        "default_values", "ordinal", "continuous"]:
                if key in data_dict:
                    data_dict[key] = np.delete(data_dict[key], idx)
        else:
            print(f"Feature {feature_name} not found in feature names.")


# ---------------------- One-Hot Encoding ----------------------
def one_hot_encode(data_dict):
    """One-hot encode categorical features in a data dictionary."""
    one_hot_features = data_dict["feature_names"][data_dict["one_hot"] == 1]
    binary_features = data_dict["feature_names"][data_dict["binary"] == 1]
    for feature in one_hot_features:
        if feature not in binary_features:
            idx = np.where(data_dict["feature_names"] == feature)[0][0]
            unique_values = np.unique(data_dict["x_train"][:, idx])
            for value in unique_values:
                one_hot_encoded = (data_dict["x_train"][:, idx] == value).astype(int)
                data_dict["x_train"] = np.column_stack((data_dict["x_train"], one_hot_encoded))
                one_hot_encoded_test = (data_dict["x_test"][:, idx] == value).astype(int)
                data_dict["x_test"] = np.column_stack((data_dict["x_test"], one_hot_encoded_test))
                data_dict["feature_names"] = np.append(data_dict["feature_names"], f"{feature}_{value}")
                for key in ["useless", "health_related", "better_elsewhere",
                            "bad_format_no_better", "binary", "one_hot", "zero_values",
                            "default_values", "ordinal", "continuous"]:
                    data_dict[key] = np.append(data_dict[key], data_dict[key][idx])
            # Drop original feature
            for key in ["x_train", "x_test", "feature_names", "useless", "health_related",
                        "better_elsewhere", "bad_format_no_better", "binary", "one_hot",
                        "zero_values", "default_values", "ordinal", "continuous"]:
                data_dict[key] = np.delete(data_dict[key], idx, axis=0 if key not in ["x_train", "x_test"] else 1)


# ---------------------- Outlier Handling ----------------------
def clip_outliers(x_train, x_test=None, n_std=3):
    """Clip outliers to mean ± n_std * std."""
    mean = np.mean(x_train, axis=0)
    std = np.std(x_train, axis=0)
    clip_min, clip_max = mean - n_std * std, mean + n_std * std
    n_clipped_train = np.sum((x_train < clip_min) | (x_train > clip_max))
    x_train_clipped = np.clip(x_train, clip_min, clip_max)
    n_clipped_test = None
    if x_test is not None:
        n_clipped_test = np.sum((x_test < clip_min) | (x_test > clip_max))
        x_test = np.clip(x_test, clip_min, clip_max)
    return (x_train_clipped, x_test, n_clipped_train, n_clipped_test) if x_test is not None else (x_train_clipped, n_clipped_train)


# ---------------------- Dimensionality Reduction ----------------------
def pca_reduce(x_train, x_test=None, variance_threshold=0.95):
    """Perform PCA to reduce dimensionality while preserving variance."""
    cov = np.cov(x_train, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[idx], eigvecs[:, idx]
    explained_variance = eigvals / np.sum(eigvals)
    cumulative_variance = np.cumsum(explained_variance)
    k = np.searchsorted(cumulative_variance, variance_threshold) + 1
    x_train_pca = np.dot(x_train, eigvecs[:, :k])
    if x_test is not None:
        x_test_pca = np.dot(x_test, eigvecs[:, :k])
        return x_train_pca, x_test_pca, eigvecs[:, :k], explained_variance[:k]
    return x_train_pca, eigvecs[:, :k], explained_variance[:k]


# ---------------------- Oversampling ----------------------
def oversample_data(x, y, ratio=1.0, seed=42):
    """Randomly oversample minority class to desired ratio."""
    np.random.seed(seed)
    x_min, x_maj = x[y == 1], x[y == 0]
    n_min, n_maj = len(x_min), len(x_maj)
    target_min = int(ratio * n_maj)
    if target_min <= n_min:
        return x, y
    idx = np.random.choice(n_min, target_min - n_min, replace=True)
    x_extra, y_extra = x_min[idx], np.ones(len(idx))
    x_res = np.vstack((x, x_extra))
    y_res = np.hstack((y, y_extra))
    perm = np.random.permutation(len(y_res))
    return x_res[perm], y_res[perm]


# ---------------------- Utilities ----------------------
def print_shapes(data):
    """Print shapes of all arrays in a data dictionary."""
    for key, value in data.items():
        print(f"{key}: {type(value)} with shape {value.shape if isinstance(value, np.ndarray) else 'N/A'}")


# ---------------------- Preprocessing Pipeline ----------------------

def preprocess_data(
    data,
    nan_drop_threshold=0.2,
    correlation_threshold=0.02,
    n_std=3,
    only_health_related=True,
    split_val=False,
    val_size=0.1,
    one_hot=True,
    drop_correlated=True,
):
    """
    Preprocess data dictionary with standard steps:
        - Convert and clean bad format features
        - Replace zeros and default values
        - Drop features with too many missing values
        - Impute remaining NaNs
        - Keep only health-related features if specified
        - One-hot encode categorical features
        - Drop features with low correlation to target
        - Drop highly correlated features
        - Clip outliers
        - Min-max normalize features
        - Optionally split training data into train/validation

    Args:
        data (dict): Data dictionary with keys like x_train, x_test, y_train, feature_names, etc.
        nan_drop_threshold (float): Fraction of missing values to drop a feature.
        correlation_threshold (float): Minimum correlation to keep a feature.
        n_std (float): Number of std deviations for outlier clipping.
        only_health_related (bool): Keep only health-related features.
        split_val (bool): Whether to split training data into validation.
        val_size (float): Fraction of training data for validation.
        one_hot (bool): Whether to one-hot encode categorical features.
        drop_correlated (bool): Whether to drop highly correlated features.
    """
    # ---------------------- Step 1: Clean & Convert ----------------------
    convert_to_times_per_week(data["x_train"], data["bad_format_no_better"])
    convert_to_times_per_week(data["x_test"], data["bad_format_no_better"])
    replace_by_zero(data["x_train"], data["x_test"], data["zero_values"])
    replace_default_with_nan(data["x_train"], data["x_test"], data["default_values"])

    print("Initial data shapes:")
    print_shapes(data)

    # ---------------------- Step 2: Drop features with too many missing values ----------------------
    nan_features = identify_too_many_missing(data["x_train"], data["feature_names"], nan_drop_threshold)
    drop_features_from_dictionnary(data, nan_features)
    print(f"{len(nan_features)} features with too many missing values dropped.")

    # ---------------------- Step 3: Impute remaining NaNs ----------------------
    replace_nan(data["x_train"], data["x_test"], data["continuous"])

    # ---------------------- Step 4: Keep only health-related features ----------------------
    if only_health_related:
        non_health_features = data["feature_names"][~data["health_related"]].tolist()
        drop_features_from_dictionnary(data, non_health_features)
        print(f"{len(non_health_features)} non health-related features dropped.")

    # ---------------------- Step 5: One-hot encode categorical features ----------------------
    if one_hot:
        n_features_before = data["x_train"].shape[1]
        one_hot_encode(data)
        n_features_after = data["x_train"].shape[1]
        print(f"One-hot encoding completed. Features increased from {n_features_before} to {n_features_after}.")

    # ---------------------- Step 6: Drop features with low correlation to target ----------------------
    low_corr_features, _ = identify_low_correlation(data["x_train"], data["y_train"], data["feature_names"], threshold=correlation_threshold)
    drop_features_from_dictionnary(data, low_corr_features)
    print(f"{len(low_corr_features)} features with low correlation to target dropped.")

    # ---------------------- Step 7: Drop highly correlated features ----------------------
    if drop_correlated:
        high_corr_features, _ = drop_highly_correlated(data["x_train"], data["feature_names"])
        drop_features_from_dictionnary(data, high_corr_features)
        print(f"{len(high_corr_features)} highly correlated features dropped.")

    # ---------------------- Step 8: Clip outliers ----------------------
    clip_outliers(data["x_train"], data["x_test"], n_std=n_std)

    # ---------------------- Step 9: Normalize features ----------------------
    data["x_train"], data["x_test"] = min_max_normalize(data["x_train"], data["x_test"])
    data["y_train"] = (data["y_train"] == 1).astype(int)

    # ---------------------- Step 10: Optional train/validation split ----------------------
    if split_val:
        data["x_train"], data["y_train"], data["x_val"], data["y_val"] = split_train_val(data["x_train"], data["y_train"], val_size=val_size)

    print("Preprocessing completed. Final data shapes:")
    print_shapes(data)
