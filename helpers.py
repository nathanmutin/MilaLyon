import csv
import numpy as np
import os


def load_csv_data(data_path, sub_sample=False):
    """
    This function loads the data and returns the respectinve numpy arrays.
    Remember to put the 3 files in the same folder and to not change the names of the files.

    Args:
        data_path (str): datafolder path
        sub_sample (bool, optional): If True the data will be subsempled. Default to False.

    Returns:
        x_train (np.array): training data
        x_test (np.array): test data
        y_train (np.array): labels for training data in format (-1,1)
        train_ids (np.array): ids of training data
        test_ids (np.array): ids of test data
    """
    max_rows = None
    if sub_sample:
        max_rows = 500
    
    y_train = np.genfromtxt(
        os.path.join(data_path, "y_train.csv"),
        delimiter=",",
        skip_header=1,
        dtype=int,
        usecols=1,
        max_rows=max_rows,
    )
    x_train = np.genfromtxt(
        os.path.join(data_path, "x_train.csv"),
        delimiter=",",
        skip_header=1,
        max_rows=max_rows,
    )
    x_test = np.genfromtxt(
        os.path.join(data_path, "x_test.csv"),
        delimiter=",",
        skip_header=1,
        max_rows=max_rows,
    )

    train_ids = x_train[:, 0].astype(dtype=int)
    test_ids = x_test[:, 0].astype(dtype=int)
    x_train = x_train[:, 1:]
    x_test = x_test[:, 1:]
    
      # --- Get column names from headers ---
    with open(os.path.join(data_path, "x_train.csv"), "r") as f:
        train_columns = f.readline().strip().split(",")[1:]  # skip "Id"

    with open(os.path.join(data_path, "x_test.csv"), "r") as f:
        test_columns = f.readline().strip().split(",")[1:]  # skip "Id"

    return x_train, x_test, y_train, train_ids, test_ids, train_columns, test_columns

def create_csv_submission(ids, y_pred, name):
    """
    This function creates a csv file named 'name' in the format required for a submission in Kaggle or AIcrowd.
    The file will contain two columns the first with 'ids' and the second with 'y_pred'.
    y_pred must be a list or np.array of 1 and -1 otherwise the function will raise a ValueError.

    Args:
        ids (list,np.array): indices
        y_pred (list,np.array): predictions on data correspondent to indices
        name (str): name of the file to be created
    """
    # Check that y_pred only contains -1 and 1
    if not all(i in [-1, 1] for i in y_pred):
        raise ValueError("y_pred can only contain values -1, 1")

    with open(name, "w", newline="") as csvfile:
        fieldnames = ["Id", "Prediction"]
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({"Id": int(r1), "Prediction": int(r2)})


def replace_missing(X, default_missing, exceptions):
    X_clean = X.astype(float)  # allow np.nan
    for feature_group, codes in exceptions.items():
        for col in feature_group:
            mask = np.isin(X_clean[:, col], codes)
            X_clean[mask, col] = np.nan

    all_excepted_cols = [col for group in exceptions.keys() for col in group]
    for col in range(X_clean.shape[1]):
        if col not in all_excepted_cols:
            mask = np.isin(X_clean[:, col], default_missing)
            X_clean[mask, col] = np.nan

    return X_clean

def drop_too_many_missing(x_train, x_test, train_columns, threshold=0.3):
    """
    Drops features (columns) with more than a given percentage of missing values (NaNs).

    Args:
        x_train (np.array): shape = (N, D) training feature matrix
        x_test (np.array): shape = (M, D) test feature matrix
        train_columns (list or np.array): feature names corresponding to columns
        threshold (float): fraction of allowed missing values before dropping (default 0.3)

    Returns:
        tuple:
            x_train_reduced (np.array): training data with dropped columns
            x_test_reduced (np.array): test data with dropped columns
            cols_to_keep (np.array): boolean mask of kept columns
    """
    nan_ratio = np.isnan(x_train).sum(axis=0) / x_train.shape[0]
    cols_to_keep = nan_ratio <= threshold

    # Identify dropped features
    dropped_cols = np.where(~cols_to_keep)[0]
    dropped_names = [train_columns[i] for i in dropped_cols]

    print(f"Dropped {len(dropped_cols)} features ({np.mean(~cols_to_keep)*100:.1f}%)")
    if len(dropped_names) > 0:
        print("Dropped feature names:", dropped_names)

    # Reduce both train and test sets
    x_train_reduced = x_train[:, cols_to_keep]
    x_test_reduced = x_test[:, cols_to_keep]

    return x_train_reduced, x_test_reduced, cols_to_keep

import numpy as np

def drop_highly_correlated(x_train, x_test, feature_names, threshold=0.9):
    """
    Drops one feature from each highly correlated pair based on the number of NaNs.
    The feature with more NaNs is dropped.

    Args:
        x_train (np.array): shape = (N, D) training feature matrix
        x_test (np.array): shape = (M, D) test feature matrix
        feature_names (list or np.array): feature names corresponding to columns
        threshold (float): correlation threshold to consider a pair highly correlated

    Returns:
        tuple:
            x_train_reduced (np.array): training data with correlated features removed
            x_test_reduced (np.array): test data with correlated features removed
            kept_cols (np.array): boolean mask of kept columns
            dropped_names (list): names of the dropped features
    """
    # Compute correlation matrix
    corr = np.corrcoef(x_train, rowvar=False)

    # Count NaNs per feature
    nan_counts = np.isnan(x_train).sum(axis=0)

    # Track columns to drop
    drop_cols = set()
    D = corr.shape[0]

    for i in range(D):
        for j in range(i + 1, D):
            if abs(corr[i, j]) > threshold:
                # Drop the feature with more NaNs; if equal, drop j
                if nan_counts[i] > nan_counts[j]:
                    drop_cols.add(i)
                else:
                    drop_cols.add(j)

    # Build mask of columns to keep
    kept_cols = np.array([i not in drop_cols for i in range(D)])

    # Reduce train and test sets
    x_train_reduced = x_train[:, kept_cols]
    x_test_reduced = x_test[:, kept_cols]

    # Get dropped feature names
    dropped_names = [feature_names[i] for i in range(D) if i in drop_cols]

    print(f"Dropped {len(dropped_names)} highly correlated features:")
    print(dropped_names)

    return x_train_reduced, x_test_reduced, kept_cols, dropped_names

