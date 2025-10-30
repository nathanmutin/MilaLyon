"""
helpers.py

This module contains utility functions for data loading and submission generation.

Functions:
- load_csv_data: Load training and test data along with metadata and feature descriptions.
- create_csv_submission: Generate a CSV file in the required format for submission to Kaggle or AIcrowd.
"""

import csv
import numpy as np
import os


def load_csv_data(data_path, max_rows=None, dictionnary=False):
    """
    Load dataset and extract feature metadata.

    Note:
        Ensure that the three required files are in the same folder:
        - x_train.csv
        - y_train.csv
        - x_test.csv
        - features_description.csv

    Args:
        data_path (str): Path to the folder containing the data files.
        max_rows (int, optional): Limit the number of rows loaded from each file.
        dictionnary (bool, optional): If True, return data as a dictionary.

    Returns:
        tuple or dict: Depending on `dictionnary`, returns either a tuple of:
            x_train, x_test, y_train, train_ids, test_ids, feature_names,
            zero_values, default_values, useless, health_related,
            better_elsewhere, bad_format_no_better, binary, one_hot,
            ordinal, continuous
        or a dictionary with the same keys.
    """
        # Load CSV files
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

    # Separate IDs from features
    train_ids = x_train[:, 0].astype(int)
    test_ids = x_test[:, 0].astype(int)
    x_train = x_train[:, 1:]
    x_test = x_test[:, 1:]

    # Load feature names from CSV header
    with open(os.path.join(data_path, "x_train.csv"), "r") as f:
        feature_names = np.array(f.readline().strip().split(",")[1:])

    # Initialize arrays and dictionaries for feature metadata
    zero_values = np.zeros(len(feature_names), dtype=object)
    default_values = np.zeros(len(feature_names), dtype=object)
    useless = np.zeros(len(feature_names), dtype=bool)
    health_related = np.zeros(len(feature_names), dtype=bool)
    better_elsewhere = np.zeros(len(feature_names), dtype=bool)
    bad_format_no_better = np.zeros(len(feature_names), dtype=bool)
    binary = np.zeros(len(feature_names), dtype=bool)
    one_hot = np.zeros(len(feature_names), dtype=bool)
    ordinal = np.zeros(len(feature_names), dtype=bool)
    continuous = np.zeros(len(feature_names), dtype=bool)

    # Read features_description.csv to extract metadata
    with open(os.path.join(data_path, "features_description.csv"), "r") as f:
        reader = csv.reader(f, delimiter=",")
        next(reader)  # Skip header

        for i, row in enumerate(reader):
            # Validate feature name matches header
            if row[0] != feature_names[i]:
                raise ValueError(
                    f"Feature n°{i} mismatch: {row[0]} != {feature_names[i]}"
                )

            # Column 2: zero value
            try:
                zero_values[i] = int(row[1])
            except ValueError:
                zero_values[i] = None

            # Column 3 + 14: feature is useless
            try:
                if int(row[2]) == 1 or int(row[13] == 1):
                    useless[i] = True
            except ValueError:
                useless[i] = False

            # Column 4: health-related
            try:
                if int(row[3]) == 1:
                    health_related[i] = True
            except ValueError:
                health_related[i] = False

            # Column 5: better format exists elsewhere
            try:
                if int(row[4]) == 1:
                    better_elsewhere[i] = True
            except ValueError:
                better_elsewhere[i] = False

            # Column 6: bad format with no better alternative
            try:
                if int(row[5]) == 1:
                    bad_format_no_better[i] = True
            except ValueError:
                bad_format_no_better[i] = False

            # Column 7: binary
            try:
                if int(row[6]) == 1:
                    binary[i] = True
            except ValueError:
                binary[i] = False

            # Column 8: one-hot encoded
            try:
                if int(row[7]) == 1:
                    one_hot[i] = True
            except ValueError:
                one_hot[i] = False

            # Columns 9–11: default values for no response
            default_values[i] = []
            for val in row[8:11]:
                try:
                    default_values[i].append(float(val))
                except ValueError:
                    pass

            # Column 12: ordinal
            try:
                if int(row[11]) == 1:
                    ordinal[i] = True
            except ValueError:
                ordinal[i] = False

            # Column 13: continuous
            try:
                if int(row[12]) == 1:
                    continuous[i] = True
            except ValueError:
                continuous[i] = False

    # Return as dictionary or tuple
    if dictionnary:
        return {
            "x_train": x_train,
            "x_test": x_test,
            "y_train": y_train,
            "train_ids": train_ids,
            "test_ids": test_ids,
            "feature_names": feature_names,
            "useless": useless,
            "health_related": health_related,
            "better_elsewhere": better_elsewhere,
            "bad_format_no_better": bad_format_no_better,
            "binary": binary,
            "one_hot": one_hot,
            "zero_values": zero_values,
            "default_values": default_values,
            "ordinal": ordinal,
            "continuous": continuous,
        }

    return (
        x_train,
        x_test,
        y_train,
        train_ids,
        test_ids,
        feature_names,
        zero_values,
        default_values,
        useless,
        health_related,
        better_elsewhere,
        bad_format_no_better,
        binary,
        one_hot,
        ordinal,
        continuous,
    )


def create_csv_submission(ids, y_pred, name):
    """
    Generate a CSV file for submission to Kaggle or AIcrowd.

    Args:
        ids (list or np.array): Data sample IDs.
        y_pred (list or np.array): Predictions (must be -1 or 1).
        name (str): Filename of the CSV to create.

    Raises:
        ValueError: If `y_pred` contains values other than -1 or 1.
    """
    # Validate predictions
    if not all(i in [-1, 1] for i in y_pred):
        raise ValueError("y_pred can only contain values -1 or 1")

    # Write CSV file
    with open(name, "w", newline="") as csvfile:
        fieldnames = ["Id", "Prediction"]
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({"Id": int(r1), "Prediction": int(r2)})