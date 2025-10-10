import csv
import numpy as np
import os


def load_csv_data(data_path, max_rows = None):
    """
    This function loads the data and returns the respectinve numpy arrays.
    Remember to put the 3 files in the same folder and to not change the names of the files.

    Args:
        data_path (str): datafolder path
        max_rows (int, optional): If specified, limits the number of rows loaded from each file.

    Returns:
        x_train (np.array): training data
        x_test (np.array): test data
        y_train (np.array): labels for training data in format (-1,1)
        train_ids (np.array): ids of training data
        test_ids (np.array): ids of test data
        feature_names (np.array): list of feature names for training data
        zero_values (dict): dictionary of values representing zero for each feature
        default_values (dict of lists): dictionary of default values for each feature
        useless (np.array): boolean array indicating if a feature is useless because is only a simple combination of other features
        better_elsewhere (np.array): boolean array indicating if a feature has a better format elsewhere
        bad_format_no_better (np.array): boolean array indicating if a feature is in bad format with no better alternative
    """
    
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
        feature_names = f.readline().strip().split(",")[1:]  # skip "Id"
    feature_names = np.array(feature_names)
    
    # The file "default_values.csv" contains default values for each feature
    # First line is header
    # Columns are:
    # - Feature
    # - Value for zero
    # - Combination of other indicators
    # - Bad format, better format elsewhere
    # - Bad format, no better
    # - Value for no response 1
    # - Value for no response 2
    # - ...
    with open(os.path.join(data_path, "default_values.csv"), "r") as f:
        reader = csv.reader(f, delimiter=",")
        next(reader) # Skip header
        
        # Initialize dictionaries and arrays
        zero_values = dict()
        default_values = dict()
        useless = np.zeros(len(feature_names), dtype=bool)
        better_elsewhere = np.zeros(len(feature_names), dtype=bool)
        bad_format_no_better = np.zeros(len(feature_names), dtype=bool)
        
        # Parse the file row by row
        for i, row in enumerate(reader):
            # First column is feature name
            feature_name = row[0]
            
            # Check that feature_name matches the i-th feature of the dataset
            if feature_name != feature_names[i]:
                raise ValueError(f"Feature n°{i} mismatch in default_values.csv: {feature_name} != {feature_names[i]}")
            
            # Second column is the value representing zero
            try:
                zero_values[feature_name] = int(row[1])
            except ValueError:
                zero_values[feature_name] = None  # no zero value
            
            # Third column indicates if the feature is a combination of other indicators
            try:
                if int(row[2]) == 1: # in CSV, True is represented as 1
                    useless[i] = True
            except ValueError:
                useless[i] = False

            # Fourth column indicates if the feature has a better format elsewhere
            try:
                if int(row[3]) == 1:
                    better_elsewhere[i] = True
            except ValueError:
                better_elsewhere[i] = False

            # Fifth column indicates if the feature is in bad format with no better alternative
            try:
                if int(row[4]) == 1:
                    bad_format_no_better[i] = True
            except ValueError:
                bad_format_no_better[i] = False

            # Remaining columns are default values for no response
            default_values[feature_name] = []
            for val in row[5:]:
                try:
                    default_values[feature_name].append(float(val))
                except ValueError:
                    pass  # skip non-numeric default values

    return x_train, x_test, y_train, train_ids, test_ids, feature_names, zero_values, default_values, useless, better_elsewhere, bad_format_no_better


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
