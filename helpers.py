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
        feature_names (np.array): list of feature names for training data
        default_values (list of lists): list of default values for each feature
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
        feature_names = f.readline().strip().split(",")[1:]  # skip "Id"
    feature_names = np.array(feature_names)
    
    # The file "default_values.csv" contains default values for each feature
    with open(os.path.join(data_path, "default_values.csv"), "r") as f:
        # No header
        # First column is feature name, second, third, ... are default values
        reader = csv.reader(f, delimiter=",")
        default_values = []
        for row in reader:
            default_values.append([])  # in case no valid default value is found
            for val in row[1:]:
                try:
                    default_values[-1].append(float(val))
                except ValueError:
                    pass  # skip non-numeric default values

    return x_train, x_test, y_train, train_ids, test_ids, feature_names, default_values


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
