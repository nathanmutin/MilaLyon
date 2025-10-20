import numpy as np  
import csv
import os

def load_csv_data_new(data_path, max_rows = None):
    """
    to write
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
    
    # The file "feature_dataset.csv" contains default values for each feature
    # First line is header
    # Columns are:
    # - Feature
    # - Value for zero
    # - Combination of other indicators
    # - Health related feature
    # - Bad format, better format elsewhere
    # - Bad format, no better
    # - Value for no response 1
    # - Value for no response 2
    # - ...
    with open(os.path.join(data_path, "feature_dataset.csv"), "r") as f:
        reader = csv.reader(f, delimiter=",")
        next(reader) # Skip header
        
        # Initialize dictionaries and arrays
        zero_values = dict()
        default_values = dict()
        useless = np.zeros(len(feature_names), dtype=bool)
        health_related = np.zeros(len(feature_names), dtype=bool)
        better_elsewhere = np.zeros(len(feature_names), dtype=bool)
        bad_format_no_better = np.zeros(len(feature_names), dtype=bool)
        binary = np.zeros(len(feature_names), dtype=bool)
        one_hot = np.zeros(len(feature_names), dtype=bool)
        ordinal = np.zeros(len(feature_names), dtype=bool)
        continuos = np.zeros(len(feature_names), dtype=bool)


        
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
                if (int(row[2]) == 1 | int(row[13]==1)): # in CSV, True is represented as 1
                    useless[i] = True
            except ValueError:
                useless[i] = False

            # Fourth column indicates if the feature is health related
            try:
                if int(row[3]) == 1:
                    health_related[i] = True
            except ValueError:
                health_related[i] = False

            # Fifth column indicates if the feature has a better format elsewhere
            try:
                if int(row[4]) == 1:
                    better_elsewhere[i] = True
            except ValueError:
                better_elsewhere[i] = False

            # Sixth column indicates if the feature is in bad format with no better alternative
            try:
                if int(row[5]) == 1:
                    bad_format_no_better[i] = True
            except ValueError:
                bad_format_no_better[i] = False

            try:
                if int(row[9]) == 1:
                    binary[i] = True
            except ValueError:
                binary[i] = False
                
            try:
                if int(row[10]) == 1:
                    one_hot[i] = True
            except ValueError:
                one_hot[i] = False
                
            try:
                if int(row[11]) == 1:
                    ordinal[i] = True
            except ValueError:
                ordinal[i] = False
                
            try:
                if int(row[12]) == 1:
                    continuos[i] = True
            except ValueError:
                continuos[i] = False
            
            # Remaining columns are default values for no response
            default_values[feature_name] = []
            for val in row[6:9]:
                try:
                    default_values[feature_name].append(float(val))
                except ValueError:
                    pass  # skip non-numeric default values
                

    return x_train, x_test, y_train, train_ids, test_ids, feature_names, zero_values, default_values, useless, health_related, better_elsewhere, bad_format_no_better, binary, one_hot, ordinal, continuos


def replace_default_with_nan(x_train, x_test, feature_names, default_values):
    """
    Replaces default values in the dataset with NaN.

    Args:
        x_train (np.array): shape = (N, D) training feature matrix
        x_test (np.array): shape = (M, D) test feature matrix
        feature_names (list or np.array): feature names corresponding to columns
        default_values (dict of lists): dictionary of default values for each feature

    Returns:
        None: The function modifies x_train and x_test in place.
    """
    for i, feature in enumerate(feature_names):
        for default_value in default_values[feature]:
            x_train[x_train[:, i] == default_value, i] = np.nan
            x_test[x_test[:, i] == default_value, i] = np.nan
            
def drop_too_many_missing(x_train, x_test, train_columns, threshold=0.2):
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

def summarize_feature_flags(feature_names, **flags):
    """
    Summarizes how many features fall under each flag type.

    Args:
        feature_names (list or np.array): names of features (after filtering)
        **flags: named boolean arrays (e.g. binary=binary_flags, ordinal=ordinal_flags, ...)

    Returns:
        None — prints summary to stdout
    """
    total = len(feature_names)
    print(f"\n📊 Feature flag summary ({total} total features):")
    print("-" * 50)

    for name, arr in flags.items():
        arr = np.array(arr)
        if arr.size != total:
            print(f"⚠️  {name} flag length mismatch ({arr.size} vs {total})")
            continue

        count = np.sum(arr)
        pct = 100 * count / total
        print(f"{name:<20} {count:>4}/{total:<4}  ({pct:5.1f}%)")

    print("-" * 50)
    print("✅ Summary complete.\n")
    
def stand_binary(x_train, x_test, feature_names, binary_flags):
    """ Standardize binary features to have values 0 and 1."""
    x_train_mapped = x_train.copy()
    x_test_mapped = x_test.copy()

    for i, feature in enumerate(feature_names):
        if binary_flags[i] == 1:
            col_train = x_train[:, i]
            col_test = x_test[:, i]


            x_train_mapped[:, i] = np.where(col_train == 1, 1,
                                    np.where(col_train == 2, 0, col_train))
            x_test_mapped[:, i] = np.where(col_test == 1, 1,
                                    np.where(col_test == 2, 0, col_test))

    return x_train_mapped, x_test_mapped

def drop_useless(x_train, x_test, feature_names, drop_flags):
            
    # Create mask for features to keep
    keep_mask = np.array(drop_flags) == 0

    # Names of dropped features
    dropped_features = np.array(feature_names)[~keep_mask].tolist()
    
    # Apply mask
    x_train_kept = x_train[:, keep_mask]
    x_test_kept = x_test[:, keep_mask]
    feature_names_kept = np.array(feature_names)[keep_mask].tolist()

    print(f"Dropped {np.sum(~keep_mask)} features, kept {np.sum(keep_mask)}.")
    if dropped_features:
        print("Dropped features:", dropped_features)
    
    return x_train_kept, x_test_kept, keep_mask

def convert_to_times_per_week(x, feature_flags):
    """
    Converts bad-format features to 'times per week'.
    Assumes the feature has been flagged with 1 in feature_flags.

    Args:
        x (np.array): shape (N, D) feature matrix
        feature_flags (np.array): shape (D,), 1 if feature needs conversion

    Returns:
        np.array: x with converted features
    """
    x_converted = x.copy()
    for i, flag in enumerate(feature_flags):
        if flag == 1:
            col = x[:, i].astype(float)
            
            # Example conversion:
            # If values are coded like:
            # 101-199 → times per week (divide by 1 if already weekly)
            # 201-299 → times per month → divide by 4.33 to get per week
            # 888 → Never → map to 0
            # 777 → Don't know → leave as np.nan
            # 999 → Refused → leave as np.nan

            # Replace "Never" and "Don't know / Refused" with NaN
            col[col == 888] = 0      # Never = 0 times/week
            col[col == 777] = np.nan
            col[col == 999] = np.nan

            # Convert month codes (example)
            col[(col >= 201) & (col <= 299)] /= 4.33  # approximate month→week

            # Update column
            x_converted[:, i] = col

    return x_converted

def clip_outliers(x, feature_names, continuos_flags, n_std=3):
    """
    Clips outliers in continuous features at mean ± n_std*std.

    Args:
        x (np.array): shape (N, D) feature matrix
        feature_names (list): names of features
        continuos_flags (np.array or list of 0/1): 1 if the feature is continuous
        n_std (float): number of standard deviations for clipping

    Returns:
        x_clipped (np.array): feature matrix with clipped continuous features
        clipped_counts (dict): number of values clipped per continuous feature
    """
    x_clipped = x.copy()
    clipped_counts = {}

    for i, feature in enumerate(feature_names):
        if continuos_flags[i] == 1:
            col = x[:, i].astype(float)
            mean = np.nanmean(col)
            std = np.nanstd(col)
            min_val = mean - n_std * std
            max_val = mean + n_std * std

            # Count clipped values
            below = np.sum(col < min_val)
            above = np.sum(col > max_val)
            clipped_counts[feature] = below + above

            # Clip values
            col = np.clip(col, min_val, max_val)
            x_clipped[:, i] = col

    return x_clipped, clipped_counts

def standardize_continuous_features(x_train, x_test, feature_names, continuos_flags):
    """
    Standardizes continuous features (zero mean, unit variance) using only training statistics.

    Args:
        x_train (np.ndarray): training feature matrix
        x_test (np.ndarray): test feature matrix
        feature_names (list or np.ndarray): names of all features
        continuos_flags (np.ndarray): boolean mask of continuous features

    Returns:
        x_train_std (np.ndarray): standardized training data
        x_test_std (np.ndarray): standardized test data
        stats (dict): mean and std per standardized feature
    """
    x_train_std = x_train.copy().astype(float)
    x_test_std = x_test.copy().astype(float)
    stats = {}

    continuous_indices = np.where(continuos_flags)[0]

    for i in continuous_indices:
        col_train = x_train[:, i].astype(float)
        mean = np.nanmean(col_train)
        std = np.nanstd(col_train)
        if std == 0 or np.isnan(std):
            std = 1.0  # avoid division by zero

        # Standardize train and test using training stats
        x_train_std[:, i] = (x_train[:, i] - mean) / std
        x_test_std[:, i] = (x_test[:, i] - mean) / std

        

    print(f"✅ Standardized {len(continuous_indices)} continuous features.")
    return x_train_std, x_test_std

def one_hot_encode(x, feature_names, one_hot_flags):
    """
    One-hot encodes categorical features flagged in one_hot_flags.

    Args:
        x (np.array): shape (N, D) feature matrix
        feature_names (list): list of feature names (length D)
        one_hot_flags (np.array or list of 0/1): 1 if the feature should be one-hot encoded

    Returns:
        x_encoded (np.array): new feature matrix with one-hot columns
        feature_names_encoded (list): updated feature names
    """
    x_encoded_list = []
    feature_names_encoded = []

    for i, feature in enumerate(feature_names):
        col = x[:, i]
        if one_hot_flags[i] == 1:
            # Find unique values ignoring NaN
            unique_vals = np.unique(col[~np.isnan(col)])
            
            # Create one-hot columns
            for val in unique_vals:
                new_col = (col == val).astype(float)
                x_encoded_list.append(new_col.reshape(-1, 1))
                feature_names_encoded.append(f"{feature}_{int(val)}")
        else:
            # Keep original column
            x_encoded_list.append(col.reshape(-1, 1))
            feature_names_encoded.append(feature)

    # Concatenate all columns
    x_encoded = np.hstack(x_encoded_list)

    return x_encoded, feature_names_encoded

def expand_flags_after_onehot(feature_names_old, feature_names_new, flags_old):
    """
    Expands flags to match one-hot encoded feature list.
    Each original flag value is copied to all derived one-hot columns.

    Args:
        feature_names_old (list): feature names before encoding
        feature_names_new (list): feature names after encoding
        flags_old (np.array or list): original flags per feature

    Returns:
        np.array: expanded flags matching feature_names_new
    """
    expanded_flags = []

    for old_name, flag in zip(feature_names_old, flags_old):
        # find all one-hot columns derived from this original feature
        matches = [f for f in feature_names_new if f.startswith(old_name)]
        if matches:
            expanded_flags.extend([flag] * len(matches))
        else:
            expanded_flags.append(flag)

    return np.array(expanded_flags)

def mean_mode_imputation(x_train, x_test, continuous_flags):
    """
    Applies mean imputation to continuous features and mode imputation to others.

    Args:
        x_train (np.array): training data (N, D)
        x_test (np.array): test data (M, D)
        continuous_flags (np.array of 0/1): 1 = continuous, 0 = categorical

    Returns:
        x_train_filled, x_test_filled: imputed datasets
    """
    x_train_filled = x_train.copy()
    x_test_filled = x_test.copy()

    cont_idx = np.where(continuous_flags == 1)[0]
    cat_idx = np.where(continuous_flags == 0)[0]

    # Mean imputation for continuous
    if len(cont_idx) > 0:
        x_train_filled[:, cont_idx], x_test_filled[:, cont_idx] = mean_imputation(
            x_train[:, cont_idx], x_test[:, cont_idx]
        )

    # Mode imputation for categorical
    if len(cat_idx) > 0:
        x_train_filled[:, cat_idx], x_test_filled[:, cat_idx] = mode_imputation(
            x_train[:, cat_idx], x_test[:, cat_idx]
        )

    return x_train_filled, x_test_filled

def health_related_selection(x_train, x_test, feature_names, health_related_flags):
    """
    Selects only health-related features based on provided flags.

    Args:
        x (np.array): shape=(N, D) feature matrix
        feature_names (list): list of feature names
        health_related_flags (np.array or list of 0/1): 1 if feature is health-related
    Returns:
        x_health (np.array): reduced feature matrix with only health-related features
        feature_names_health (list): reduced feature names  
    """
    x_train_health = x_train[:,health_related_flags]
    x_test_health = x_test[:,health_related_flags]
    feature_names_health = np.array(feature_names)[health_related_flags]
    
    return x_train_health, x_test_health, feature_names_health

    print(f"Selected {np.sum(keep_mask)} health-related features out of {len(feature_names)} total.")
    return x_health, feature_names_health

def remove_highly_correlated(x, feature_names, threshold=0.9): #can be changed
    """
    Remove one feature from each pair of highly correlated features.

    Args:
        x (np.array): shape=(N, D)
        feature_names (list): list of feature names
        threshold (float): correlation threshold above which to remove a feature

    Returns:
        x_new (np.array): reduced feature matrix
        feature_names_new (list): reduced feature names
    """
    corr = np.corrcoef(x, rowvar=False)
    D = corr.shape[0]
    to_remove = set()

    for i in range(D):
        for j in range(i + 1, D):
            if abs(corr[i, j]) > threshold:
                # Mark the second feature for removal
                to_remove.add(j)
                print(f"Dropping {feature_names[j]} because it's highly correlated with {feature_names[i]} ({corr[i,j]:.2f})")

    keep_mask = np.array([i not in to_remove for i in range(D)])
    x_new = x[:, keep_mask]
    feature_names_new = np.array(feature_names)[keep_mask].tolist()

    return x_new, feature_names_new

def mean_imputation(x_train, x_test):
    """Impute missing values with the mean of each feature. 
    Drops columns that are entirely NaN.

    Args:
        x_train (np.array): shape=(N,D) training feature matrix with NaNs for missing values
        x_test (np.array): shape=(M,D) test feature matrix with NaNs for missing values

    Returns:
        None: The function modifies x_train and x_test in place.
    """
    # Compute means on the training set (ignoring NaNs)
    mean_x = np.nanmean(x_train, axis=0)

    # Impute training set
    inds_train = np.where(np.isnan(x_train))
    x_train[inds_train] = mean_x[inds_train[1]]

    # Impute test set using train means
    inds_test = np.where(np.isnan(x_test))
    x_test[inds_test] = mean_x[inds_test[1]]
    
    return x_train, x_test

def mode_imputation(x_train, x_test):
    """
    Impute missing values with the mode (most frequent value) of each feature using only NumPy.
    Drops columns that are entirely NaN.

    Args:
        x_train (np.array): shape=(N,D)
        x_test (np.array): shape=(M,D)

    Returns:
        x_train_imputed, x_test_imputed
    """
    x_train_imputed = x_train.copy()
    x_test_imputed = x_test.copy()

    D = x_train.shape[1]
    mode_x = np.full(D, np.nan)

    for i in range(D):
        col = x_train[:, i]
        valid = col[~np.isnan(col)]
        if valid.size > 0:
            # Compute mode manually
            vals, counts = np.unique(valid, return_counts=True)
            mode_x[i] = vals[np.argmax(counts)]

    # Impute train
    inds_train = np.where(np.isnan(x_train_imputed))
    x_train_imputed[inds_train] = mode_x[inds_train[1]]

    # Impute test using train mode
    inds_test = np.where(np.isnan(x_test_imputed))
    x_test_imputed[inds_test] = mode_x[inds_test[1]]

    return x_train_imputed, x_test_imputed

def preprocess_data(data_folder, missing_threshold=0.2, health_selection = True, outlier_std=3, one_hot_selection = True, apply_mean_mode_imputation = True, corr_threshold=0.9):
    """
    Full preprocessing pipeline for the dataset.

    Args:
        data_path (str): path to the dataset directory"""
        
    x_train, x_test, y_train, train_ids, test_ids, feature_names, zero_values, default_values, useless, health_related, better_elsewhere, bad_format_no_better, binary, one_hot, ordinal, continuos = _data = load_csv_data_new(data_folder)
    replace_default_with_nan(x_train, x_test, feature_names, default_values)
    x_train, x_test, feature_names_drop = drop_too_many_missing(x_train, x_test, feature_names, threshold=missing_threshold)
    useless = useless[feature_names_drop]
    health_related = health_related[feature_names_drop]
    better_elsewhere = better_elsewhere[feature_names_drop]
    bad_format_no_better = bad_format_no_better[feature_names_drop]
    binary = binary[feature_names_drop]
    one_hot = one_hot[feature_names_drop]
    ordinal = ordinal[feature_names_drop]
    continuos = continuos[feature_names_drop]
    feature_names = feature_names[feature_names_drop]
    
    if health_selection:
        x_train, x_test, feature_names = health_related_selection(x_train, x_test, feature_names, health_related)
        useless = useless[health_related]
        better_elsewhere = better_elsewhere[health_related]
        bad_format_no_better = bad_format_no_better[health_related]
        binary = binary[health_related]
        one_hot = one_hot[health_related]
        ordinal = ordinal[health_related]
        continuos = continuos[health_related]
        
    summarize_feature_flags(feature_names,
                            binary=binary,
                            one_hot=one_hot,
                            ordinal=ordinal,
                            continuos=continuos,
                            useless=useless,
                            better_elsewhere=better_elsewhere,
                            bad_format_no_better=bad_format_no_better
                            )
    
    x_train = convert_to_times_per_week(x_train, bad_format_no_better)
    x_test = convert_to_times_per_week(x_test, bad_format_no_better)
    
    x_train, outlier_counts = clip_outliers(x_train, feature_names, continuos, n_std=outlier_std)
    x_test, _ = clip_outliers(x_test, feature_names, continuos, n_std=outlier_std)  
    
    x_train, x_test = stand_binary(x_train, x_test, feature_names, binary)
    x_train, x_test, keep_mask = drop_useless(x_train, x_test, feature_names, useless)
    better_elsewhere = better_elsewhere[keep_mask]
    bad_format_no_better = bad_format_no_better[keep_mask]
    binary = binary[keep_mask]
    one_hot = one_hot[keep_mask]
    ordinal = ordinal[keep_mask]
    feature_names = feature_names[keep_mask]
    continuos = continuos[keep_mask]
    
    if one_hot_selection:
        x_train, feature_names_oh = one_hot_encode(x_train, feature_names, one_hot)
        x_test, _ = one_hot_encode(x_test, feature_names, one_hot)
        binary = expand_flags_after_onehot(feature_names, feature_names_oh, binary)
        ordinal = expand_flags_after_onehot(feature_names, feature_names_oh, ordinal)
        continuos = expand_flags_after_onehot(feature_names, feature_names_oh, continuos)
        feature_names = feature_names_oh
    
    if apply_mean_mode_imputation:
        x_train, x_test = mean_mode_imputation(x_train, x_test, continuos)
        
    x_train, x_test = standardize_continuous_features(x_train, x_test, feature_names, continuos)
    x_train, feature_names_cor = remove_highly_correlated(x_train, feature_names, threshold=corr_threshold)
    x_test, _ = remove_highly_correlated(x_test, feature_names, threshold=corr_threshold)   
    
    return x_train, x_test, y_train, feature_names
    
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def logistic_regression_train(x_train, y_train, lr=0.01, epochs=1000, lambda_=0.0):
    """Train logistic regression using gradient descent."""
    N, D = x_train.shape
    w = np.zeros(D)
    b = 0.0

    for _ in range(epochs):
        z = np.dot(x_train, w) + b
        y_pred = sigmoid(z)

        # Gradient of loss with L2 regularization
        dw = (1/N) * np.dot(x_train.T, (y_pred - y_train)) + lambda_ * w
        db = (1/N) * np.sum(y_pred - y_train)

        w -= lr * dw
        b -= lr * db

    return w, b


def logistic_regression_predict(x, w, b, threshold=0.5):
    """Predict binary class labels."""
    probs = sigmoid(np.dot(x, w) + b)
    return (probs >= threshold).astype(int), probs


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


def split_train_val(x_train, y_train, val_size=0.1, random_seed=42):
    """
    Splits the training data into training and validation sets.

    Args:
        x_train (np.array): shape = (N, D) training feature matrix
        y_train (np.array): shape = (N,) target values
        val_size (float): fraction of data to use for validation
        random_seed (int): random seed for reproducibility
    Returns:
        x_train_new (np.array): shape = (N*(1-val_size), D) new training feature matrix
        y_train_new (np.array): shape = (N*(1-val_size),) new training target values
        x_val (np.array): shape = (N*val_size, D) validation feature matrix
        y_val (np.array): shape = (N*val_size,) validation target values
    """
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
