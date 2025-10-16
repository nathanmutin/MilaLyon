import numpy as np

def normalize(x, x_test=None):
    """Normalizes the data set to have zero mean and unit variance.

    Args:
        x (np.array): shape=(N,D) feature matrix
        x_test (np.array, optional): shape=(M,D) test feature matrix.

    Returns:
        np.array: shape=(N,D) normalized feature matrix
        np.array: shape=(M,D) normalized test feature matrix (if provided)
    """
    mean_x = np.nanmean(x, axis=0)
    std_x = np.nanstd(x, axis=0)
    std_x[std_x == 0] = 1.0  # avoid division by zero    
    
    x = (x - mean_x) / std_x
    
    if x_test is not None:
        x_test = (x_test - mean_x) / std_x
        return x, x_test
    return x

def min_max_normalize(x, x_test=None):
    """Min-max normalizes the data set to the range [0, 1].

    Args:
        x (np.array): shape=(N,D) feature matrix
        x_test (np.array, optional): shape=(M,D) test feature matrix.

    Returns:
        np.array: shape=(N,D) normalized feature matrix
        np.array: shape=(M,D) normalized test feature matrix (if provided)
    """

    min_x = np.nanmin(x, axis=0)
    max_x = np.nanmax(x, axis=0)
    range_x = max_x - min_x  
    range_x[range_x == 0] = 1. # avoid division by zero 

    x = (x - min_x) / range_x
    
    if x_test is not None:
        x_test = (x_test - min_x) / range_x
        return x, x_test
    return x

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

def replace_by_zero(x_train, x_test, zero_values):
    """
    Replaces specified values in the dataset with zero.

    Args:
        x_train (np.array): shape = (N, D) training feature matrix
        x_test (np.array): shape = (M, D) test feature matrix
        zero_values (dict): dictionary of values representing zero for each feature (can be NaN)

    Returns:
        None: The function modifies x_train and x_test in place.
    """
    for i, feature in enumerate(zero_values):
        zero_value = zero_values[feature]
        
        x_train[x_train[:, i] == zero_value, i] = 0
        x_test[x_test[:, i] == zero_value, i] = 0
        
        if zero_value != None and np.isnan(zero_value):
            x_train[np.isnan(x_train[:, i]), i] = 0
            x_test[np.isnan(x_test[:,i]),i]=0

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

def identify_too_many_missing(x_train, feature_names, threshold=0.2):
    """
    Return feature names with more than a given percentage of missing values (NaNs).

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
    dropped_names = [feature_names[i] for i in dropped_cols]
    return dropped_names

def identify_low_correlation(x_train, y_train, feature_names, threshold=0.1):
    """
    Return feature names with correlation to target below a given threshold.

    Args:
        x_train (np.array): shape = (N, D) training feature matrix
        y_train (np.array): shape = (N,) target values
        feature_names (list or np.array): feature names corresponding to columns
        threshold (float): minimum absolute correlation to keep a feature

    Returns:
        low_corr_names (list): names of features with low correlation to target
        correlations (np.array): array of correlation values for all features
    """
    correlations = np.array([np.corrcoef(x_train[:, i], y_train)[0, 1] for i in range(x_train.shape[1])])
    low_corr_mask = np.abs(correlations) < threshold
    low_corr_names = [feature_names[i] for i in np.where(low_corr_mask)[0]]
    
    return low_corr_names, correlations


def replace_nan(x_train, x_test):
    for i in range(x_train.shape[1]):
        # Get non-nan values
        non_nan_values = x_train[~np.isnan(x_train[:, i]), i]
        unique_values = np.unique(non_nan_values)
        
        if len(unique_values) < 10:
            # Use mode for categorical features
            mode = np.bincount(non_nan_values.astype(int)).argmax()
            x_train[np.isnan(x_train[:, i]), i] = mode
            x_test[np.isnan(x_test[:, i]), i] = mode
        else:
            # Use mean for continuous features
            mean = np.mean(non_nan_values)
            x_train[np.isnan(x_train[:, i]), i] = mean
            x_test[np.isnan(x_test[:, i]), i] = mean
    return x_train, x_test

def detect_feature_type(x, cat_threshold=11):
    """
    Automatically detect feature type.
    
    Args:
        x (np.array): shape = (N,D) feature matrix
        cat_threshold (int): maximum number of unique values to consider a feature categorical
    
    Returns:
        cat (np.array): shape = (D,) array of strings indicating feature type ('binary', 'categorical', 'continuous', 'constant', 'unknown')
    """
    cat = np.empty(x.shape[1], dtype=object)
    
    for i in range(x.shape[1]):
        n_unique_vals = len(np.unique(x[~np.isnan(x[:, i]), i]))  # ignore NaNs
        if n_unique_vals == 0:
            cat[i] = 'unknown'  # all values are NaN
        elif n_unique_vals == 1:
            cat[i] = 'constant'  # only one unique value
        elif n_unique_vals == 2:
            cat[i] = 'binary'
        elif n_unique_vals <= cat_threshold:
            cat[i] = 'categorical'
        else:
            cat[i] = 'continuous'
    
    return cat

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

def clip_outliers(x_train, x_test=None, n_std=3):
    """
    Clips outliers to within mean ± n_std * std for each feature.
    Reports how many values were clipped.

    Args:
        x_train (np.array): shape=(N,D) training feature matrix
        x_test (np.array, optional): shape=(M,D) test feature matrix
        n_std (float): number of standard deviations for clipping

    Returns:
        tuple:
            x_train_clipped (np.array): clipped training data
            x_test_clipped (np.array or None): clipped test data (if provided)
            n_clipped (int): number of values clipped in training set
    """
    mean = np.mean(x_train, axis=0)
    std = np.std(x_train, axis=0)
    
    clip_min = mean - n_std * std
    clip_max = mean + n_std * std

    # Count values to be clipped (before applying np.clip)
    below_min = x_train < clip_min
    above_max = x_train > clip_max
    n_clipped = np.sum(below_min | above_max)

    # Apply clipping
    x_train_clipped = np.clip(x_train, clip_min, clip_max)

    print(f"Clipped {n_clipped} values in x_train "
          f"({n_clipped / x_train.size * 100:.2f}% of all entries)")

    if x_test is not None:
        below_min_test = x_test < clip_min
        above_max_test = x_test > clip_max
        n_clipped_test = np.sum(below_min_test | above_max_test)
        x_test_clipped = np.clip(x_test, clip_min, clip_max)
        print(f"Clipped {n_clipped_test} values in x_test "
              f"({n_clipped_test / x_test.size * 100:.2f}%)")
        return x_train_clipped, x_test_clipped, n_clipped, n_clipped_test

    return x_train_clipped, n_clipped

def pca_reduce(x_train, x_test=None, variance_threshold=0.95):
    """
    Perform PCA and reduce dimensionality to preserve given variance.

    Args:
        x_train (np.array): training data, shape (N, D)
        x_test (np.array, optional): test data, shape (M, D)
        variance_threshold (float): fraction of variance to keep (e.g. 0.95)

    Returns:
        x_train_pca (np.array)
        x_test_pca (np.array or None)
        eigvecs (np.array): principal component directions
        explained_variance (np.array): explained variance ratio per component
    """
  

    # Covariance
    cov = np.cov(x_train, rowvar=False)

    # Eigen decomposition
    eigvals, eigvecs = np.linalg.eigh(cov)

    # Sort by descending eigenvalue
    idx = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[idx], eigvecs[:, idx]

    # Compute explained variance
    explained_variance = eigvals / np.sum(eigvals)
    cumulative_variance = np.cumsum(explained_variance)

    # Determine number of components
    k = np.searchsorted(cumulative_variance, variance_threshold) + 1
    print(f"Keeping {k} components explaining {cumulative_variance[k-1]*100:.2f}% variance")

    # Project data
    x_train_pca = np.dot(x_train, eigvecs[:, :k])

    if x_test is not None:
        x_test_pca = np.dot(x_test, eigvecs[:, :k])
        return x_train_pca, x_test_pca, eigvecs[:, :k], explained_variance[:k]

    return x_train_pca, eigvecs[:, :k], explained_variance[:k]


def drop_features_from_dictionnary(data_dict, feature_names_to_drop):
    """
    This function drops features from the data dictionnary.

    Args:
        data_dict (dict): dictionnary containing the data and meta-data
        feature_names_to_drop (list): list of feature names to drop
    Returns:
        data_dict (dict): dictionnary containing the data and meta-data with features dropped
    """
    for feature_name in feature_names_to_drop:
        if feature_name in data_dict['feature_names']:
            index = np.where(data_dict['feature_names'] == feature_name)[0][0]
            data_dict['x_train'] = np.delete(data_dict['x_train'], index, axis=1)
            data_dict['x_test'] = np.delete(data_dict['x_test'], index, axis=1)
            data_dict['feature_names'] = np.delete(data_dict['feature_names'], index)
            data_dict['useless'] = np.delete(data_dict['useless'], index)
            data_dict['health_related'] = np.delete(data_dict['health_related'], index)
            data_dict['better_elsewhere'] = np.delete(data_dict['better_elsewhere'], index)
            data_dict['bad_format_no_better'] = np.delete(data_dict['bad_format_no_better'], index)
            data_dict['binary'] = np.delete(data_dict['binary'], index)
            data_dict['one_hot'] = np.delete(data_dict['one_hot'], index)
        else:
            print(f"Feature {feature_name} not found in feature names.")

def one_hot_encode(data_dict):
    """
    This function one-hot encodes the categorical features in the data dictionnary.

    Args:
        data_dict (dict): dictionnary containing the data and meta-data
    Returns:
        data_dict (dict): dictionnary containing the data and meta-data with categorical features one-hot encoded
    """
    one_hot_features = data_dict['feature_names'][data_dict['one_hot'] == 1]
    binary_features = data_dict['feature_names'][data_dict['binary'] == 1]

    for feature in one_hot_features:
        if feature not in binary_features:
            # Get unique values for the feature
            unique_values = np.unique(data_dict['x_train'][:, np.where(data_dict['feature_names'] == feature)[0][0]])
            for value in unique_values:
                # Create one-hot encoded feature
                idx = np.where(data_dict['feature_names'] == feature)[0][0]

                one_hot_encoded = (data_dict['x_train'][:, idx] == value).astype(int)
                data_dict['x_train'] = np.column_stack((data_dict['x_train'], one_hot_encoded))

                one_hot_encoded_test = (data_dict['x_test'][:, idx] == value).astype(int)
                data_dict['x_test'] = np.column_stack((data_dict['x_test'], one_hot_encoded_test))

                data_dict['feature_names'] = np.append(data_dict['feature_names'], f"{feature}_{value}")
                data_dict['useless'] = np.append(data_dict['useless'], data_dict['useless'][idx])
                data_dict['health_related'] = np.append(data_dict['health_related'], data_dict['health_related'][idx])
                data_dict['better_elsewhere'] = np.append(data_dict['better_elsewhere'], data_dict['better_elsewhere'][idx])
                data_dict['bad_format_no_better'] = np.append(data_dict['bad_format_no_better'], data_dict['bad_format_no_better'][idx])
                data_dict['binary'] = np.append(data_dict['binary'], 1)
                data_dict['one_hot'] = np.append(data_dict['one_hot'], 1)

            # Drop the original feature
            idx = np.where(data_dict['feature_names'] == feature)[0][0]
            data_dict['x_train'] = np.delete(data_dict['x_train'], idx, axis=1)
            data_dict['x_test'] = np.delete(data_dict['x_test'], idx, axis=1)
            data_dict['feature_names'] = np.delete(data_dict['feature_names'], idx)
            data_dict['useless'] = np.delete(data_dict['useless'], idx)
            data_dict['health_related'] = np.delete(data_dict['health_related'], idx)
            data_dict['better_elsewhere'] = np.delete(data_dict['better_elsewhere'], idx)
            data_dict['bad_format_no_better'] = np.delete(data_dict['bad_format_no_better'], idx)
            data_dict['binary'] = np.delete(data_dict['binary'], idx)
            data_dict['one_hot'] = np.delete(data_dict['one_hot'], idx)
