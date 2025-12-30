"""Dynamic stratification model for estimating stand-level forest characteristics."""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

def dynamic_stratify(data, target_vars, aux_vars, n_strata=5):
    """
    Perform dynamic stratification using auxiliary variables.
    
    Parameters
    ----------
    data : pandas.DataFrame
        Input data containing target and auxiliary variables.
    target_vars : list of str
        Column names of target variables (basal area, mean diameter, etc.).
    aux_vars : list of str
        Column names of auxiliary variables (Landsat TM PCs, etc.).
        Can include numeric and categorical (object or category) columns.
    n_strata : int, default=5
        Number of strata to create.
        
    Returns
    -------
    pandas.DataFrame
        Data with an additional 'stratum' column indicating stratum membership.
    """
    # Separate numeric and categorical auxiliary variables
    numeric_vars = []
    categorical_vars = []
    
    for var in aux_vars:
        if pd.api.types.is_numeric_dtype(data[var]):
            numeric_vars.append(var)
        else:
            # Assume categorical (object, category, string)
            categorical_vars.append(var)
    
    # Prepare numeric data
    numeric_data = data[numeric_vars] if numeric_vars else pd.DataFrame(index=data.index)
    
    # Prepare categorical data via one-hot encoding
    categorical_dummies = pd.DataFrame(index=data.index)
    if categorical_vars:
        # Use get_dummies with prefix to avoid column name collisions
        categorical_dummies = pd.get_dummies(data[categorical_vars], prefix_sep='_', drop_first=False)
    
    # Combine numeric and dummy variables
    X = pd.concat([numeric_data, categorical_dummies], axis=1)
    
    # Ensure there is at least one feature
    if X.shape[1] == 0:
        raise ValueError("No valid auxiliary variables provided.")
    
    # Perform k-means clustering
    kmeans = KMeans(n_clusters=n_strata, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    
    # Assign stratum labels
    result = data.copy()
    result['stratum'] = labels
    
    return result