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
    n_strata : int, default=5
        Number of strata to create.
        
    Returns
    -------
    pandas.DataFrame
        Data with an additional 'stratum' column indicating stratum membership.
    """
    # Check that auxiliary variables are numeric
    for var in aux_vars:
        if not pd.api.types.is_numeric_dtype(data[var]):
            raise TypeError(f"Auxiliary variable '{var}' must be numeric, got {data[var].dtype}")
    
    # Extract auxiliary data
    X = data[aux_vars].values
    
    # Perform k-means clustering
    kmeans = KMeans(n_clusters=n_strata, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    
    # Assign stratum labels
    result = data.copy()
    result['stratum'] = labels
    
    return result