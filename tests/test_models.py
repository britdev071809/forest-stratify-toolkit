"""Tests for dynamic stratification model."""

import pandas as pd
import numpy as np
import pytest
from src.models import dynamic_stratify

def test_dynamic_stratify_numeric():
    """Test dynamic stratification with numeric auxiliary variables."""
    n = 100
    data = pd.DataFrame({
        'basal_area': np.random.randn(n),
        'mean_diameter': np.random.randn(n),
        'pc1': np.random.randn(n),
        'pc2': np.random.randn(n),
        'pc3': np.random.randn(n)
    })
    target_vars = ['basal_area', 'mean_diameter']
    aux_vars = ['pc1', 'pc2', 'pc3']
    
    result = dynamic_stratify(data, target_vars, aux_vars, n_strata=3)
    
    assert isinstance(result, pd.DataFrame)
    assert 'stratum' in result.columns
    assert result['stratum'].nunique() == 3
    assert len(result) == n

def test_dynamic_stratify_categorical_raises():
    """Test that categorical auxiliary variables raise TypeError."""
    n = 50
    data = pd.DataFrame({
        'basal_area': np.random.randn(n),
        'development_stage_class': np.random.choice(['young', 'mature', 'old'], n)
    })
    target_vars = ['basal_area']
    aux_vars = ['development_stage_class']
    
    with pytest.raises(TypeError, match="must be numeric"):
        dynamic_stratify(data, target_vars, aux_vars)