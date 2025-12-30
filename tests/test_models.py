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

def test_dynamic_stratify_with_categorical_aux():
    """Test dynamic stratification with categorical auxiliary variables."""
    n = 50
    data = pd.DataFrame({
        'basal_area': np.random.randn(n),
        'development_stage_class': np.random.choice(['young', 'mature', 'old'], n),
        'species_mix': np.random.choice(['pine', 'spruce', 'fir'], n)
    })
    target_vars = ['basal_area']
    aux_vars = ['development_stage_class', 'species_mix']
    
    # Should not raise TypeError
    result = dynamic_stratify(data, target_vars, aux_vars, n_strata=2)
    
    assert isinstance(result, pd.DataFrame)
    assert 'stratum' in result.columns
    assert result['stratum'].nunique() == 2
    assert len(result) == n
    # Ensure original columns unchanged
    assert 'development_stage_class' in result.columns
    assert 'species_mix' in result.columns

def test_dynamic_stratify_mixed_numeric_categorical():
    """Test with both numeric and categorical auxiliary variables."""
    n = 80
    data = pd.DataFrame({
        'basal_area': np.random.randn(n),
        'mean_height': np.random.randn(n),
        'pc1': np.random.randn(n),
        'development_stage_class': np.random.choice(['young', 'mature', 'old'], n)
    })
    target_vars = ['basal_area', 'mean_height']
    aux_vars = ['pc1', 'development_stage_class']
    
    result = dynamic_stratify(data, target_vars, aux_vars, n_strata=4)
    
    assert isinstance(result, pd.DataFrame)
    assert 'stratum' in result.columns
    assert result['stratum'].nunique() == 4
    assert len(result) == n

def test_dynamic_stratify_no_aux_vars():
    """Test error when no auxiliary variables provided."""
    data = pd.DataFrame({'basal_area': [1, 2, 3]})
    with pytest.raises(ValueError, match="No valid auxiliary variables"):
        dynamic_stratify(data, ['basal_area'], [])