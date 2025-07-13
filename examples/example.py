#!/usr/bin/env python3
"""
Example usage of pyhtelasso package

This example demonstrates:
1. Basic usage of TreatmentHeterogeneityDetector
2. Data simulation with heterogeneous treatment effects
3. Model fitting and result interpretation
4. Performance evaluation
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyhtelasso import HTELasso

# Simulate data with heterogeneous treatment effects
def simulate_data(n=1000, p_features=20, p_het=5, treatment_prob=0.5,
                  noise_level=1.0, effect_size=2.0, random_state=42):
    """
    Simulate data with heterogeneous treatment effects
    
    Parameters:
    -----------
    n : int
        Number of observations
    p_features : int
        Total number of features
    p_het : int
        Number of features that moderate treatment effect
    treatment_prob : float
        Probability of treatment assignment
    noise_level : float
        Standard deviation of noise
    effect_size : float
        Magnitude of treatment effect
    random_state : int
        Random seed
        
    Returns:
    --------
    X, y, t : arrays
        Simulated data
    true_moderators : array
        Indices of true moderating variables
    """
    np.random.seed(random_state)
    
    # Generate covariates
    X = np.random.randn(n, p_features)
    
    # Generate treatment assignment
    t = np.random.binomial(1, treatment_prob, n)
    
    # True moderators (first p_het features)
    true_moderators = np.arange(p_het)
    
    # Generate outcome with heterogeneous treatment effects
    # Base outcome depends on some covariates
    y_base = X[:, :3].sum(axis=1) + np.random.normal(0, noise_level, n)
    
    # Treatment effect depends on moderating variables
    treatment_effect = effect_size * (1 + X[:, true_moderators].sum(axis=1))
    
    # Final outcome
    y = y_base + t * treatment_effect
    
    return X, y, t, true_moderators

# Simulate data with heterogeneous treatment effects

X, y, t, true_moderators = simulate_data(
    n=1000, p_features=20, p_het=3, 
    treatment_prob=0.4, random_state=42
)

# Create feature names
feature_names = [f"X{i}" for i in range(X.shape[1])]

# Initialize the heterogeneity detector
detector = HTELasso(
    alpha=None,              # Auto-select regularization via CV
    random_state=42,         # Random seed
    fit_intercept=True       # Fit intercept term
)

print("Fitting HTELasso...")
detector.fit(X, y, t)


# Print statsmodels-like summary
print("\n" + "="*80)
print("REGRESSION SUMMARY")
print("="*80)
print(detector.summary(feature_names))
