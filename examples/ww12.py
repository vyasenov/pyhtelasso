import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import numpy as np
from pyhtelasso import HTELasso

# Simulate data with heterogeneous treatment effects
np.random.seed(1988)
n, p_features, p_het = 1000, 20, 3
X = np.random.randn(n, p_features)
t = np.random.binomial(1, 0.4, n)
y_base = X[:, :3].sum(axis=1) + np.random.normal(0, 1, n)
treatment_effect = 2.0 * (1 + X[:, :p_het].sum(axis=1))
y = y_base + t * treatment_effect

# Initialize and fit the detector
detector = HTELasso(
    alpha=None,              # Auto-select regularization via CV
    random_state=1988,         # Random seed
    fit_intercept=True       # Fit intercept term
)
detector.fit(X, y, t)

# Print detailed summary
print(detector.summary())
