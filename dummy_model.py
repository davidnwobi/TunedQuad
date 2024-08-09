import numpy as np
from scipy.special import comb

__all__ = ["model_name", "n_params", "poly_degree", "coeffs", "intercept"]
model_name = "dummy"
n_params = 2
poly_degree = 3
coeffs = comb(n_params + poly_degree, poly_degree, exact=True)
intercept  = 2.0





