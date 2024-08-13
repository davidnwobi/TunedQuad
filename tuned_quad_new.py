import numpy as np
from scipy.special import comb
import importlib


def combinations_with_replacement_py(elements, r):
    def generate_combinations(start, current_combination):
        # Base case: when the current combination is of length r
        if len(current_combination) == r:
            result.append(tuple(current_combination))
            return
        
        # Recursive case: try to add each element starting from 'start' index
        for i in range(start, len(elements)):
            current_combination.append(elements[i])
            generate_combinations(i, current_combination)
            current_combination.pop()  # Backtrack
    result = []
    generate_combinations(0, [])
    return result

def eval_poly(var_val, pol_deg, coeffs, intercept):
    '''
    Evaluate an n-degree polynomial with coefficients coeffs at the point var_val.
    The coeffients are ordered in similar fashion to the output of PolynomialFeatures from scikit-learn.
    They can be generated using itertools.combinations_with_replacement.

    Parameters
    ----------

    var_val : list
        A list of values for each variable in the polynomial.
    pol_deg : int
        The degree of the polynomial.
    coeffs : list
        The coefficients of the polynomial.
    intercept : float
        The intercept of the polynomial.

    Returns
    -------
    float
    '''
    res = 0.0
    n_var = len(var_val)
    coeff_iter = 0
    if n_var > pol_deg:
        raise ValueError("Number of variables is greater than polynomial degree")
    
    for d in range(pol_deg + 1):
        for combo in combinations_with_replacement_py(range(n_var), d):
            
            # Calculate the exponents for each variable in this combination.
            exponents = [0] * n_var
            for idx in combo:
                exponents[idx] += 1
            terms = 1.0

            # Calculate the value of the term by raising variables to their respective exponents.
            for i, exp in enumerate(exponents):
                terms *= var_val[i]**exp if exp > 0 else 1.0

            terms *= coeffs[coeff_iter]
            coeff_iter += 1
            
            res += np.prod(terms)
            
    return res + intercept


def integrate_from_model(model_name, func, a, b, params=()):
    model = importlib.import_module(model_name)
    return model.integrate(func, a, b, params)
