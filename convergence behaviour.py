import pandas as pd
import numpy as np
from itertools import product
import plotly.graph_objects as go
import plotly.express as px
from scipy.ndimage import gaussian_filter

n = 62
n_params = 2
A_p = np.log2(np.geomspace(1, 500000, n))
B_p = np.log2(np.geomspace(1, 500000, n))

kronrod = np.loadtxt("tuned_mat_62_behaviour.csv")
kronrod = kronrod.reshape((n, n))
print(kronrod.max())
kronrod = np.log2(kronrod)
kronrod = gaussian_filter(kronrod, 1, mode="nearest")

# px.imshow(kronrod, text_auto=True, labels=dict(x="B", y="A", color="log2(Kronrod)"), title="Kronrod behaviour", x=B_p, y=A_p).show()


A, B = np.meshgrid(A_p, B_p)    
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

degree = n_params
model = make_pipeline(PolynomialFeatures(degree), LinearRegression())

A = A.flatten()
B = B.flatten()
kronrod = kronrod.flatten()
x = np.vstack([A, B]).T

model.fit(x, kronrod)

coeff = model.named_steps['linearregression'].coef_
intercept = model.named_steps['linearregression'].intercept_
coeff_dict = print(dict(zip( model.named_steps['polynomialfeatures'].get_feature_names_out(), coeff.round(4))))

kronrod_pred = model.predict(x)

# Print the coefficients, intercept, fitted values, and original target values
print("Coefficients:", coeff)
print("Intercept:", intercept)
print("Coefficients dict:", coeff_dict)
# print("Fitted values:", kronrod_pred)
# print("Original values:", kronrod)

kronrod_pred = kronrod_pred.reshape((n, n))
kronrod = kronrod.reshape((n, n))   

def percentage_greater(pred, orig, threshold=.99):
    return (pred >= orig).mean() > threshold

l = 0
r = kronrod_pred.max()-kronrod.min()
eps = 1e-6
while(abs(l-r) > eps): # probably not the best way to do this but I know binary search works
    m = (l+r)/2
    if percentage_greater(kronrod_pred+m, kronrod):
        r = m
    else:
        l = m
kronrod_pred+=l

def next_power_of_2(n):
    if n <= 0:
        return 1
    # Initialize position counter
    position = 0
    # Loop to find the position of the highest set bit
    while n > 0:
        n >>= 1
        position += 1
    # Shift 1 left by position to get the next power of 2
    return 1 << position

with np.nditer(kronrod_pred, op_flags=['readwrite']) as it:
    for element in it:
        element[...] = np.log2(min(next_power_of_2(int(2**(element))), 2**15))


fig = go.Figure(data=[go.Surface(z=kronrod_pred, x=B_p, y=A_p, colorscale='Viridis')])
fig.update_layout(title='Kronrod Pred behaviour')

fig.add_surface(z=kronrod, x=B_p, y=A_p, showscale=False, colorscale='Cividis')

fig.show()


{'1': 0.0, 'x0': 0.5707, 'x1': 0.5926, 'x0^2': 0.0094, 'x0 x1': -0.0454, 'x1^2': 0.0079} # possible optimization use horner's method extend to multivariate 
{'1': 0.0, 'x0': 0.3381, 'x1': 0.3474, 'x0^2': 0.0484, 'x0 x1': -0.0581, 'x1^2': 0.0474, 'x0^3': -0.0015, 'x0^2 x1': 0.0004, 'x0 x1^2': 0.0002, 'x1^3': -0.0015}
{'1': 0.0, 'x0': -0.1044, 'x1': -0.0977, 'x0^2': 0.1108, 'x0 x1': -0.0112, 'x1^2': 0.1114, 'x0^3': -0.0035, 'x0^2 x1': -0.0042, 'x0 x1^2': -0.0042, 'x1^3': -0.0037, 'x0^4': -0.0, 'x0^3 x1': 0.0004, 'x0^2 x1^2': -0.0003, 'x0 x1^3': 0.0003, 'x1^4': -0.0}
{'1': 0.0, 'x0': 0.2853, 'x1': 0.5926, 'x2': 0.2853, 'x0^2': 0.0031, 'x0 x1': -0.0227, 'x0 x2': 0.0031, 'x1^2': 0.0079, 'x1 x2': -0.0227, 'x2^2': 0.0031}
{'1': 0.0, 'x0': 0.1691, 'x1': 0.3474, 'x2': 0.1691, 'x0^2': 0.0161, 'x0 x1': -0.029, 'x0 x2': 0.0161, 'x1^2': 0.0474, 'x1 x2': -0.029, 'x2^2': 0.0161, 'x0^3': -0.0004, 'x0^2 x1': 0.0001, 'x0^2 x2': -0.0004, 'x0 x1^2': 0.0001, 'x0 x1 x2': 0.0001, 'x0 x2^2': -0.0004, 'x1^3': -0.0015, 'x1^2 x2': 0.0001, 'x1 x2^2': 0.0001, 'x2^3': -0.0004}
n_params = 2
pol_deg = n_params+1


        

from itertools import combinations_with_replacement 
# NOTE investigate this further tmrw
from collections import defaultdict


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


evaled_poly = np.empty(kronrod_pred.ravel().shape)

for i in range(len(x)):
    evaled_poly[i] = eval_poly(x[i], degree, coeffs=coeff[:], intercept=intercept+l)

for ev, kp in zip(evaled_poly, model.predict(x)+l):
    assert np.allclose(ev, kp)


