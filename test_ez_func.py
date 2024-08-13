import numpy as np
from pyparsing import line
# from numba import float64, njit, vectorize
# from j1 import j1
from quad_tuning_new import fit_model
from tuned_quad_new import integrate_from_model

a = 0
b = np.pi/2

def integrand(x,b):
    return x**7*np.sin(b*x)


# (7 ((π^6 b^6)/64 - (15 π^4 b^4)/8 + 90 π^2 b^2 - 720) sin((π b)/2) + 1/2 π b (-1/64 π^6 b^6 + (21 π^4 b^4)/8 - 210 π^2 b^2 + 5040) cos((π b)/2))/b^8
def analyt_integrand(b):
    return 1/b**8*(7*((np.pi**6 * b**6)/64 - (15 * np.pi**4 * b**4)/8 + 90 * np.pi**2 * b**2 - 720) * np.sin(np.pi*b/2) + 1/2 * np.pi * b * (-1/64 * np.pi**6 * b**6 + (21 * np.pi**4 * b**4)/8 - 210 * np.pi**2 * b**2 + 5040) * np.cos(np.pi*b/2))

n = 60

reg_params2 = [np.geomspace(1, 100000, n)]
params_names = ["b"]
fit_model("ezy_func", integrand, a, b, params_names, reg_params2, np.geomspace(1e-5, 1e-1, n), update=True)

print(integrate_from_model("ezy_func_model", integrand, a, b, (1e-5, 10000)))
print(analyt_integrand(10000))

