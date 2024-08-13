import numpy as np
from pyparsing import line
# from numba import float64, njit, vectorize
# from j1 import j1
from scipy.special import j1
from quad_tuning_new import fit_model
from fit_integrate import integrate_from_model

a = 0
b = np.pi/2


# @vectorize([float64(float64)])
def J1x_nb(x):
    
    return np.where(x != 0, j1(x)/x, 0.5)

def integrand_2param(
    x, A, B):

    return (np.sinc(A * np.cos(x)/np.pi) * J1x_nb(B * np.sin(x)))**2*np.sin(x)

n = 6
reg_params2 = [np.geomspace(1, 1000000, n), np.geomspace(1, 1000000, n)]
params_names = ["A", "B"]
fit_model("cylinder_accurate", integrand_2param, a, b, params_names, reg_params2, 1e-4, update=True)

# integrate_from_model("cylinder_small_model", integrand_2param, a, b, (1e-5, 1000000, 100000))