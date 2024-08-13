"""
Test the accuracy of Integrate[F(q, alpha)^2 * Sin[alpha], {x, 0, Pi/2}] function for the cylinder model.

F(q, alpha) for the cylinder model has the form of C * J1[B * Sin[alpha]] / B * Sin[alpha]  * Sinc[A * Cos[alpha]] where A = q * length / 2 and B = q * radius.
where C is a constant that depends on the sld, sld_solvent, radius, and length of the cylinder but has been set to 1 for this test.

The integral is calculated using the tuned_quad_integrate function.

To test the accuracy of the integral, we set one of the parameters to 0 and vary the other as 10, 100, ..., 1000000.

When A is 0, the solution to the integral is -(C^2 (-1 + Hypergeometric0F1Regularized[2, -B^2])))/(2 B^2)
When B is 0, the solution to the integral is (0.25 C^2 (-1. Sin[A]^2 + A SinIntegral[2 A]))/A^2
"""

import numpy as np
import scipy.special as sp
from numba import vectorize, njit, float64
from numba.typed import Dict
from j1 import j1
import pandas as pd
from scipy.integrate import fixed_quad
from fit_integrate import integrate_from_model

a = 0
b = np.pi / 2


def integral_a0(B, C=1):
    term = -C ** 2 * (-1 + sp.hyp0f1(2, -B ** 2))
    result = term / (2 * B ** 2)
    return result


def integral_b0(A, C=1):
    sin_term = np.sin(A) ** 2
    si, ci = sp.sici(2 * A)
    term = 0.25 * C ** 2 * (-1 * sin_term + A * si)
    result = term / A ** 2
    return result


@vectorize([float64(float64)])
def J1x_nb(x):
    return j1(x) / x if x != 0 else 0.5


@njit
def integrand_2param(
        x,
        params):
    A = params['A']
    B = params['B']

    return (np.sinc(A * np.cos(x) / np.pi) * J1x_nb(B * np.sin(x))) ** 2 * np.sin(x)

def integrand_2param_py(
        x, A, B
):
    return (np.sinc(A * np.cos(x) / np.pi) * J1x_nb(B * np.sin(x))) ** 2 * np.sin(x)

A = 0
B = 10

result_dict = {}

for A in [0]:
    for B in [10, 100, 1000, 10000, 100000, 1000000]:

        result_numerical = integrate_from_model('cylinder_accurate_model', integrand_2param_py, a, b, (A, B))
        result_numerical_quad_76 = fixed_quad(integrand_2param_py, a, b, args=(A, B), n=76)[0]
        result_analytical = integral_a0(B)
        result_dict[(A, B)] = (
            result_numerical, result_analytical, abs((result_numerical - result_analytical) / result_analytical), result_numerical_quad_76, abs((result_numerical_quad_76 - result_analytical) / result_analytical))
        print(
            f"Result for A={A}, B={B}: Numerical: {result_numerical}, Analytical: {result_analytical}, Actual Relative Error: {abs((result_numerical - result_analytical) / result_analytical)}")

for B in [0]:
    for A in [10, 100, 1000, 10000, 100000, 1000000]:


        result_numerical = integrate_from_model('cylinder_accurate_model', integrand_2param_py, a, b, (A, B))
        result_numerical_quad_76 = fixed_quad(integrand_2param_py, a, b, args=(A, B), n=76)[0]
        result_analytical = integral_b0(A)

        result_dict[(A, B)] = (
            result_numerical, result_analytical, abs((result_numerical - result_analytical) / result_analytical), result_numerical_quad_76, abs((result_numerical_quad_76 - result_analytical) / result_analytical))
        print(
            f"Result for A={A}, B={B}: Numerical: {result_numerical}, Analytical: {result_analytical}, Actual Relative Error: {abs((result_numerical - result_analytical) / result_analytical)}")

index = pd.MultiIndex.from_tuples(result_dict.keys(), names=["A", "B"])
df = pd.DataFrame(result_dict.values(), index=index)
df.columns = ["Numerical", "Analytical", "True Relative Error", "Numerical Quad 76", "True Relative Error Quad 76"]

df.to_excel("cylinder_accurate_test.xlsx")
