import numpy as np
from eval_poly import eval_poly_2_2
from tuned_quad_new import eval_poly
from math import comb

def binary_search(arr, target):
    l = 0
    r = len(arr)-1
    while l < r:
        m = (l+r)//2
        if arr[m] < target:
            l = m+1
        else:
            r = m
    return l

def direct_2d_poly_eval(x, y, coeffs, intercept):

    return 1 * coeffs[0] + x * coeffs[1] + y * coeffs[2] + x**2 * coeffs[3] + x * y * coeffs[4] + y**2 * coeffs[5] + intercept  

rng = np.random.default_rng(42)
param_space_size = 15


n_params = 3
degree = n_params+1
coeff_size = comb(degree+n_params, degree)

param_space = np.geomspace(1, 1000000, param_space_size)
coeffs = rng.random(coeff_size)
params = rng.random(n_params)
random_ndmatrix = rng.random((param_space_size,)*n_params)


def search_bench():    
    return random_ndmatrix[tuple(binary_search(param_space, 1000000) for _ in range(n_params))]

def eval_bench():
    return eval_poly(params, degree, coeffs, 1)

def eval_direct_bench():
    return direct_2d_poly_eval(params[0], params[1], coeffs, 1)

def eval_gen_poly():
    return eval_poly_2_2(params, coeffs,1)

print(eval_bench())
print(eval_direct_bench())
print(eval_gen_poly())


import timeit

print("Search_bench")
print(timeit.timeit("search_bench()", globals=globals(), number=100000))
print("Eval_bench")
print(timeit.timeit("eval_bench()", globals=globals(), number=100000))
print("Eval_direct_bench")
print(timeit.timeit("eval_direct_bench()", globals=globals(), number=100000))
print("Eval_gen_poly")
print(timeit.timeit("eval_gen_poly()", globals=globals(), number=100000))





