import numpy as np
import numpy.typing as npt
from itertools import product
from kron import get_kronrod_w_higher_order
from tuned_quad_new import _integrate
import typing as tp
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
import typing as tp
import numpy as np
import os

"""
Types specifed below are not exactly the ones used. They are just placeholders to show the structure of the types used in the functions.
They would be however if the function were not compiled with numba and simply run in python. The actual types will be mentioned in the docstrings.
"""


n_kronrod = np.power(2, np.arange(1, 10))

def rel_error_kronrod(
    n: int,
    func: tp.Callable[[np.float64, tp.Dict[str, np.float64]], np.float64],
    a: np.float64,
    b: np.float64,
    params: tp.Tuple[np.float64])->np.float64:

    xg, wg, xk, wk = get_kronrod_w_higher_order(n)

    sol_h = _integrate(xk, wk, func, a, b, params)
    sol_l = _integrate(xg, wg, func, a, b, params)
    return np.abs((sol_h-sol_l)/sol_h)

def compute_tuned_quad_dict(
    func: tp.Callable[[np.float64, tp.Dict[str, np.float64]], np.float64],
    a: int,
    b: int, 
    param_prod: tp.List[tp.Tuple[np.float64]],
    rtol: np.float64)->tp.Dict[tp.Tuple[np.float64], np.int64]:



    # NOTE: Alternatively you could have used mesh grid to get the parameter combinations.
    # then a matrix of the same shape as the mesh grid would be created to store the tuned quadrature points.
    # But this is easier to understand would prevent need to fill a matrix later but would be very inefficient for large parameter spaces.

    
    tuned_quad = dict() # Specify the type for this
    out = np.zeros(len(param_prod), dtype=np.int64)
    for k in range(len(param_prod)):
        print(param_prod[k])
        rel_err = 0.0
        # Find the minimum n such that the relative error is less than the tolerance
        for i, n in enumerate(n_kronrod):
            rel_err = rel_error_kronrod(n, func, a,b, param_prod[k])
            
            if rel_err < rtol:
                n = i
                out[k] = n
                break
        else:
            print("Kronrod failed for Param", param_prod[k], " with error ", rel_err)
            print("n has been set to the maximum value")
            n = len(n_kronrod)-1
            out[k] = n

        # Store the tuned quadrature points in the tuned_quad dictionary 
        print("Relative error for Param", param_prod[k], " is ", rel_err)
        
    for k, param in enumerate(param_prod):
        tuned_quad[param] = n_kronrod[out[k]]

    return tuned_quad

def tune_quadrature(
        func: tp.Callable[[np.float64, tp.Dict[str, np.float64]], np.float64],
        a: int,
        b: int,
        reg_params_list: tp.List[npt.NDArray], 
        tol: np.float64) -> tp.Dict[tp.Tuple[np.float64, ...], np.int64]:
    
    """
    Tune the quadrature points for the given function and parameter space.

    Parameters
    ----------

    func : `Callable`
        The function to be integrated

    a : `float64`
        The lower bound of the integral

    b : `float64`
        The upper bound of the integral

    """
    
    param_prod = list(product(*reg_params_list))

    return compute_tuned_quad_dict(
        func=func, 
        a=a, 
        b=b, 
        param_prod=param_prod, 
        rtol=tol)


def save_model_py_file(model_name: str, n_params, poly_degree, coeffs, intercept):

    py_file_name = model_name + "_model.py"

    py_script = f"""

import numpy as np

__all__ = ["model_name", "n_params", "poly_degree", "coeffs", "intercept"]
model_name = "{model_name}"
n_params = {n_params}
poly_degree = {poly_degree}
coeffs = {np.array(np.array2string(coeffs, separator=",", precision=10))}
intercept  = {intercept}
"""

    with open(py_file_name, "w") as f:
        f.write(py_script)

def get_shift(pred, orig, threshold=.99):
    '''
    Get the shift needed to make the predicted values greater than the original values (threshold)% of the time.
    '''
    def percentage_greater(pred_shifted):
        return (pred_shifted >= orig).mean() > threshold
    
    l = 0
    r = pred.max()-orig.min() # unless there is some weird edge case, this should be positive and a sufficient starting point
    eps = 1e-6
    while(abs(l-r) > eps): # probably not the best way to do this but I know binary search works
        m = (l+r)/2
        if percentage_greater(pred+m):
            r = m
        else:
            l = m
    return l


def fit_dict(tuned_quad_dict: tp.Dict[tp.Tuple[np.float64, ...], int], degree, threshold=.99) -> tp.Tuple[int, int, np.ndarray, np.float64]:
    '''
    Fit a polynomial to the tuned quadrature dictionary.
    
    Parameters
    ----------
    tuned_quad_dict : `DictType(Tuple(int32, ...), int32)`
        The tuned matrix.
    
    Returns
    -------
    `Tuple(int, int, np.ndarray, np.float64)`
    '''
    n_params = len(list(tuned_quad_dict.keys())[0])
    if n_params > degree:
        raise ValueError("Number of variables is greater than polynomial degree")
    
    # Fit an n-dimennsional polynomial to the tuned matrix
    features = np.vstack(np.array([np.array(k) for k in tuned_quad_dict.keys()]))
    values = np.array(list(tuned_quad_dict.values()))

    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    model.fit(features, values)
    prediction = model.predict(features)

    # Get the shift needed to make the predicted values greater than the original values (threshold)% of the time.
    shift = get_shift(prediction, values, threshold)

    coeff = model.named_steps['linearregression'].coef_
    intercept = model.named_steps['linearregression'].intercept_+shift

    return n_params, degree, coeff, intercept


def fit_model(
        model_name: str, 
        integrand,
        a: int,
        b: int,
        reg_param_list: tp.List[np.ndarray], 
        tol: np.float64,
        degree: tp.Optional[int] = None,
        update=False):
    

    file_name = f'{model_name}_model.py'
    if not os.path.exists(file_name) or update:
        if degree is None:
            degree = len(reg_param_list)
        tuned_quad_dict = tune_quadrature(integrand, a, b, reg_param_list, tol)
        n_params, degree, coeffs, intercept = fit_dict(tuned_quad_dict, degree)
        save_model_py_file(model_name, n_params, degree, coeffs, intercept)
