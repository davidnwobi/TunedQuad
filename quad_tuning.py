import numpy as np
import numpy.typing as npt
from itertools import product
from fixed_quad import FixedQuad
from tuned_quad import ParametersDictType, KronrodPointsDictType, RegisteredParametersDictType, TunedQuad
from numba import types, prange
from kronrod import kronrod_points_dict 
from numba.typed import Dict
import numba
import typing as tp
from numba import njit, typeof
from tqdm import tqdm
import pickle
from tuned_quad import TunedQuad, RegisteredParametersDictType, ParametersDictType, tuned_quad_init
from fixed_quad import fixed_quad_init, fixed_quad_kronrod_init, fixed_quad_integrate
import h5py
from datetime import datetime
from scipy.ndimage import gaussian_filter

"""
Types specifed below are not exactly the ones used. They are just placeholders to show the structure of the types used in the functions.
They would be however if the function were not compiled with numba and simply run in python. The actual types will be mentioned in the docstrings.
"""

tuned_quad_model_loc = "tuned_quad.h5" 

@njit
def rel_error_kronrod(
    n: int,
    func: tp.Callable[[np.float64, tp.Dict[str, np.float64]], np.float64],
    a: np.float64,
    b: np.float64,
    params: tp.Dict[str, np.float64],
    kronrod_points_dict: tp.Dict[int, tp.Tuple[np.ndarray, np.ndarray, np.ndarray]])->np.float64:

    """
    Compute the relative error between using the Gauss-Kronrod Rule

    Parameters
    ----------

    n : `int`
        The number of quadrature points

    func : `Callable`
        The function to be integrated

    a : `float64`
        The lower bound of the integral

    b : `float64`
        The upper bound of the integral

    params : `ParametersType -> numba.typed.Dict(*ParametersDictType) -> numba.typed.Dict(unicode_type, float64)`
        The parameters of the function

    kronrod_points_dict : `Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]]`
        The Kronrod points. This is a global variable loaded in when `kronrod.py` is imported. 
        It must be passed in as an argument to the function to prevent numba from trying to compile it.

    """

    integrator_h = fixed_quad_kronrod_init(n, kronrod_points_dict)
    integrator_l = fixed_quad_init(n, kronrod_points_dict)
    sol_h = fixed_quad_integrate(integrator_h, func, a,b, params)
    sol_l = fixed_quad_integrate(integrator_l, func, a,b, params)
    return np.abs((sol_h-sol_l)/sol_h)

@njit(parallel=True)   
def compute_tuned_quad_dict(
    func: tp.Callable[[np.float64, tp.Dict[str, np.float64]], np.float64],
    a: int,
    b: int, 
    reg_params: tp.Dict[str, np.ndarray],
    param_prod: tp.List[tp.Tuple[np.float64]],
    rtol: np.float64,
    n_kronrod: npt.NDArray[np.int64],
    kronrod_points_dict : tp.Dict[int, tp.Tuple[np.ndarray, np.ndarray, np.ndarray]])->tp.Dict[tp.Tuple[np.float64], np.int64]:

    """
    
    Compute the tuned quadrature points for the given function and parameter space.
    
    Parameters
    ----------
    
    func : `Callable`
        The function to be integrated

    a : `float64`
        The lower bound of the integral

    b : `float64`
        The upper bound of the integral

    reg_params : `RegisteredParametersType` -> `numba.typed.Dict(*RegisteredParametersDictType)` -> `numba.typed.Dict(unicode_type, float64)`
        The registered parameters for the function. They are a range of values for each parameter. Preferably, they should be sorted for the problem at hand.

    param_prod : `List[Tuple[np.float64]]`
        The product of the registered parameters. This is a list of tuples where each tuple is a parameter combination.
    
    """

    # NOTE: Alternatively you could have used mesh grid to get the parameter combinations.
    # then a matrix of the same shape as the mesh grid would be created to store the tuned quadrature points.
    # But this is easier to understand would prevent need to fill a matrix later but would be very inefficient for large parameter spaces.

    
    tuned_quad = dict() # Specify the type for this
    out = np.zeros(len(param_prod), dtype=np.int64)
    for k in prange(len(param_prod)):
        param = param_prod[k]
        params = Dict.empty(*ParametersDictType)


        # Assign the current parameter combination in value to the parameters dictionary        
        for j, key in enumerate(reg_params.keys()):
            params[key] = float(param[j])

        rel_err = 0.0
        # Find the minimum n such that the relative error is less than the tolerance
        for i, n in enumerate(n_kronrod):
            rel_err = rel_error_kronrod(n, func, a,b, params, kronrod_points_dict)
            # rel_err = err_model(n, func, a, b, params, kronrod_points_dict[n], kronrod_points_dict[19950][0], kronrod_points_dict[19950][1])
            
            if rel_err < rtol:
                n = i
                out[k] = n
                break
        else:
            print("Kronrod failed for Param", param, " with error ", rel_err)
            print("n has been set to the maximum value")
            n = len(n_kronrod)-1
            out[k] = n

        # Store the tuned quadrature points in the tuned_quad dictionary 
        print("Relative error for Param", param, " is ", rel_err)
        
    for k, param in enumerate(param_prod):
        tuned_quad[param] = n_kronrod[out[k]]

    return tuned_quad

def tune_quadrature(
        func: tp.Callable[[np.float64, tp.Dict[str, np.float64]], np.float64],
        a: int,
        b: int,
        reg_params: tp.Dict[str, np.ndarray], 
        tol: np.float64,
        n_kronrod: npt.NDArray[np.int64]) -> tp.Dict[tp.Tuple[np.float64], np.int64]:
    
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
    
    param_prod = list(product(*reg_params.values()))

    return compute_tuned_quad_dict(
        func=func, 
        a=a, 
        b=b, 
        reg_params=reg_params, 
        param_prod=param_prod, 
        rtol=tol,
        n_kronrod=n_kronrod, 
        kronrod_points_dict=kronrod_points_dict)


def fill_tuned_matrix(
        index: tp.List[int], 
        k: int,
        tuned_mat: npt.NDArray[np.int64], 
        tuned_dict: tp.Dict[tp.Tuple[np.float64], np.int64],
        reg_params: tp.Dict[str, np.ndarray],
        dims: tp.List[int]):
    
    """

    Converts the tuned dictionary to a n-dimensional matrix. 
    This is a recursive function that fills the matrix with the tuned quadrature points.

    The matrix is n dimensional where n is the number of registered parameters and, 
    Each dimension corresponds to the number of values that have been registered for that parameter.

    Each index combination is iterated over and the corresponding tuned quadrature point is stored in the matrix.

    NOTE: Replace recursion with np.ndindex. Cleaner and more intuitive.


    Parameters 
    ----------

    index : `List[int]`
        The current index of the matrix. This will eventually cycle through all the values of the matrix.

    k : `int`
        The current dimension of the matrix. This is used to keep track of the current dimension the function is on.

    tuned_mat : `NDArray[int64]`
        The matrix to store the tuned quadrature points. N-dimensional matrix where N is the number of registered parameters.
    
    tuned_dict : `Dict[Tuple[float64], int64]`
        The dictionary of tuned quadrature points. The keys are the parameter combinations and the values are the tuned quadrature points.

    reg_params : `RegisteredParametersType` -> `numba.typed.Dict(*RegisteredParametersDictType)` -> `numba.typed.Dict(unicode_type, float64)`
        The registered parameters for the function. They are a range of values for each parameter. Preferably, they should be sorted for the problem at hand.

    dims : `List[int]`
        The dimensions of the matrix. This is a list of the number of values for each parameter.
    """


    if k == len(dims):
        key = []
        for i, value in zip(index, reg_params.values()):
            key.append(value[i])

        tuned_mat[tuple(index)] = tuned_dict[tuple(key)]
        return
    
    for i in range(dims[k]):# This recurive function can be replaced with np.ndindex
        index[k] = i
        fill_tuned_matrix(index, k+1, tuned_mat, tuned_dict, reg_params, dims)

def make_tuned_matrix(
        tuned_quad_dict: tp.Dict[tp.Tuple[np.float64], np.int64],
        reg_params: tp.Dict[str, np.ndarray]) -> npt.NDArray[np.int64]:
    
    """
    Transforms the tuned quadrature dictionary to a n-dimensional matrix of tuned quadrature points.

    Parameters
    ----------

    tuned_quad_dict : `Dict[Tuple[float64], int64]`
        The dictionary of tuned quadrature points. The keys are the parameter combinations and the values are the tuned quadrature points.

    reg_params : `RegisteredParametersType` -> `numba.typed.Dict(*RegisteredParametersDictType)` -> `numba.typed.Dict(unicode_type, float64)`
        The registered parameters for the function. They are a range of values for each parameter. Preferably, they should be sorted for the problem at hand.

    Returns
    -------
    `NDArray[int64]`

    """

    # Get the dimensions of the registered parameters
    dims = []
    for _, v in reg_params.items():
        dims.append(len(v))

    # Create a n dimensional matrix to store the tuned quadrature points
    tuned_matrix = np.zeros(dims, dtype=int)
    index_ = [0]*len(dims)

        
    fill_tuned_matrix(index_, 0, tuned_matrix, tuned_quad_dict, reg_params, dims)
    return tuned_matrix

    # Above done once for a model problem. The tuned matrix and the parameter space are saved in a file. 

def save_tuned_quad_h5(
        tuned_quad: TunedQuad, 
        model_name: str):
    """
    Save the tuned quadrature object to an h5 file. Enough data is saved to recreate the tuned quadrature object, i.e. the tuned matrix and the registered parameters.

    Parameters
    ----------

    tuned_quad : `TunedQuad`
        The tuned quadrature object

    model_name : `str`
        The name of the model. This is used to store the tuned quadrature object in the h5 file.


    """
    with h5py.File(tuned_quad_model_loc, "a") as file:
        if model_name in file:
            del file[model_name]
        model = file.create_group(model_name)
        model.attrs['COMMENT'] = f"Tuned quadrature properties for {model_name}"
        model.attrs['Created_on'] = datetime.now().strftime("%Y-%m-%d")

        reg_params = model.create_group("reg_params")
        model.create_dataset("tuned_mat", data=tuned_quad.tuned_mat, dtype=np.int32)
        
        for k, v in tuned_quad.reg_params.items():
            reg_params.create_dataset(k, data=v, dtype=np.float64)



def load_tuned_quad_h5(
        model_name: str
        ) -> TunedQuad:
    
    """
    Load the tuned quadrature object propeties from a file and construct the tuned quadrature object.

    Parameters
    ----------

    model_name : `str`
        The name of the model. This is used to retrieve the tuned quadrature object from the h5 file.

    Returns
    -------
    `TunedQuad`

    """

    with h5py.File(tuned_quad_model_loc, "r") as f:
        if model_name not in f:
            raise FileNotFoundError(f"{model_name} not found.")  # Check if the model exists
        
        file = f[model_name]
        reg_params = Dict.empty(*RegisteredParametersDictType) # in a sane world, this would be just dict()
        for k in file["reg_params"]:
            reg_params[k] = np.array(file["reg_params"][k][:]) 
        tuned_mat = np.array(file["tuned_mat"][:])
        
        tuned_quad = tuned_quad_init(reg_params, tuned_mat)

    return tuned_quad     
      
def smooth_matrix(
        tuned_quad_matrix: npt.NDArray[np.int64], 
        filter_sigma: float = 1.0) -> npt.NDArray[np.int64]:
    
    """
    Smooth the tuned quadrature matrix via a gaussian filter. This is done to avoid sharp changes in the number of Kronrod points used to integrate the function across the parameter space.

    Parameters
    ----------

    tuned_quad_matrix : `NDArray[int64]`
        The tuned quadrature matrix. This is a n-dimensional matrix where N is the number of registered parameters.

    filter_sigma : `float`
        The standard deviation of the gaussian filter. This controls the amount of smoothing applied to the matrix.
    """


    # NOTE: This can be done with a convolution. 
    # The idea is to smooth the matrix to reduce the number of kronrod points needed to achieve a certain tolerance
    # The smoothing is done by averaging the number of kronrod points in the neighbourhood of a point.
    # This is done by convolving the matrix with a kernel. The kernel is a 3x3 matrix with 1s in all the elements.
    # The matrix is then divided by the sum of the kernel to get the average number of kronrod points in the neighbourhood.
    # The matrix is then thresholded to get the final matrix. 
    # The threshold is the maximum number of kronrod points that can be used to integrate the function.
    # The threshold is set to the maximum number of kronrod points in the matrix.
    # The threshold is applied to the matrix to get the final matrix. 
    # The final matrix is then used to create the tuned quadrature object.

    
    tuned_quad_matrix = gaussian_filter(tuned_quad_matrix, sigma=filter_sigma, mode='nearest')
    # for j in range(1, z_smooth.shape[1]):
    #     z_smooth[:, j] = np.maximum(z_smooth[:, j], z_smooth[:, j-1])
    # for i in range(1, z_smooth.shape[0]):
    #     z_smooth[i, :] = np.maximum(z_smooth[i, :], z_smooth[i-1, :])
    # zi = z_smooth


    # Replace the interpolated values with the nearest integer Kronrod points that are available in the dictionary. 
    # Smoothing may result in interpolated values that are not in the dictionary.
    # That's all that's happening below
        
    all_kron_p = sorted(kronrod_points_dict.keys())

    prev = np.nan
    prev_nearest = np.nan
    for i in range(tuned_quad_matrix.shape[0]):
        for j in range(tuned_quad_matrix.shape[1]):
            target = tuned_quad_matrix[i, j]

            if target == prev:
                tuned_quad_matrix[i, j] = prev_nearest
                continue

            nearest_greater = np.searchsorted(all_kron_p, target)
            if nearest_greater == len(all_kron_p):
                nearest_greater -= 1

            prev = target
            prev_nearest = all_kron_p[nearest_greater]

            tuned_quad_matrix[i, j] = all_kron_p[nearest_greater]

    return tuned_quad_matrix


def tune(
        model_name: str, 
        integrand: tp.Callable[[np.float64, tp.Dict[str, np.float64]], np.float64],
        a: int,
        b: int,
        registered_parameters: tp.Dict[str, np.ndarray], 
        tol: np.float64,
        n_kronrod: npt.NDArray[np.int64],
        update=False):
    
    """
    Obtain the quadrature points needed to integrate the function across the parameter space.

    Parameters
    ----------

    model_name : `str`
        The name of the model. This is used to store the tuned quadrature object in the h5 file.

    integrand : `Callable`
        The function to be integrated
        
    a : `float64`
        The lower bound of the integral

    b : `float64`
        The upper bound of the integral

    registered_parameters : `RegisteredParametersType` -> `numba.typed.Dict(*RegisteredParametersDictType)` -> `numba.typed.Dict(unicode_type, float64)`
        The registered parameters for the function. They are a range of values for each parameter. Preferably, they should be sorted for the problem at hand.

    tol : `float64`
        The tolerance for the relative error between the Gauss-Kronrod Rule
    
    n_kronrod : `NDArray[int64]`
        List of Kronrod points to be used to check when tuning the quadrature points.

    update : `bool`
        If True, the tuned quadrature points are recomputed. If False, the tuned quadrature points are loaded from the h5 file or computed if they do not exist.

    Returns
    -------

    `TunedQuad`

    """



    # print("Tuning quadrature for ", model_name)
    file = h5py.File(tuned_quad_model_loc, "a")
    if not model_name in file or update:
        if model_name in file:
            del file[model_name]

        print(f"Computing tuned quadrature for {model_name}")
        tuned_quad_dict = tune_quadrature(integrand, a, b, registered_parameters, tol, n_kronrod=n_kronrod)
        tuned_quad_matrix = make_tuned_matrix(tuned_quad_dict, registered_parameters)
        print(f"Tuned matrix for {model_name}, unsmoothed")
        print(tuned_quad_matrix)

        tuned_quad_matrix = smooth_matrix(tuned_quad_matrix)

        print(f"Tuned matrix for {model_name}")
        print(tuned_quad_matrix)

        tuned_quad_matrix = tuned_quad_matrix.flatten()

        tuned_quad = tuned_quad_init(registered_parameters, tuned_quad_matrix)
        file.close()

        save_tuned_quad_h5(tuned_quad, model_name)
        return tuned_quad  
    file.close()
    return load_tuned_quad_h5(model_name)