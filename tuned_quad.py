import numpy as np
from numba import float64, njit  # import the types
from collections import namedtuple
from numba import types, vectorize
from numba.typed import Dict
import numpy.typing as npt
import typing as tp
import numba
from kronrod import kronrod_points_dict
from fixed_quad import FixedQuad, fixed_quad_init, fixed_quad_integrate
from kronrod import kronrod_points_dict

ParametersDictType = (types.unicode_type, types.float64)
ParametersType = types.DictType(*ParametersDictType)

ParametersDictType = (types.unicode_type, types.float64)
ParametersType = types.DictType(*ParametersDictType)

RegisteredParametersDictType = (types.unicode_type, types.float64[:])
RegisteredParametersType = types.DictType(*RegisteredParametersDictType)

TunedMatType = types.int32[:]
DimsType = types.int32[:]

KronrodPointsDictType = (types.int32, types.UniTuple(types.float64[:], 3))
KronrodPointsType = types.DictType(*KronrodPointsDictType)

"""
Types specifed below may not exactly the ones used. They are just placeholders to show the structure of the types used in the functions.
They would be however if the function were not compiled with numba and simply run in python. 

"""
# print(numba.typeof())

# QuadCacheDictType = (types.int32, FixedQuadType)
# QuadCacheType = types.DictType(*QuadCacheDictType)
# exit()
# TunedQuadSpec = [
#     ('reg_params', RegisteredParametersType),
#     ('tuned_mat', TunedMatType),
#     ('quad_cache', QuadCacheType),
#     ('dims', DimsType)
# ]

TunedQuad = tp.NamedTuple("TunedQuad", [("reg_params", tp.Dict[str, np.float64]), ("tuned_mat", npt.NDArray[np.int32]),
                                        ("quad_cache", tp.Dict[int, FixedQuad]), ("dims", np.ndarray)])


@njit
def tuned_quad_init(
        reg_params: tp.Dict[str, np.float64],
        tuned_mat: npt.NDArray[np.int32],
        kronrod_points_dict) -> TunedQuad:
    """
    Create a TunedQuad object that will be used to integrate functions with tuned quadrature points.
    Numba needs dictionary types to be explicitly typed. 

    To simplify the process, _DictType is used to the required dictionary.
    For instance, to create a dictionary for the registered parameters, import `RegisteredParametersDictType` from the current directory,
    then use `numba.typed.Dict(*RegisteredParametersDictType)` to create the dictionary and fill it as normal.

    Parameters
    ----------
    reg_params : `RegisteredParametersType` -> `numba.typed.Dict(*RegisteredParametersDictType)` -> `numba.typed.Dict(unicode_type, float64)`
        A dictionary containing the parameters used for tuning the quadrature. These will be used to select the number of Kronrod points from the tuned matrix.
    
    tuned_mat : `NDArray[np.int32]`
        This is a flattened n-dimensional matrix that contains the number of Kronrod points for each combination of parameters.
        It is flattened because numba needs to know the arrays dimensions at compile time.
    
    Returns
    -------
    `TunedQuad`
    
    """

    quad_cache = {}
    for n in kronrod_points_dict:
        integrator = fixed_quad_init(n, kronrod_points_dict)
        quad_cache[n] = integrator
    # Get the dimensions of the tuned matrix
    dims = np.empty(len(reg_params), dtype=np.int32)
    for i, value in enumerate(reg_params.values()):
        dims[i] = len(value)
    return TunedQuad(reg_params, tuned_mat, quad_cache, dims)


@njit
def tuned_quad_ravel_multi_index(
        tuned_quad: TunedQuad,
        multi_index: npt.NDArray[np.int32]) -> int:
    """
        Converts a multidimensional index to a flat index. It assumes C (row-major) order.

        Parameters
        ----------
        tuned_quad : `TunedQuad`
            The TunedQuad object

        multi_index : `NDArray[int32]`
            The multidimensional index

        Returns
        -------
        `int`

        """

    strides = np.cumprod(np.concatenate((np.array([1], dtype=np.int32), tuned_quad.dims[::-1][:-1].astype(np.int32))))[
              ::-1]
    return (multi_index * strides).sum()


@njit
def tuned_quad_get_n_kronrod(
        tuned_quad: TunedQuad,
        params: tp.Dict[str, np.float64]) -> int:
    """
    Selects the number of Kronrod points based on the parameters as defined by the tuned matrix.

    Parameters
    ----------
    tuned_quad : `TunedQuad`
        The TunedQuad object

    params : `ParametersType -> numba.typed.Dict(*ParametersDictType) -> numba.typed.Dict(unicode_type, float64)`
        The parameters to be used to select the number of Kronrod points

    Returns
    -------
    `int`

    """

    index = np.empty(len(tuned_quad.reg_params), dtype=np.int32)

    # Binary search for the index of the parameter for each parameter. This corresponds to each dimension of the tuned matrix
    for k, key in enumerate(tuned_quad.reg_params):
        i = np.searchsorted(tuned_quad.reg_params[key], params[key])

        # clamp the values to the maximum index
        i = min(i, len(tuned_quad.reg_params[key]) - 1)
        index[k] = i

    # Convert the multidimensional index to a flat index
    loc = tuned_quad_ravel_multi_index(tuned_quad, index)

    if loc >= len(tuned_quad.tuned_mat):
        raise ValueError("Index out of bounds")

    return tuned_quad.tuned_mat[loc]


@njit
def tuned_quad_integrate(
        tuned_quad: TunedQuad,
        func: tp.Callable[
            [tp.Union[np.float64, npt.NDArray[np.float64]], np.float64, np.float64, tp.Dict[str, np.float64]], tp.Union[
                np.float64, npt.NDArray[np.float64]]],
        a: np.float64,
        b: np.float64,
        params: tp.Dict[str, np.float64]) -> tp.Union[np.float64]:
    """
    Integrate a function using the tuned quadrature points. The function must be vectorized. Furthermore, since it uses a fixed quadrature, the function cannot have singularities.
    
    Parameters
    ----------
    tuned_quad : `TunedQuad`
        The TunedQuad object

    func : `Callable`
        Function to be integrated
        
    a : `float64`
        The lower bound of the integral
    
    b : `float64`
        The upper bound of the integral
    
    params : `Dict[str, float64]`
    
    Returns
    -------
    `float64`

    """

    tuned_quad_check_params(tuned_quad, params)
    n = tuned_quad_get_n_kronrod(tuned_quad, params)
    # if n in tuned_quad.quad_cache:
    integrator = tuned_quad.quad_cache[n]
    # else:
    #     integrator = fixed_quad_init(n, kronrod_points_dict)
    #     tuned_quad.quad_cache[n] = integrator

    return fixed_quad_integrate(integrator, func, a, b, params)


@njit
def tuned_quad_check_params(
        tuned_quad: TunedQuad,
        params: tp.Dict[str, np.float64]
):
    """
    Check if the parameters are registered in the TunedQuad object

    Parameters
    ----------

    tuned_quad : `TunedQuad`
        The TunedQuad object

    params : `Dict`
        A dictionary containing the parameters to be checked

    Raises
    ------
    ValueError
        If a parameter is not registered in the TunedQuad object
    """

    for key in params:
        if key not in tuned_quad.reg_params:
            raise ValueError(f"Parameter {key} is not registered")


if __name__ == "__main__":
    from j1 import j1


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


    reg_params2 = Dict.empty(*RegisteredParametersDictType)
    reg_params2['A'] = np.geomspace(1, 1000000, 15)
    reg_params2['B'] = np.geomspace(1, 1000000, 15)

    tuned_mat = np.array(
        [10, 12, 19, 35, 72, 166, 403, 924, 1800, 2750, 3350, 3850, 6050, 11050, 16300, 11, 15, 23, 37, 72, 164, 396,
         904, 1750, 2700, 3450, 4200, 6350, 11150, 16300, 17, 21, 27, 40, 73, 162, 386, 870, 1700, 2750, 3650, 4600,
         6850, 11450, 16350, 34, 35, 39, 49, 82, 170, 389, 853, 1700, 2800, 3850, 4850, 7200, 11650, 16400, 71, 71, 73,
         82, 111, 194, 401, 827, 1550, 2550, 3500, 4300, 6650, 11400, 16350, 163, 161, 162, 168, 191, 256, 423, 770,
         1350, 2200, 3100, 3750, 6050, 11100, 16300, 374, 361, 353, 359, 384, 427, 525, 779, 1250, 2000, 2950, 3850,
         6200, 11100, 16300, 802, 749, 696, 714, 806, 867, 853, 955, 1300, 2000, 3050, 4400, 6800, 11400, 16350, 1500,
         1400, 1300, 1350, 1600, 1750, 1550, 1350, 1550, 2050, 3050, 4750, 7250, 11600, 16400, 2200, 2150, 2050, 2150,
         2600, 2900, 2400, 1900, 1900, 2200, 3100, 4750, 7150, 11400, 16300, 2900, 2950, 3000, 3050, 3450, 3750, 3350,
         2900, 2800, 2900, 3600, 5000, 7050, 10950, 15650, 4500, 4650, 4950, 5050, 5000, 4950, 4650, 4450, 4550, 4650,
         5100, 6100, 7450, 10000, 13450, 6950, 7250, 7800, 8100, 7650, 7200, 6700, 6400, 6700, 7100, 7350, 7850, 8300,
         9100, 10650, 8850, 9100, 9700, 10200, 10050, 9750, 9050, 8500, 8700, 9050, 9150, 9200, 9200, 9600, 10650, 9900,
         10000, 10350, 10800, 11500, 12000, 11050, 10100, 9950, 10050, 10000, 9750, 9550, 10550, 12850])
    tuned_quad = tuned_quad_init(reg_params2, tuned_mat, kronrod_points_dict)
    params = Dict.empty(*ParametersDictType)
    params['A'] = 10000
    params['B'] = 10000

    integral = tuned_quad_integrate(tuned_quad, integrand_2param, 0, np.pi / 2, params)

    print(integral)
