import numpy as np
from numba import float64, njit    # import the types
from collections import namedtuple
from numba import types
from numba.typed import Dict
import numpy.typing as npt
import typing as tp
import numba
from kronrod import kronrod_points_dict
from numba.extending import typeof_impl, type_callable, register_model, make_attribute_wrapper, models
from numba.core import types

from numba import typeof

ParametersDictType = (types.unicode_type, types.float64)
ParametersType = types.DictType(*ParametersDictType)

# Ideal case but numba prefers structs
FixedQuad = tp.NamedTuple("FixedQuad", [("points", npt.NDArray[np.float64]), ("weights", npt.NDArray[np.float64])])

# ideal case but numba is dumb and will make `kronrod_points` a compile time constant. 🤮
# def fixed_quad_init(n):
#     if n in kronrod_points:
#         return FixedQuad(kronrod_points[n][0], kronrod_points[n][1])
#     else:
#         raise ValueError(f"n = {n} not found in Kronrod points.")

# ideal case but numba is dumb. It won't accept dict as a parameter.
# @njit
# def fixed_quad_integrate(
#         fixed_quad: FixedQuad,             
#         func: tp.Callable[[npt.NDArray[np.float64], np.float64, np.float64, tp.Dict[str, np.float64]],  npt.NDArray[np.float64]],
#         a: np.float64,
#         b: np.float64,
#         params: tp.Dict[str, np.float64] = dict())->tp.Union[np.float64]:
    
#     y = (b-a)*(fixed_quad.points+1)/2.0 + a
#     return (b-a)/2.0 * np.sum(fixed_quad.weights*func(y, params), axis=-1)

@njit
def fixed_quad_init(n, kronrod_points):
    """
    Initialize the fixed quadrature points and weights for a given n using the Legendre points. 
    These are nested within the Kronrod points.

    Parameters
    ----------
    n : `int`
        The number of quadrature points.
    kronrod_points : `DictType(int32, UniTuple(float64[:], 3))`
        The Kronrod points.

    Returns
    -------
    `FixedQuad`

    Raises
    ------
    `ValueError` if n is not found in Kronrod points.

    """
    # kronrod_points = kronrod_points_dict
    if n in kronrod_points:
        x =  kronrod_points[n][0][1:-1:2].copy().astype(np.float64) # numba requires contiguous arrays
        w =  kronrod_points[n][2][1:-1:2].copy().astype(np.float64)
        return FixedQuad(x, w)
    else:
        raise ValueError(f"n = {n} not found in Kronrod points.")
@njit 
def fixed_quad_kronrod_init(n, kronrod_points):
    """
    Initialize the fixed quadrature points and weights for a given n using the Kronrod points.

    Parameters
    ----------
    n : `int`
        The number of quadrature points.
    kronrod_points : `DictType(int32, UniTuple(float64[:], 3))`
        The Kronrod points.

    Returns
    -------
    `FixedQuad`

    Raises
    ------
    `ValueError` if n is not found in Kronrod points.

    """
    if n in kronrod_points:
        x = kronrod_points[n][0].copy().astype(np.float64)
        w = kronrod_points[n][1].copy().astype(np.float64)
        return FixedQuad(x, w)
    else:
        raise ValueError(f"n = {n} not found in Kronrod points.")
@njit
def fixed_quad_integrate(
        fixed_quad: FixedQuad,             
        func: tp.Callable[[npt.NDArray[np.float64], np.float64, np.float64, tp.Dict[str, np.float64]],  npt.NDArray[np.float64]],
        a: np.float64,
        b: np.float64,
        params: tp.Dict[str, np.float64] = dict())->tp.Union[np.float64]:
    
    """
    Integrate a function using fixed quadrature.

    Parameters
    ----------

    fixed_quad : `FixedQuad`
        The fixed quadrature points and weights.

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
    
    y = (b-a)*(fixed_quad.points+1)/2.0 + a
    return (b-a)/2.0 * np.sum(fixed_quad.weights*func(y, params), axis=-1)


fixed_quad_h = fixed_quad_kronrod_init(4, kronrod_points_dict)
if __name__ == "__main__":
    from scipy.special import roots_legendre

    n = 5
    fixed_quad_h = fixed_quad_kronrod_init(4, kronrod_points_dict)
    fixed_quad_l = fixed_quad_init(4, kronrod_points_dict)

    @njit
    def f(x, params):
        return x**7
    
    sol_h = fixed_quad_integrate(fixed_quad_h, f, 0, 1, Dict.empty(*ParametersDictType))
    sol_l = fixed_quad_integrate(fixed_quad_l, f, 0, 1, Dict.empty(*ParametersDictType))
    print(sol_h, sol_l)
    print(np.abs((sol_h-sol_l)/sol_h))  