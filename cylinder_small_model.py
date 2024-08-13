

import numpy as np
import typing as tp
from collections.abc import Sequence
from kron import get_gauss_points

__all__ = ["integrate"]
coeffs = [ 0.          , 0.6617372473, 0.6645981289, 0.0067881973,-0.0465574793,
  0.0072142637]
intercept  = 3.542896397504518


        
def eval_poly(vars: Sequence[float]) -> float:
    return 1 * coeffs[0] + vars[0] * coeffs[1] + vars[1] * coeffs[2] + vars[0] ** 2 * coeffs[3] + vars[0] * vars[1] * coeffs[4] + vars[1] ** 2 * coeffs[5] + intercept
    

lb: int = 1
ub: int = 15

def integrate(f: tp.Callable[[float, tp.Tuple[float, ...]], float], a: float, b: float, params: tp.Tuple[float, ...] = ()) -> float:

    expo = int(eval_poly([np.log2(param) for param in params]) + 1)   
    n = int(pow(2, max(lb, min(ub, expo))))

    xg, wg = get_gauss_points(n)

    y = (b-a)*(xg+1)/2 + a
    return (b-a)/2 * np.sum(wg*f(y, *params))
    