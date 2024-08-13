

import numpy as np
import typing as tp
from collections.abc import Sequence
from kron import get_gauss_points

__all__ = ["integrate"]
coeffs = [ 0.          ,-0.9148286559, 0.0561389246,-0.0999854514,-0.0515047232,
  0.0551857439,-0.0036181641,-0.0033739435,-0.001450217 ,-0.0021050108]
intercept  = -0.6157187001159432
limits = [(1e-05, 0.1), (1.0, 100000.0)]


        
def eval_poly(vars: Sequence[float]) -> float:
    return 1 * coeffs[0] + vars[0] * coeffs[1] + vars[1] * coeffs[2] + vars[0] ** 2 * coeffs[3] + vars[0] * vars[1] * coeffs[4] + vars[1] ** 2 * coeffs[5] + vars[0] ** 3 * coeffs[6] + vars[0] ** 2 * vars[1] * coeffs[7] + vars[0] * vars[1] ** 2 * coeffs[8] + vars[1] ** 3 * coeffs[9] + intercept
    

lb: int = 1
ub: int = 15

def integrate(f: tp.Callable[[float, tp.Tuple[float, ...]], float], a: float, b: float, params: tp.Tuple[float, ...] = ()) -> float:

    expo = int(eval_poly([np.log2(max(limit[0], min(limit[1],param))) for limit, param in zip(limits,params)]) + 1) 
    n = int(pow(2, max(lb, min(ub, expo))))

    xg, wg = get_gauss_points(n)

    if len(params) > 1:
        params = params[1:]
    y = (b-a)*(xg+1)/2 + a
    return (b-a)/2 * np.sum(wg*f(y, *params))
    