

import numpy as np
import typing as tp
from collections.abc import Sequence
from kron import get_gauss_points

__all__ = ["integrate"]
coeffs = [ 0.0000000000e+00,-2.0729294048e-01, 1.3543765090e-01,-6.0094918816e-03,
  7.9030412468e-05]
intercept  = 2.995115752982257
limits = [(1.0, 100000.0)]


        
def eval_poly(vars: Sequence[float]) -> float:
    return 1 * coeffs[0] + vars[0] * coeffs[1] + vars[0] ** 2 * coeffs[2] + vars[0] ** 3 * coeffs[3] + vars[0] ** 4 * coeffs[4] + intercept
    

lb: int = 1
ub: int = 15

def integrate(f: tp.Callable[[float, tp.Tuple[float, ...]], float], a: float, b: float, params: tp.Tuple[float, ...] = ()) -> float:

    expo = int(eval_poly([np.log2(max(limit[0], min(limit[1],param))) for limit, param in zip(limits,params)]) + 1)   
    n = int(pow(2, max(lb, min(ub, expo))))

    xg, wg = get_gauss_points(n)

    y = (b-a)*(xg+1)/2 + a
    return (b-a)/2 * np.sum(wg*f(y, *params))
    