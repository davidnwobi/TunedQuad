

import numpy as np
import typing as tp
from collections.abc import Sequence
from kron import get_gauss_points

__all__ = ["integrate"]
coeffs = [ 0.0000000000e+00,-7.2228749439e-01, 1.7809836968e-01, 2.9125928727e-01,
 -5.8550680451e-02,-2.7596649032e-02,-1.8635419656e-02, 1.7849264538e-02,
 -3.7001407501e-02, 2.0322693191e-02,-1.3809967483e-03, 1.3860681086e-04,
  1.6166294379e-04, 3.4399690332e-04, 1.4501943311e-03, 7.6233745974e-05,
 -2.6483079103e-04, 1.9435621694e-04, 5.9082903216e-04,-6.0454855404e-04]
intercept  = 1.7100441737001955


        
def eval_poly(vars: Sequence[float]) -> float:
    return 1 * coeffs[0] + vars[0] * coeffs[1] + vars[1] * coeffs[2] + vars[2] * coeffs[3] + vars[0] ** 2 * coeffs[4] + vars[0] * vars[1] * coeffs[5] + vars[0] * vars[2] * coeffs[6] + vars[1] ** 2 * coeffs[7] + vars[1] * vars[2] * coeffs[8] + vars[2] ** 2 * coeffs[9] + vars[0] ** 3 * coeffs[10] + vars[0] ** 2 * vars[1] * coeffs[11] + vars[0] ** 2 * vars[2] * coeffs[12] + vars[0] * vars[1] ** 2 * coeffs[13] + vars[0] * vars[1] * vars[2] * coeffs[14] + vars[0] * vars[2] ** 2 * coeffs[15] + vars[1] ** 3 * coeffs[16] + vars[1] ** 2 * vars[2] * coeffs[17] + vars[1] * vars[2] ** 2 * coeffs[18] + vars[2] ** 3 * coeffs[19] + intercept
    

lb: int = 1
ub: int = 15

def integrate(f: tp.Callable[[float, tp.Tuple[float, ...]], float], a: float, b: float, params: tp.Tuple[float, ...] = ()) -> float:

    expo = int(eval_poly([np.log2(param) for param in params]) + 1)   
    n = int(pow(2, max(lb, min(ub, expo))))

    xg, wg = get_gauss_points(n)

    if len(params) > 1:
        _, params = params[0], params[1:]
    y = (b-a)*(xg+1)/2 + a
    return (b-a)/2 * np.sum(wg*f(y, *params))
    