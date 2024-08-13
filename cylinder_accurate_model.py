

import numpy as np
import typing as tp
from collections.abc import Sequence
from kron import get_gauss_points

__all__ = ["integrate"]
coeffs = [ 0.0000000000e+00, 3.1043359934e-01, 4.7157543384e-01,-5.7683081074e-02,
  5.7737285710e-02,-1.4412745461e-02,-2.9930180395e-02,-1.0529071535e-02,
 -3.3014105172e-02, 5.4198827244e-02, 2.5460451076e-03, 4.4166088601e-04,
  6.2352125083e-04,-5.0228100762e-04, 2.2793165725e-03, 1.1431222932e-03,
  2.6942276271e-05, 6.1197456100e-04, 7.7362821863e-04,-1.4548829186e-03]
intercept  = 2.0765714285718566
limits = [(1e-05, 0.1), (1.0, 1000000.0), (1.0, 1000000.0)]


        
def eval_poly(vars: Sequence[float]) -> float:
    return 1 * coeffs[0] + vars[0] * coeffs[1] + vars[1] * coeffs[2] + vars[2] * coeffs[3] + vars[0] ** 2 * coeffs[4] + vars[0] * vars[1] * coeffs[5] + vars[0] * vars[2] * coeffs[6] + vars[1] ** 2 * coeffs[7] + vars[1] * vars[2] * coeffs[8] + vars[2] ** 2 * coeffs[9] + vars[0] ** 3 * coeffs[10] + vars[0] ** 2 * vars[1] * coeffs[11] + vars[0] ** 2 * vars[2] * coeffs[12] + vars[0] * vars[1] ** 2 * coeffs[13] + vars[0] * vars[1] * vars[2] * coeffs[14] + vars[0] * vars[2] ** 2 * coeffs[15] + vars[1] ** 3 * coeffs[16] + vars[1] ** 2 * vars[2] * coeffs[17] + vars[1] * vars[2] ** 2 * coeffs[18] + vars[2] ** 3 * coeffs[19] + intercept
    

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
    