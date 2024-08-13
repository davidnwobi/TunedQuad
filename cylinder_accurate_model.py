

import numpy as np
import typing as tp
from collections.abc import Sequence
from kron import get_gauss_points

__all__ = ["integrate"]
coeffs = [-5.0324740893e+01,-1.4278241775e+06,-3.4326318487e+05, 8.1784810780e+05,
  7.3658500450e-03, 1.9661900371e+05,-1.6847406605e+05,-1.3319499070e-02,
 -2.3880550140e-02,-4.0502862084e+04, 1.5966067211e+04, 1.8660057200e-03,
 -4.0029286287e-04, 3.4865832285e-03, 3.8384025711e+03,-7.0680367073e+02,
 -5.1935075537e-05,-3.2944128179e-05, 7.8860903159e-06,-1.6372041136e-04,
 -1.6992267054e+02, 1.1820505731e+01,-3.1664967537e-07, 3.0610826798e-06,
 -2.8354697861e-06, 1.8181162886e-06, 2.3119500838e-06, 2.8417676423e+00]
intercept  = 52.28994722995493
limits = [(1.0, 1000000.0), (1.0, 1000000.0)]


        
def eval_poly(vars: Sequence[float]) -> float:
    return 1 * coeffs[0] + vars[0] * coeffs[1] + vars[1] * coeffs[2] + vars[0] ** 2 * coeffs[3] + vars[0] * vars[1] * coeffs[4] + vars[1] ** 2 * coeffs[5] + vars[0] ** 3 * coeffs[6] + vars[0] ** 2 * vars[1] * coeffs[7] + vars[0] * vars[1] ** 2 * coeffs[8] + vars[1] ** 3 * coeffs[9] + vars[0] ** 4 * coeffs[10] + vars[0] ** 3 * vars[1] * coeffs[11] + vars[0] ** 2 * vars[1] ** 2 * coeffs[12] + vars[0] * vars[1] ** 3 * coeffs[13] + vars[1] ** 4 * coeffs[14] + vars[0] ** 5 * coeffs[15] + vars[0] ** 4 * vars[1] * coeffs[16] + vars[0] ** 3 * vars[1] ** 2 * coeffs[17] + vars[0] ** 2 * vars[1] ** 3 * coeffs[18] + vars[0] * vars[1] ** 4 * coeffs[19] + vars[1] ** 5 * coeffs[20] + vars[0] ** 6 * coeffs[21] + vars[0] ** 5 * vars[1] * coeffs[22] + vars[0] ** 4 * vars[1] ** 2 * coeffs[23] + vars[0] ** 3 * vars[1] ** 3 * coeffs[24] + vars[0] ** 2 * vars[1] ** 4 * coeffs[25] + vars[0] * vars[1] ** 5 * coeffs[26] + vars[1] ** 6 * coeffs[27] + intercept
    

lb: int = 1
ub: int = 15

def integrate(f: tp.Callable[[float, tp.Tuple[float, ...]], float], a: float, b: float, params: tp.Tuple[float, ...] = ()) -> float:

    expo = int(eval_poly([np.log2(max(limit[0], min(limit[1],param))) for limit, param in zip(limits,params)]) + 1)   
    n = int(pow(2, max(lb, min(ub, expo))))

    xg, wg = get_gauss_points(n)

    y = (b-a)*(xg+1)/2 + a
    return (b-a)/2 * np.sum(wg*f(y, *params))
    