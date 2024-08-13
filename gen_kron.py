import numpy as np
import numpy.typing as npt


upper_bound  = 15
py_file = '''

import numpy as np
import numpy.typing as npt
import typing as tp

'''
all_gauss = ", ".join([f'"GAUSS_{2**i}_XG", "GAUSS_{2**i}_WG", "GAUSS_{2**i}_XK", "GAUSS_{2**i}_WK"' for i in range(1, upper_bound+1)])

py_file += f'''

__all__ = ["get_gauss_kronrod_points, get_gauss_points"]

'''

for i in range(1, upper_bound+1):
    xk, wk, wg = np.loadtxt(f"{2**i}_mp.txt", unpack=True)
    xk = np.concatenate([-xk[:-1], xk[::-1]] )
    wk = np.concatenate([wk[:-1], wk[::-1]] )
    wg = np.concatenate([wg[:-1], wg[::-1]] )
    xg = xk[1:-1:2]
    wg = wg[1:-1:2]
    py_file += f'''

GAUSS_{2**i}_XG: npt.NDArray[np.float64] = np.array([{", ".join(map(str, xg))}])
GAUSS_{2**i}_WG: npt.NDArray[np.float64] = np.array([{", ".join(map(str, wg))}])
GAUSS_{2**i}_XK: npt.NDArray[np.float64] = np.array([{", ".join(map(str, xk))}])
GAUSS_{2**i}_WK: npt.NDArray[np.float64] = np.array([{", ".join(map(str, wk))}])

'''
py_file += '''

def get_gauss_kronrod_points(n: int) -> tp.Tuple[npt.NDArray[np.float64], 4]:

    match n:
'''

for i in range(1, upper_bound+1):
    py_file += f'''
        case {2**i}:
            return GAUSS_{2**i}_XK, GAUSS_{2**i}_WK, GAUSS_{2**i}_XG, GAUSS_{2**i}_WG
'''

py_file += f'''
def get_gauss_points(n: int) -> tp.Tuple[npt.NDArray[np.float64], 2]:
    
    match n:
'''

for i in range(1, upper_bound+1):
    py_file += f'''
        case {2**i}:
            return GAUSS_{2**i}_XG, GAUSS_{2**i}_WG
'''
with open("kron.py", "w") as f:
    f.write(py_file)