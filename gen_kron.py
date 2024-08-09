from kronrod import get_kronrod_h5

upper_bound  = 9
py_file = '''

import numpy as np

'''
all_gauss = ", ".join([f'"GAUSS_{2**i}_XG", "GAUSS_{2**i}_WG", "GAUSS_{2**i}_XK", "GAUSS_{2**i}_WK"' for i in range(1, upper_bound+1)])

py_file += f'''

__all__ = ["get_kronrod, get_kronrod_w_higher_order"]

'''

for i in range(1, upper_bound+1):
    xk, wk, wg = get_kronrod_h5(2**i)
    xg = xk[1:-1:2]
    wg = wg[1:-1:2]
    py_file += f'''

GAUSS_{2**i}_XG = np.array([{", ".join(map(str, xg))}])
GAUSS_{2**i}_WG = np.array([{", ".join(map(str, wg))}])
GAUSS_{2**i}_XK = np.array([{", ".join(map(str, xk))}])
GAUSS_{2**i}_WK = np.array([{", ".join(map(str, wk))}])

'''
py_file += '''

def get_kronrod_w_higher_order(n):
    return eval(f'GAUSS_{n}_XG'), eval(f'GAUSS_{n}_WG'), eval(f'GAUSS_{n}_XK'), eval(f'GAUSS_{n}_WK')

def get_kronrod(n):
    return eval(f'GAUSS_{n}_XG'), eval(f'GAUSS_{n}_WG')

'''
with open("kron.py", "w") as f:
    f.write(py_file)