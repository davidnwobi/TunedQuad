import numpy as np
upper_bound  = 15
py_file = '''

import numpy as np

'''
all_gauss = ", ".join([f'"GAUSS_{2**i}_XG", "GAUSS_{2**i}_WG", "GAUSS_{2**i}_XK", "GAUSS_{2**i}_WK"' for i in range(1, upper_bound+1)])

py_file += f'''

__all__ = ["get_kronrod, get_kronrod_w_higher_order"]

'''

for i in range(1, upper_bound+1):
    xk, wk, wg = np.loadtxt(f"{2**i}_mp.txt", unpack=True)
    xk = np.concatenate([-xk[:-1], xk[::-1]] )
    wk = np.concatenate([wk[:-1], wk[::-1]] )
    wg = np.concatenate([wg[:-1], wg[::-1]] )
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