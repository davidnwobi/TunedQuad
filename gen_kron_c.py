import numpy as np
upper_bound  = 15

h_file = '''
#ifndef KRON_H
#define KRON_H

void get_gauss_points(int n, const double **xg, const double **wg);

#endif // KRON_H
'''

with open("kron.h", "w") as f:
    f.write(h_file)
    
c_file = '''

#include <stdio.h>
#include <stdlib.h>
#include "kron.h"
'''


for i in range(1, upper_bound+1):
    xk, wk, wg = np.loadtxt(f"{2**i}_mp.txt", unpack=True)
    xk = np.concatenate([-xk[:-1], xk[::-1]] )
    wk = np.concatenate([wk[:-1], wk[::-1]] )
    wg = np.concatenate([wg[:-1], wg[::-1]] )
    xg = xk[1:-1:2]
    wg = wg[1:-1:2]
    c_file += f'''

const double GAUSS_{2**i}_XG[] = {{{", ".join(map(str, xg))}}};
const double GAUSS_{2**i}_WG[] = {{{", ".join(map(str, wg))}}};

'''

    
c_file += '''

void get_gauss_points(int n, const double **xg, const double **wg){
    switch(n){

'''
for i in range(1, upper_bound+1):
    c_file += f'''
        case {2**i}:
            *xg = GAUSS_{2**i}_XG;
            *wg = GAUSS_{2**i}_WG;
            break;
'''

c_file += '''

    default:
        printf("Invalid n value\\n");
        *xg = NULL;
        *wg = NULL;
        break;

    }
}
'''

with open("kron.c", "w") as f:
    f.write(c_file)
    
