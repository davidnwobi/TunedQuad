#include <stdio.h>
#include <stdlib.h>
#include "kron.h"

int main() {
    const double *xg, *wg;
    int n = 256;

    get_gauss_points(n, &xg, &wg);  
    if (xg == NULL || wg == NULL) {
        printf("Error: xg or wg is NULL\n");
        return 1;
    }
    for (int i = 0; i < n; i++) {
        printf("%f \n", *(xg+i));
    }
}

