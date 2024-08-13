
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "kron.h"

const double coeffs[] = {0.0, 0.6617372472686641, 0.6645981288759101, 0.006788197308080113, -0.04655747932217508, 0.007214263690975553};
const double intercept = 3.542896397504518;


        
double eval_poly(double var0, double var1){
    return 1 * coeffs[0] + var0 * coeffs[1] + var1 * coeffs[2] + (var0 * var0) * coeffs[3] + var0 * var1 * coeffs[4] + (var1 * var1) * coeffs[5] + intercept;
}
    

const int lb = 1;
const int ub = 15;
typedef double (*Integrand)(double x, double A, double B);

double integrate(Integrand f, double a, double b, double A, double B){


    // Determine the number of points for the Gauss quadrature
    int expo = (int)(eval_poly(log2(A), log2(B)) + 1);
    int n = (int)(pow(2, max(lb, min(ub, expo))));
    
    double *xg, *wg;
    get_gauss_points(n, &xg, &wg);
    
    // Perform the integration
    double sum = 0;
    for (int i = 0; i < n; i++){
        sum += f(a + (b - a) * 0.5 * (xg[i] + 1), A, B) * wg[i];
    }
    sum *= (b - a) * 0.5;
    return sum;
    
}