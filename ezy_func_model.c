
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "kron.h"
#define min(a,b) ((a) < (b) ? (a) : (b))
#define max(a,b) ((a) > (b) ? (a) : (b))

const double coeffs[] = {0.0, -0.20729294048122573, 0.13543765089637877, -0.006009491881603881, 7.903041246754006e-05};
const double intercept = 2.995115752982257;
const double limits[][2] = {{1.0, 100000.0}};


        
double eval_poly(double var0){
    return 1 * coeffs[0] + var0 * coeffs[1] + (var0 * var0) * coeffs[2] + (var0 * var0 * var0) * coeffs[3] + (var0 * var0 * var0 * var0) * coeffs[4] + intercept;
}
    

const int lb = 1;
const int ub = 15;
typedef double (*Integrand)(double x, double B);

double integrate(Integrand f, double a, double b, double B){


    // Determine the number of points for the Gauss quadrature
    int expo = (int)(eval_poly(log2(B)) + 1);
    int n = (int)(pow(2, max(lb, min(ub, expo))));
    
    double *xg, *wg;
    get_gauss_points(n, &xg, &wg);
    
    // Perform the integration
    double sum = 0;
    for (int i = 0; i < n; i++){
        sum += f(a + (b - a) * 0.5 * (xg[i] + 1), B) * wg[i];
    }
    sum *= (b - a) * 0.5;
    return sum;
    
}