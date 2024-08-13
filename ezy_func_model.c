
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "kron.h"

const double coeffs[] = {0.0, -0.9148286558753346, 0.05613892455508609, -0.09998545138804422, -0.051504723211393426, 0.05518574393180811, -0.0036181641266642695, -0.003373943473984037, -0.0014502169867654235, -0.0021050108055385985};
const double intercept = -0.6157187001159432;
const double limits[][2] = {{1e-05, 0.1}, {1.0, 100000.0}};


        
double eval_poly(double var0, double var1){
    return 1 * coeffs[0] + var0 * coeffs[1] + var1 * coeffs[2] + (var0 * var0) * coeffs[3] + var0 * var1 * coeffs[4] + (var1 * var1) * coeffs[5] + (var0 * var0 * var0) * coeffs[6] + (var0 * var0) * var1 * coeffs[7] + var0 * (var1 * var1) * coeffs[8] + (var1 * var1 * var1) * coeffs[9] + intercept;
}
    

const int lb = 1;
const int ub = 15;
typedef double (*Integrand)(double x, double b);

double integrate(Integrand f, double a, double b, double rtol, double b){


    // Determine the number of points for the Gauss quadrature
    int expo = (int)(eval_poly(log2(max(limits[0][0],min(limits[0][1], rtol))), log2(max(limits[1][0],min(limits[1][1], b)))) + 1);
    int n = (int)(pow(2, max(lb, min(ub, expo))));
    
    double *xg, *wg;
    get_gauss_points(n, &xg, &wg);
    
    // Perform the integration
    double sum = 0;
    for (int i = 0; i < n; i++){
        sum += f(a + (b - a) * 0.5 * (xg[i] + 1), b) * wg[i];
    }
    sum *= (b - a) * 0.5;
    return sum;
    
}