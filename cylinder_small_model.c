
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "kron.h"

const double coeffs[] = {0.0, -0.7222874943870399, 0.17809836968147344, 0.2912592872681659, -0.05855068045133701, -0.027596649032355522, -0.018635419655927805, 0.017849264538213443, -0.037001407500919486, 0.020322693190970995, -0.0013809967483094806, 0.00013860681086138067, 0.00016166294379345094, 0.0003439969033204761, 0.0014501943310794663, 7.623374597408272e-05, -0.00026483079103346467, 0.00019435621694249838, 0.0005908290321576803, -0.0006045485540440709};
const double intercept = 1.7100441737001955;


        
double eval_poly(double var0, double var1, double var2){
    return 1 * coeffs[0] + var0 * coeffs[1] + var1 * coeffs[2] + var2 * coeffs[3] + (var0 * var0) * coeffs[4] + var0 * var1 * coeffs[5] + var0 * var2 * coeffs[6] + (var1 * var1) * coeffs[7] + var1 * var2 * coeffs[8] + (var2 * var2) * coeffs[9] + (var0 * var0 * var0) * coeffs[10] + (var0 * var0) * var1 * coeffs[11] + (var0 * var0) * var2 * coeffs[12] + var0 * (var1 * var1) * coeffs[13] + var0 * var1 * var2 * coeffs[14] + var0 * (var2 * var2) * coeffs[15] + (var1 * var1 * var1) * coeffs[16] + (var1 * var1) * var2 * coeffs[17] + var1 * (var2 * var2) * coeffs[18] + (var2 * var2 * var2) * coeffs[19] + intercept;
}
    

const int lb = 1;
const int ub = 15;
typedef double (*Integrand)(double x, double A, double B);

double integrate(Integrand f, double a, double b, double rtol, double A, double B){


    // Determine the number of points for the Gauss quadrature
    int expo = (int)(eval_poly(log2(rtol), log2(A), log2(B)) + 1);
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