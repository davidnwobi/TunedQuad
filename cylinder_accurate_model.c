
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "kron.h"

const double coeffs[] = {0.0, 0.31043359933842984, 0.47157543384234224, -0.05768308107386915, 0.05773728571017609, -0.014412745461281792, -0.02993018039503277, -0.010529071534583826, -0.03301410517237654, 0.05419882724359467, 0.0025460451075696444, 0.00044166088600665126, 0.0006235212508331179, -0.0005022810076154822, 0.002279316572490046, 0.0011431222931939857, 2.6942276270633497e-05, 0.0006119745610029953, 0.0007736282186263793, -0.0014548829186104562};
const double intercept = 2.0765714285718566;
const double limits[][2] = {{1e-05, 0.1}, {1.0, 1000000.0}, {1.0, 1000000.0}};


        
double eval_poly(double var0, double var1, double var2){
    return 1 * coeffs[0] + var0 * coeffs[1] + var1 * coeffs[2] + var2 * coeffs[3] + (var0 * var0) * coeffs[4] + var0 * var1 * coeffs[5] + var0 * var2 * coeffs[6] + (var1 * var1) * coeffs[7] + var1 * var2 * coeffs[8] + (var2 * var2) * coeffs[9] + (var0 * var0 * var0) * coeffs[10] + (var0 * var0) * var1 * coeffs[11] + (var0 * var0) * var2 * coeffs[12] + var0 * (var1 * var1) * coeffs[13] + var0 * var1 * var2 * coeffs[14] + var0 * (var2 * var2) * coeffs[15] + (var1 * var1 * var1) * coeffs[16] + (var1 * var1) * var2 * coeffs[17] + var1 * (var2 * var2) * coeffs[18] + (var2 * var2 * var2) * coeffs[19] + intercept;
}
    

const int lb = 1;
const int ub = 15;
typedef double (*Integrand)(double x, double A, double B);

double integrate(Integrand f, double a, double b, double rtol, double A, double B){


    // Determine the number of points for the Gauss quadrature
    int expo = (int)(eval_poly(log2(max(limits[0][0],min(limits[0][1], rtol))), log2(max(limits[1][0],min(limits[1][1], A))), log2(max(limits[2][0],min(limits[2][1], B)))) + 1);
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