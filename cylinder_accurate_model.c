
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "kron.h"

const double coeffs[] = {-50.32474089283958, -1427824.1774757393, -343263.1848687853, 817848.1078012217, 0.007365850044950771, 196619.00371076705, -168474.0660522161, -0.013319499069662196, -0.023880550140073252, -40502.862084472596, 15966.067211021598, 0.0018660057200263651, -0.0004002928628653257, 0.0034865832284832732, 3838.402571070239, -706.8036707316887, -5.1935075537054696e-05, -3.294412817922421e-05, 7.886090315878391e-06, -0.00016372041136492044, -169.92267053519754, 11.820505730807781, -3.166496753692627e-07, 3.0610826797783375e-06, -2.835469786077738e-06, 1.8181162886321545e-06, 2.311950083822012e-06, 2.841767642297782};
const double intercept = 52.28994722995493;
const double limits[][2] = {{1.0, 1000000.0}, {1.0, 1000000.0}};


        
double eval_poly(double var0, double var1){
    return 1 * coeffs[0] + var0 * coeffs[1] + var1 * coeffs[2] + (var0 * var0) * coeffs[3] + var0 * var1 * coeffs[4] + (var1 * var1) * coeffs[5] + (var0 * var0 * var0) * coeffs[6] + (var0 * var0) * var1 * coeffs[7] + var0 * (var1 * var1) * coeffs[8] + (var1 * var1 * var1) * coeffs[9] + (var0 * var0 * var0 * var0) * coeffs[10] + (var0 * var0 * var0) * var1 * coeffs[11] + (var0 * var0) * (var1 * var1) * coeffs[12] + var0 * (var1 * var1 * var1) * coeffs[13] + (var1 * var1 * var1 * var1) * coeffs[14] + (var0 * var0 * var0 * var0 * var0) * coeffs[15] + (var0 * var0 * var0 * var0) * var1 * coeffs[16] + (var0 * var0 * var0) * (var1 * var1) * coeffs[17] + (var0 * var0) * (var1 * var1 * var1) * coeffs[18] + var0 * (var1 * var1 * var1 * var1) * coeffs[19] + (var1 * var1 * var1 * var1 * var1) * coeffs[20] + (var0 * var0 * var0 * var0 * var0 * var0) * coeffs[21] + (var0 * var0 * var0 * var0 * var0) * var1 * coeffs[22] + (var0 * var0 * var0 * var0) * (var1 * var1) * coeffs[23] + (var0 * var0 * var0) * (var1 * var1 * var1) * coeffs[24] + (var0 * var0) * (var1 * var1 * var1 * var1) * coeffs[25] + var0 * (var1 * var1 * var1 * var1 * var1) * coeffs[26] + (var1 * var1 * var1 * var1 * var1 * var1) * coeffs[27] + intercept;
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