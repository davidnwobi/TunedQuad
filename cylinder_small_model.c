
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "kron.h"

const double coeffs[] = {-6.78964299561582e-12, 0.24866118618371216, 0.20651964137842968, 2.129801543460097e-12, 4.796024688502598e-13, 0.08284075835016554, -0.06165197984475282, 9.037237104492224e-14, -1.1055219240052594e-13, 0.0804684009489045, -2.0702883851697607e-13, -1.1610677697060368e-13, -3.3363275250136715e-13, -9.685503267464757e-15, -6.099981630924844e-14, -0.0029599668754413666, -0.0008069992834263992, -7.813194535799539e-15, 1.2975731600306517e-14, -0.0020598953153576228, 2.431214951581495e-15, 1.790234627208065e-15, 3.643786661289283e-15, -3.2146594414195206e-15, -2.467210463708014e-15, -0.0016235444413780602, 1.1770342729991445e-14, 6.451002926288751e-16, 6.32393443167345e-15, 6.978575703420198e-15, 5.467848396278896e-15, 1.8832158055204218e-14, 1.0712324039971466e-15, -3.715777685542321e-15, -5.30825383648903e-16, -1.3750599006934205e-05, 0.00021267593131354956, 7.979727989493313e-17, -2.3592239273284576e-16, -0.000244985161905964, -1.5764299587939234e-16, -4.85722573273506e-17, 2.2898349882893854e-16, 8.326672684688674e-17, 1.5265566588595902e-16, 0.0002545825187645519, -1.491862189340054e-16, -9.71445146547012e-17, -1.5265566588595902e-16, 3.2612801348363973e-16, -2.498001805406602e-16, -1.491862189340054e-16, -1.431146867680866e-17, -3.2959746043559335e-17, 9.540979117872439e-17, -6.875299503660004e-05, -1.0234868508263162e-16, 9.80118763926896e-17, -1.8041124150158794e-16, -3.469446951953614e-16, -2.0816681711721685e-17, 5.204170427930421e-18, -8.673617379884035e-17, 5.898059818321144e-17, -9.367506770274758e-17, -4.163336342344337e-16, -1.3183898417423734e-16, 2.1510571102112408e-16, -1.5612511283791264e-17, 2.220446049250313e-16};
const double intercept = 3.628253483993788;


        
double eval_poly(const double var0, const double var1, const double var2, const double var3){
    return 1 * coeffs[0] + var0 * coeffs[1] + var1 * coeffs[2] + var2 * coeffs[3] + var3 * coeffs[4] + (var0 * var0) * coeffs[5] + var0 * var1 * coeffs[6] + var0 * var2 * coeffs[7] + var0 * var3 * coeffs[8] + (var1 * var1) * coeffs[9] + var1 * var2 * coeffs[10] + var1 * var3 * coeffs[11] + (var2 * var2) * coeffs[12] + var2 * var3 * coeffs[13] + (var3 * var3) * coeffs[14] + (var0 * var0 * var0) * coeffs[15] + (var0 * var0) * var1 * coeffs[16] + (var0 * var0) * var2 * coeffs[17] + (var0 * var0) * var3 * coeffs[18] + var0 * (var1 * var1) * coeffs[19] + var0 * var1 * var2 * coeffs[20] + var0 * var1 * var3 * coeffs[21] + var0 * (var2 * var2) * coeffs[22] + var0 * var2 * var3 * coeffs[23] + var0 * (var3 * var3) * coeffs[24] + (var1 * var1 * var1) * coeffs[25] + (var1 * var1) * var2 * coeffs[26] + (var1 * var1) * var3 * coeffs[27] + var1 * (var2 * var2) * coeffs[28] + var1 * var2 * var3 * coeffs[29] + var1 * (var3 * var3) * coeffs[30] + (var2 * var2 * var2) * coeffs[31] + (var2 * var2) * var3 * coeffs[32] + var2 * (var3 * var3) * coeffs[33] + (var3 * var3 * var3) * coeffs[34] + (var0 * var0 * var0 * var0) * coeffs[35] + (var0 * var0 * var0) * var1 * coeffs[36] + (var0 * var0 * var0) * var2 * coeffs[37] + (var0 * var0 * var0) * var3 * coeffs[38] + (var0 * var0) * (var1 * var1) * coeffs[39] + (var0 * var0) * var1 * var2 * coeffs[40] + (var0 * var0) * var1 * var3 * coeffs[41] + (var0 * var0) * (var2 * var2) * coeffs[42] + (var0 * var0) * var2 * var3 * coeffs[43] + (var0 * var0) * (var3 * var3) * coeffs[44] + var0 * (var1 * var1 * var1) * coeffs[45] + var0 * (var1 * var1) * var2 * coeffs[46] + var0 * (var1 * var1) * var3 * coeffs[47] + var0 * var1 * (var2 * var2) * coeffs[48] + var0 * var1 * var2 * var3 * coeffs[49] + var0 * var1 * (var3 * var3) * coeffs[50] + var0 * (var2 * var2 * var2) * coeffs[51] + var0 * (var2 * var2) * var3 * coeffs[52] + var0 * var2 * (var3 * var3) * coeffs[53] + var0 * (var3 * var3 * var3) * coeffs[54] + (var1 * var1 * var1 * var1) * coeffs[55] + (var1 * var1 * var1) * var2 * coeffs[56] + (var1 * var1 * var1) * var3 * coeffs[57] + (var1 * var1) * (var2 * var2) * coeffs[58] + (var1 * var1) * var2 * var3 * coeffs[59] + (var1 * var1) * (var3 * var3) * coeffs[60] + var1 * (var2 * var2 * var2) * coeffs[61] + var1 * (var2 * var2) * var3 * coeffs[62] + var1 * var2 * (var3 * var3) * coeffs[63] + var1 * (var3 * var3 * var3) * coeffs[64] + (var2 * var2 * var2 * var2) * coeffs[65] + (var2 * var2 * var2) * var3 * coeffs[66] + (var2 * var2) * (var3 * var3) * coeffs[67] + var2 * (var3 * var3 * var3) * coeffs[68] + (var3 * var3 * var3 * var3) * coeffs[69] + intercept;
}
    

const int lb = 1;
const int ub = 15;
typedef double (*Integrand)(double x, double A, double B, double C, double D);

double integrate(Integrand f, double a, double b, double A, double B, double C, double D){


    // Determine the number of points for the Gauss quadrature
    int expo = (int)(eval_poly(log2(A), log2(B), log2(C), log2(D)) + 1);
    int n = (int)(pow(2, max(lb, min(ub, expo))));
    
    double *xg, *wg;
    get_gauss_points(n, &xg, &wg);
    
    // Perform the integration
    double sum = 0;
    for (int i = 0; i < n; i++){
        sum += f(a + (b - a)*0.5 * (xg[i]+1), A, B, C, D) * wg[i];
    }
    sum *= (b - a) * 0.5;
    return sum;
    
}