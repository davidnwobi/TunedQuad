/* This is generated code 
*/

#include <math.h>



void get_kronrod(int n, const double **xg, const double **wg);



int max(int a, int b)
{
    return a > b ? a : b;
}

int min(int a, int b)
{
    return a < b ? a : b;
}
int lb = 1;
int ub = 15;

double coeffs[] = {1, 2, 3, 4, 5};
double intercept = 1.;

double eval_poly(const double var0, const double var1); 

typedef double (*Integrand)(double x, double A, double B);

double integrate(Integrand f, double a, double b, double A, double B){
    double Alog2 = log2(A);
    double Blog2 = log2(B);

    int expo = (int)(eval_poly(Alog2, Blog2)) + 1;
    int n = (int)(pow(2, max(lb, min(ub, expo))));

    double *xg, *wg;
    get_kronrod(n, &xg, &wg);

    double sum = 0;
    for (int i = 0; i < n; i++){
        sum += f(a + (b - a)*0.5 * xg[i], A, B) * wg[i];
    }
    sum *= (b - a) * 0.5;
    return sum;

}